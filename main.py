"""Jarvis voice-assistant entry point.

Passively wait for "hey Jarvis", then listen → think → speak, and loop.

Run from the project root::

    python main.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from audio.listener import listen_and_transcribe, preload_whisper, wait_for_wake_word
from audio.speaker import preload_voice, speak
from brain.llm import JarvisReply, ask_jarvis
from brain.tools import dispatch
from config import load_config
from ui.hud import set_mode
from ui.server import start_ui

MAX_EXCHANGES = 10
MAX_TOOL_ROUNDS = 5
# Pause after TTS so playback isn't picked up as another "hey jarvis".
POST_SPEAK_COOLDOWN_S = 0.5


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def _run_tools(reply: JarvisReply) -> list[dict[str, Any]]:
    """Execute each tool_use block; failures become error strings for Claude."""
    results: list[dict[str, Any]] = []
    for call in reply.tool_calls:
        try:
            output = dispatch(call.name, call.input)
            print(f"  tool {call.name}({call.input}) -> {output}", flush=True)
        except Exception as exc:
            output = f"Tool {call.name!r} failed: {exc}"
            print(f"  {output}", file=sys.stderr, flush=True)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": _stringify_tool_result(output),
            }
        )
    return results


def _think(heard: str, history: list[dict[str, Any]]) -> str:
    """Ask Claude, run tools if requested, return the spoken reply."""
    extra: list[dict[str, Any]] = []
    reply = ask_jarvis(heard, history, extra_messages=extra or None)

    for _ in range(MAX_TOOL_ROUNDS):
        if not reply.tool_calls:
            break
        print("Jarvis is using a tool…", flush=True)
        extra.append(reply.assistant_message)
        extra.append({"role": "user", "content": _run_tools(reply)})
        reply = ask_jarvis(heard, history, extra_messages=extra)

    if reply.tool_calls:
        return reply.text or "I wasn't able to finish that tool request."
    return reply.text


def main() -> None:
    """Start the assistant and block on the interaction loop."""
    try:
        cfg = load_config()
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    history: list[dict[str, Any]] = []
    try:
        start_ui(open_browser=True)
    except OSError as exc:
        print(f"UI server failed to start: {exc}", file=sys.stderr, flush=True)

    print("Loading speech models…", flush=True)
    set_mode("thinking", heard="", reply="Loading…")
    try:
        preload_whisper(cfg.whisper_model)
        preload_voice(cfg.piper_voice)
    except Exception as exc:
        print(f"Model preload failed (will retry on first use): {exc}", file=sys.stderr, flush=True)

    print("Jarvis is ready. Say 'hey Jarvis', then speak a command.", flush=True)
    print("I'll stop recording when you pause. Ctrl+C to quit.", flush=True)
    set_mode("listening", heard="", reply="")

    while True:
        set_mode("listening")
        try:
            wait_for_wake_word(threshold=cfg.wake_threshold, sample_rate=cfg.sample_rate)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", flush=True)
            set_mode("idle")
            return
        except Exception as exc:
            print(f"Wake-word listener failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(1)
            continue

        set_mode("recording")
        try:
            heard = listen_and_transcribe(
                model_size=cfg.whisper_model,
                sample_rate=cfg.sample_rate,
                wait_for_start=False,
                stop_on_silence=True,
            )
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", flush=True)
            set_mode("idle")
            return
        except Exception as exc:
            print(f"Listening failed: {exc}", file=sys.stderr, flush=True)
            continue

        heard = (heard or "").strip()
        if not heard:
            print("Heard nothing. Try again.", flush=True)
            set_mode("listening", reply="I didn't catch that.")
            continue

        print(f"You: {heard}", flush=True)
        set_mode("thinking", heard=heard, reply="")

        try:
            answer = _think(heard, history)
        except Exception as exc:
            print(f"LLM failed: {exc}", file=sys.stderr, flush=True)
            msg = str(exc).lower()
            if "credit balance is too low" in msg or "purchase credits" in msg:
                answer = (
                    "I heard you, but I cannot reach Claude. "
                    "Your Anthropic API credit balance is too low. "
                    "Add credits in Anthropic Plans and Billing, then try again."
                )
            else:
                answer = "I heard you, but I had trouble thinking. Please try again."
            print(f"Jarvis: {answer}", flush=True)
            set_mode("speaking", heard=heard, reply=answer)
            try:
                speak(answer, voice=cfg.piper_voice)
            except Exception as speak_exc:
                print(f"Speech failed: {speak_exc}", file=sys.stderr, flush=True)
            time.sleep(POST_SPEAK_COOLDOWN_S)
            continue

        answer = (answer or "").strip()
        if not answer:
            print("Jarvis had nothing to say.", flush=True)
            set_mode("listening")
            continue

        print(f"Jarvis: {answer}", flush=True)
        set_mode("speaking", heard=heard, reply=answer)

        history.append({"role": "user", "content": heard})
        history.append({"role": "assistant", "content": answer})
        history = history[-(MAX_EXCHANGES * 2) :]

        try:
            speak(answer, voice=cfg.piper_voice)
        except Exception as exc:
            print(f"Speech failed: {exc}", file=sys.stderr, flush=True)
        time.sleep(POST_SPEAK_COOLDOWN_S)


if __name__ == "__main__":
    main()
