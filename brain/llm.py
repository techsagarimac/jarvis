"""Anthropic Claude client and tool-calling loop.

``ask_jarvis`` sends a user utterance plus rolling history to Claude and
returns either final text or tool_use blocks for ``brain.tools.dispatch``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anthropic

from brain.tools import TOOL_SCHEMAS
from config import load_config
from memory.store import get_all_facts

SYSTEM_PROMPT = (
    "You are Jarvis, a concise local voice assistant. "
    "Reply in short spoken English, no markdown or bullet lists. "
    "Use tools when they help (time, remembering or recalling facts). "
    "When the user shares something durable (their name, preferences, "
    "people, places), call remember_fact so it persists. "
    "After tools run, give a natural final answer the user can hear."
)

MAX_TOKENS = 1024

_client: anthropic.Anthropic | None = None
_model: str | None = None


class LLM:
    """Thin wrapper around the Anthropic Messages API with tool use.

    Prefer the module-level ``ask_jarvis`` function used by ``main.py``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> None:
        raise NotImplementedError

    def ask(self, user_text: str, history: list[dict] | None = None) -> str:
        """Send ``user_text`` through Claude, run any tool calls, return the reply."""
        raise NotImplementedError


@dataclass
class ToolCall:
    """One Claude ``tool_use`` block."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class JarvisReply:
    """One Messages API round-trip.

    Attributes:
        text: Concatenated text blocks (may be empty when only tools ran).
        tool_calls: ``tool_use`` blocks, if any.
        assistant_message: History-ready ``{"role": "assistant", "content": ...}``.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)


def _get_client() -> tuple[anthropic.Anthropic, str]:
    global _client, _model
    if _client is None or _model is None:
        cfg = load_config()
        _client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        _model = cfg.anthropic_model
    return _client, _model


def _blocks_to_payload(content: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert SDK content blocks into JSON the API will accept on the next turn."""
    payload: list[dict[str, Any]] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            payload.append({"type": "text", "text": block.text})
        elif btype == "tool_use":
            payload.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input) if block.input else {},
                }
            )
    return payload


def _system_prompt() -> str:
    """Base persona plus any facts currently in SQLite memory."""
    facts = get_all_facts()
    if not facts:
        return SYSTEM_PROMPT
    lines = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Known facts about the user. Treat these as true and use them "
        "when they matter; do not contradict them or invent extras:\n"
        f"{lines}"
    )


def ask_jarvis(
    user_text: str,
    history: list[dict[str, Any]] | None = None,
    extra_messages: list[dict[str, Any]] | None = None,
) -> JarvisReply:
    """Send ``user_text`` to Claude with optional rolling history.

    This is a single API round-trip. If the reply contains ``tool_calls``,
    the caller should ``dispatch`` them, append those results via
    ``extra_messages``, and call again for the spoken answer.

    Args:
        user_text: Latest transcribed user utterance.
        history: Prior ``user`` / ``assistant`` messages (last 10 exchanges).
        extra_messages: In-turn follow-ups (assistant ``tool_use`` + user
            ``tool_result``) for the tool loop.
    """
    client, model = _get_client()
    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": user_text})
    if extra_messages:
        messages.extend(extra_messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": _system_prompt(),
        "messages": messages,
    }
    if TOOL_SCHEMAS:
        kwargs["tools"] = TOOL_SCHEMAS

    response = client.messages.create(**kwargs)

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text" and getattr(block, "text", None):
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    input=dict(block.input) if block.input else {},
                )
            )

    content_payload = _blocks_to_payload(response.content)
    return JarvisReply(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        assistant_message={"role": "assistant", "content": content_payload},
    )
