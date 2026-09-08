"""Microphone input, wake-word detection, and speech-to-text.

``wait_for_wake_word`` passively listens with openWakeWord until "hey
jarvis" is heard. ``listen_and_transcribe`` then records a command
(Enter to stop), writes a temporary WAV, and transcribes with
faster-whisper.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Iterator

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as OpenWakeWordModel
from openwakeword.utils import download_models

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "base"
# openWakeWord has no standalone "Jarvis" checkpoint. The closest official
# model is hey_jarvis, trained on the phrase "hey jarvis".
WAKE_MODEL_NAME = "hey_jarvis"
WAKE_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz, as required by openWakeWord
DEFAULT_WAKE_THRESHOLD = 0.5
SPEECH_RMS = 0.015
SILENCE_SECONDS = 1.0
MAX_RECORD_SECONDS = 12.0

_whisper_models: dict[str, WhisperModel] = {}
_wake_model: OpenWakeWordModel | None = None


class Listener:
    """Capture speech from the default (or configured) microphone.

    Args:
        model_size: faster-whisper checkpoint name (e.g. ``"base"``).
        wake_word: Phrase that arms full utterance capture.
        sample_rate: Capture rate in Hz; Whisper models expect 16_000.
        device: Optional sounddevice input device index or name.
    """

    def __init__(
        self,
        model_size: str = "base",
        wake_word: str = "jarvis",
        sample_rate: int = 16_000,
        device: int | str | None = None,
    ) -> None:
        raise NotImplementedError

    def listen(self) -> Iterator[str]:
        """Block until the wake word, then yield transcribed user utterances.

        Each yielded string is one complete command after the wake word,
        with leading/trailing silence already trimmed.
        """
        raise NotImplementedError

    def transcribe(self, audio: object) -> str:
        """Run faster-whisper on a numpy PCM buffer and return the text."""
        raise NotImplementedError


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 PCM in ``[-1, 1]`` as a 16-bit WAV file."""
    if audio.ndim > 1:
        audio = audio[:, 0]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _get_wake_model() -> OpenWakeWordModel:
    """Load and cache the hey_jarvis openWakeWord model (ONNX).

    Downloads the official checkpoint plus feature/VAD models on first use.
    ONNX is used instead of TFLite so we share the project's onnxruntime
    pin and avoid tflite-runtime (Linux-only extra on openwakeword 0.6.0).
    """
    global _wake_model
    if _wake_model is None:
        print(f"Loading openWakeWord '{WAKE_MODEL_NAME}' (ONNX)…", flush=True)
        download_models(model_names=[WAKE_MODEL_NAME])
        _wake_model = OpenWakeWordModel(
            wakeword_models=[WAKE_MODEL_NAME],
            inference_framework="onnx",
        )
    return _wake_model


def wait_for_wake_word(
    threshold: float = DEFAULT_WAKE_THRESHOLD,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Block until the default mic hears the hey_jarvis wake phrase.

    Streams 80 ms int16 chunks into openWakeWord and returns when any
    model score is at or above ``threshold`` (library default: 0.5).
    """
    oww = _get_wake_model()
    oww.reset()
    print("Listening for 'hey Jarvis'…", flush=True)
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=WAKE_CHUNK_SAMPLES,
    ) as stream:
        while True:
            frames, overflowed = stream.read(WAKE_CHUNK_SAMPLES)
            if overflowed:
                print("Mic overflow while waiting for wake word.", file=sys.stderr, flush=True)
            audio = np.ascontiguousarray(frames.reshape(-1), dtype=np.int16)
            scores = oww.predict(audio)
            if any(float(score) >= threshold for score in scores.values()):
                hit = ", ".join(
                    f"{name}={float(score):.2f}"
                    for name, score in scores.items()
                    if float(score) >= threshold
                )
                print(f"Wake word detected ({hit}).", flush=True)
                return


def _get_whisper(model_size: str) -> WhisperModel:
    """Load and cache a faster-whisper model (first call may download weights)."""
    if model_size not in _whisper_models:
        print(f"Loading faster-whisper '{model_size}' (CPU, int8)…", flush=True)
        _whisper_models[model_size] = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    return _whisper_models[model_size]


def preload_whisper(model_size: str = DEFAULT_MODEL) -> None:
    """Load the Whisper model before the first command so recording can start immediately."""
    _get_whisper(model_size)


def listen_and_transcribe(
    model_size: str = DEFAULT_MODEL,
    sample_rate: int = SAMPLE_RATE,
    wait_for_start: bool = True,
    stop_on_silence: bool = False,
    silence_seconds: float = SILENCE_SECONDS,
    max_seconds: float = MAX_RECORD_SECONDS,
) -> str:
    """Record from the default mic and return a faster-whisper transcript.

    Press Enter to start recording (unless ``wait_for_start`` is false).
    Stop with Enter, or automatically after a pause when ``stop_on_silence``
    is true (used after the wake word so Jarvis can reply without a keypress).
    """
    model = _get_whisper(model_size)

    if wait_for_start:
        input("Press Enter to start recording…")

    chunks: list[np.ndarray] = []
    heard_speech = False
    silent_samples = 0
    total_samples = 0
    last_audio = time.monotonic()
    stop = threading.Event()

    def _on_audio(indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        nonlocal heard_speech, silent_samples, total_samples, last_audio
        if status:
            print(status, file=sys.stderr, flush=True)
        chunks.append(indata.copy())
        last_audio = time.monotonic()
        total_samples += indata.shape[0]
        rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
        if rms >= SPEECH_RMS:
            heard_speech = True
            silent_samples = 0
        elif heard_speech:
            silent_samples += indata.shape[0]
            if silent_samples / sample_rate >= silence_seconds:
                stop.set()
        if total_samples / sample_rate >= max_seconds:
            stop.set()

    if stop_on_silence:
        print("Recording… speak now. I'll stop when you pause.", flush=True)
    else:
        print("Recording. Press Enter to stop.", flush=True)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=_on_audio,
    ):
        if stop_on_silence:
            deadline = time.monotonic() + max_seconds
            while not stop.is_set() and time.monotonic() < deadline:
                if heard_speech and (time.monotonic() - last_audio) >= silence_seconds:
                    break
                if not heard_speech and (time.monotonic() - last_audio) >= max(2.0, silence_seconds):
                    break
                stop.wait(0.05)
        else:
            try:
                input()
            except EOFError:
                pass

    if not chunks:
        print("No audio captured.", flush=True)
        return ""

    audio = np.concatenate(chunks, axis=0)
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        _write_wav(wav_path, audio, sample_rate)
        print(f"Saved {audio.shape[0] / sample_rate:.1f}s to {wav_path}", flush=True)

        segments, info = model.transcribe(wav_path, language="en")
        text = "".join(segment.text for segment in segments).strip()
        print(f"Detected language: {info.language} ({info.language_probability:.2f})", flush=True)
        return text
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if mode == "wake":
        wait_for_wake_word()
        print("Wake word received.")
        sys.exit(0)
    if mode not in {"base", "small"}:
        print("Usage: python audio/listener.py [base|small|wake]", file=sys.stderr)
        sys.exit(1)
    result = listen_and_transcribe(model_size=mode)
    print()
    print("Transcript:")
    print(result or "(empty)")
