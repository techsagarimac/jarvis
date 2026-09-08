"""Application configuration for Jarvis.

Reads secrets and tunables from the environment (see `.env.example`) and
exposes a single `Config` object used by audio, brain, and memory modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

_PLACEHOLDER_KEY = "sk-ant-your-key-here"


@dataclass(frozen=True)
class Config:
    """Runtime settings for the voice assistant.

    Attributes:
        anthropic_api_key: Claude API key (only required cloud credential).
        anthropic_model: Claude model identifier for chat + tool use.
        whisper_model: faster-whisper size (e.g. ``base``, ``small``).
        piper_voice: Piper voice id or path to a local ``.onnx`` voice.
        wake_word: Phrase that opens a listening window (e.g. ``"jarvis"``).
        wake_threshold: openWakeWord score (0–1) required to trigger.
        sample_rate: Microphone capture rate in Hz (Whisper expects 16 kHz).
        memory_db_path: Filesystem path to the SQLite memory store.
    """

    anthropic_api_key: str
    anthropic_model: str
    whisper_model: str
    piper_voice: str
    wake_word: str
    wake_threshold: float
    sample_rate: int
    memory_db_path: str


def load_config() -> Config:
    """Load `.env` (via python-dotenv) and return a populated ``Config``.

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is missing or still the
            placeholder from ``.env.example``.
    """
    load_dotenv()
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key or key == _PLACEHOLDER_KEY:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY in a .env file (see .env.example)."
        )
    return Config(
        anthropic_api_key=key,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        piper_voice=os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
        wake_word=os.getenv("WAKE_WORD", "jarvis"),
        wake_threshold=float(os.getenv("WAKE_THRESHOLD", "0.5")),
        sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        memory_db_path=os.getenv("MEMORY_DB_PATH", "jarvis.db"),
    )
