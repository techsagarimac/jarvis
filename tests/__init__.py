"""Test helpers: put the project root on sys.path and stub optional deps."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_OPTIONAL_MODULES = (
    "sounddevice",
    "faster_whisper",
    "openwakeword",
    "openwakeword.model",
    "openwakeword.utils",
    "piper",
    "piper.download_voices",
    "anthropic",
    "dotenv",
)


def install_optional_stubs() -> None:
    """If a heavy package is missing, register a MagicMock so imports succeed."""
    for name in _OPTIONAL_MODULES:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = MagicMock()


install_optional_stubs()
