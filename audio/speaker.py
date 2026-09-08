"""Local text-to-speech output via Piper.

``speak`` synthesizes text with piper-tts and plays it on the default
output device. The first call downloads a default English voice into
``models/`` if it is not already present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from piper.download_voices import download_voice

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

DEFAULT_VOICE = "en_US-lessac-medium"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SAMPLE_SENTENCE = "Hello. I am Jarvis, your local voice assistant."

_voice_cache: dict[str, PiperVoice] = {}


class Speaker:
    """Convert assistant replies into audible speech.

    Args:
        voice: Piper voice id (e.g. ``en_US-lessac-medium``) or path to
            a ``.onnx`` file plus its ``.onnx.json`` config.
        sample_rate: Playback rate; must match the voice's native rate
            unless resampling is added later.
    """

    def __init__(self, voice: str, sample_rate: int = 22_050) -> None:
        raise NotImplementedError

    def speak(self, text: str) -> None:
        """Synthesize ``text`` with Piper and play it to completion."""
        raise NotImplementedError

    def synthesize(self, text: str) -> bytes:
        """Return raw PCM (or WAV) bytes for ``text`` without playing them."""
        raise NotImplementedError


def _resolve_onnx_path(voice: str) -> Path:
    """Return the ``.onnx`` path for ``voice``, downloading if needed."""
    given = Path(voice)
    if given.suffix == ".onnx":
        if not given.is_file():
            raise FileNotFoundError(f"Piper voice not found: {given}")
        return given

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS_DIR / f"{voice}.onnx"
    if not onnx_path.is_file():
        print(f"Downloading Piper voice '{voice}' into {MODELS_DIR}…", flush=True)
        download_voice(voice, MODELS_DIR)
    return onnx_path


def _load_voice(voice: str = DEFAULT_VOICE) -> PiperVoice:
    """Load and cache a ``PiperVoice`` from a voice id or ``.onnx`` path."""
    if voice not in _voice_cache:
        onnx_path = _resolve_onnx_path(voice)
        print(f"Loading Piper voice {onnx_path.name}…", flush=True)
        _voice_cache[voice] = PiperVoice.load(str(onnx_path))
    return _voice_cache[voice]


def preload_voice(voice: str = DEFAULT_VOICE) -> None:
    """Load (and download if needed) the Piper voice before the first reply."""
    _load_voice(voice)


def speak(text: str, voice: str = DEFAULT_VOICE) -> None:
    """Convert ``text`` to speech with piper-tts and play it.

    Uses the default output device via ``sounddevice``. Empty or
    whitespace-only strings are ignored.

    Args:
        text: Utterance to speak.
        voice: Piper voice id (``language-name-quality``) or a path to
            an existing ``.onnx`` file. Defaults to
            ``en_US-lessac-medium``.
    """
    if not text or not text.strip():
        return

    piper = _load_voice(voice)
    pieces: list[np.ndarray] = []
    sample_rate: int | None = None

    for chunk in piper.synthesize(text.strip()):
        sample_rate = chunk.sample_rate
        pieces.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))

    if not pieces or sample_rate is None:
        return

    audio = np.concatenate(pieces)
    try:
        from ui.hud import set_mode

        set_mode("speaking")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    finally:
        try:
            from ui.hud import set_mode as _reset

            _reset("listening")
        except Exception:
            pass


if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else SAMPLE_SENTENCE
    print(f"Speaking: {sentence}", flush=True)
    speak(sentence)
    print("Done.", flush=True)
