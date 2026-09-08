"""Audio-in tests: catch silent WAV/PCM mistakes without a real microphone."""

from __future__ import annotations

import os
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from tests import install_optional_stubs

install_optional_stubs()

from audio.listener import (  # noqa: E402
    SAMPLE_RATE,
    WAKE_CHUNK_SAMPLES,
    _write_wav,
    listen_and_transcribe,
    wait_for_wake_word,
)


def _read_wav(path: str) -> tuple[wave.Wave_read, bytes]:
    with wave.open(path, "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
        frames = wf.readframes(wf.getnframes())
    return params, frames


class TestWriteWav(unittest.TestCase):
    def test_writes_mono_16bit_16khz(self) -> None:
        audio = np.zeros(SAMPLE_RATE // 5, dtype=np.float32)  # 200 ms of silence
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            _write_wav(path, audio, SAMPLE_RATE)
            (channels, width, rate, nframes), raw = _read_wav(path)
            self.assertEqual(channels, 1, "Whisper needs mono; stereo headers fail silently")
            self.assertEqual(width, 2, "Must be 16-bit PCM, not float WAV")
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertEqual(nframes, len(audio))
            self.assertEqual(len(raw), len(audio) * 2)
        finally:
            os.unlink(path)

    def test_collapses_stereo_to_mono(self) -> None:
        stereo = np.column_stack(
            [
                np.ones(100, dtype=np.float32) * 0.5,
                np.ones(100, dtype=np.float32) * -0.5,
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            _write_wav(path, stereo, SAMPLE_RATE)
            (channels, _, _, nframes), _ = _read_wav(path)
            self.assertEqual(channels, 1)
            self.assertEqual(nframes, 100)
        finally:
            os.unlink(path)

    def test_clips_out_of_range_floats(self) -> None:
        audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            _write_wav(path, audio, SAMPLE_RATE)
            _, raw = _read_wav(path)
            pcm = np.frombuffer(raw, dtype=np.int16)
            self.assertEqual(int(pcm[0]), 32767)
            self.assertEqual(int(pcm[1]), -32767)
            self.assertEqual(int(pcm[2]), 0)
        finally:
            os.unlink(path)


class FakeMic:
    """Context manager that injects one float32 chunk via the InputStream callback."""

    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = chunk
        self.kwargs: dict = {}

    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return self

    def __enter__(self) -> FakeMic:
        cb = self.kwargs["callback"]
        cb(self.chunk, self.chunk.shape[0], None, None)
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class TestListenAndTranscribe(unittest.TestCase):
    def test_empty_capture_returns_empty_string(self) -> None:
        class EmptyMic:
            def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
                return self

            def __enter__(self) -> EmptyMic:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        whisper = MagicMock()
        with (
            patch("audio.listener._get_whisper", return_value=whisper),
            patch("audio.listener.sd.InputStream", EmptyMic()),
            patch("audio.listener.input", return_value=""),
        ):
            text = listen_and_transcribe(
                wait_for_start=False,
                stop_on_silence=True,
                silence_seconds=0.05,
                max_seconds=0.2,
            )
        self.assertEqual(text, "")
        whisper.transcribe.assert_not_called()

    def test_transcribe_gets_valid_wav_then_deletes_it(self) -> None:
        chunk = np.full((3200, 1), 0.2, dtype=np.float32)
        seen_path: dict[str, str] = {}

        def fake_transcribe(path: str, language: str = "en"):
            seen_path["path"] = path
            self.assertTrue(os.path.exists(path), "temp wav vanished before Whisper")
            (channels, width, rate, nframes), _ = _read_wav(path)
            self.assertEqual(channels, 1)
            self.assertEqual(width, 2)
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertGreater(nframes, 0)
            return iter([SimpleNamespace(text=" hello jarvis ")]), SimpleNamespace(
                language="en", language_probability=0.99
            )

        whisper = MagicMock()
        whisper.transcribe.side_effect = fake_transcribe
        mic = FakeMic(chunk)
        with (
            patch("audio.listener._get_whisper", return_value=whisper),
            patch("audio.listener.sd.InputStream", mic),
            patch("audio.listener.input", return_value=""),
        ):
            text = listen_and_transcribe(
                wait_for_start=False,
                sample_rate=SAMPLE_RATE,
                stop_on_silence=True,
                silence_seconds=0.05,
                max_seconds=0.3,
            )

        self.assertEqual(text, "hello jarvis")
        self.assertIn("path", seen_path)
        self.assertFalse(
            os.path.exists(seen_path["path"]),
            "temp wav should be deleted after transcribe",
        )
        self.assertEqual(mic.kwargs.get("dtype"), "float32")
        self.assertEqual(mic.kwargs.get("channels"), 1)
        self.assertEqual(mic.kwargs.get("samplerate"), SAMPLE_RATE)


class FakeWakeStream:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = list(chunks)
        self.kwargs: dict = {}

    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return self

    def __enter__(self) -> FakeWakeStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, n: int) -> tuple[np.ndarray, bool]:
        if not self.chunks:
            raise AssertionError("wait_for_wake_word kept reading after the wake score")
        return self.chunks.pop(0), False


class TestWaitForWakeWord(unittest.TestCase):
    def test_returns_when_score_crosses_threshold(self) -> None:
        silence = np.zeros((WAKE_CHUNK_SAMPLES, 1), dtype=np.int16)
        stream = FakeWakeStream([silence, silence])
        oww = MagicMock()
        oww.predict.side_effect = [
            {"hey_jarvis": 0.1},
            {"hey_jarvis": 0.86},
        ]
        with (
            patch("audio.listener._get_wake_model", return_value=oww),
            patch("audio.listener.sd.InputStream", stream),
        ):
            wait_for_wake_word(threshold=0.5)
        oww.reset.assert_called_once()
        self.assertEqual(oww.predict.call_count, 2)
        self.assertEqual(stream.kwargs.get("dtype"), "int16")
        self.assertEqual(stream.kwargs.get("blocksize"), WAKE_CHUNK_SAMPLES)
        fed = oww.predict.call_args_list[0].args[0]
        self.assertEqual(fed.dtype, np.int16)
        self.assertEqual(fed.ndim, 1)
        self.assertEqual(fed.shape[0], WAKE_CHUNK_SAMPLES)

    def test_stays_quiet_below_threshold(self) -> None:
        silence = np.zeros((WAKE_CHUNK_SAMPLES, 1), dtype=np.int16)
        stream = FakeWakeStream([silence, silence])
        oww = MagicMock()
        oww.predict.side_effect = [
            {"hey_jarvis": 0.2},
            {"hey_jarvis": 0.2},
        ]
        with (
            patch("audio.listener._get_wake_model", return_value=oww),
            patch("audio.listener.sd.InputStream", stream),
        ):
            with self.assertRaises(AssertionError):
                wait_for_wake_word(threshold=0.5)


if __name__ == "__main__":
    unittest.main()
