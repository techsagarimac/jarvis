"""Audio-out tests: empty text, PCM dtype, and play() actually being called."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from tests import install_optional_stubs

install_optional_stubs()

from audio.speaker import _resolve_onnx_path, speak  # noqa: E402


class TestResolveOnnxPath(unittest.TestCase):
    def test_missing_onnx_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _resolve_onnx_path("/no/such/voice.onnx")

    def test_existing_onnx_path_is_returned(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            path = tmp.name
        try:
            resolved = _resolve_onnx_path(path)
            self.assertEqual(resolved, Path(path))
        finally:
            os.unlink(path)


class TestSpeak(unittest.TestCase):
    def test_blank_text_does_not_touch_audio(self) -> None:
        with (
            patch("audio.speaker._load_voice") as load,
            patch("audio.speaker.sd.play") as play,
        ):
            speak("   ")
            speak("")
            load.assert_not_called()
            play.assert_not_called()

    def test_plays_int16_at_voice_sample_rate(self) -> None:
        chunk_a = SimpleNamespace(
            sample_rate=22050,
            audio_int16_bytes=np.array([100, 200], dtype=np.int16).tobytes(),
        )
        chunk_b = SimpleNamespace(
            sample_rate=22050,
            audio_int16_bytes=np.array([300, 400], dtype=np.int16).tobytes(),
        )
        piper = MagicMock()
        piper.synthesize.return_value = [chunk_a, chunk_b]
        with (
            patch("audio.speaker._load_voice", return_value=piper) as load,
            patch("audio.speaker.sd.play") as play,
            patch("audio.speaker.sd.wait") as wait,
        ):
            speak("hello")
        load.assert_called_once()
        piper.synthesize.assert_called_once_with("hello")
        play.assert_called_once()
        audio = play.call_args.args[0]
        kwargs = play.call_args.kwargs
        rate = kwargs.get("samplerate")
        if rate is None and len(play.call_args.args) > 1:
            rate = play.call_args.args[1]
        self.assertEqual(audio.dtype, np.int16, "playback dtype mismatch is a silent no-sound bug")
        np.testing.assert_array_equal(audio, np.array([100, 200, 300, 400], dtype=np.int16))
        self.assertEqual(rate, 22050)
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
