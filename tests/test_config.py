"""Tests for config.load_config — env only, never a baked-in key."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tests import install_optional_stubs

install_optional_stubs()

from config import load_config  # noqa: E402


class TestConfig(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_missing_key_raises(self, _dotenv: object) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                load_config()
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    @patch("config.load_dotenv")
    def test_placeholder_key_raises(self, _dotenv: object) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-your-key-here"}):
            with self.assertRaises(RuntimeError):
                load_config()

    @patch("config.load_dotenv")
    def test_reads_optional_overrides(self, _dotenv: object) -> None:
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "sk-ant-test-unit-not-a-real-key",
                "ANTHROPIC_MODEL": "claude-test",
                "WHISPER_MODEL": "small",
                "PIPER_VOICE": "en_US-lessac-medium",
                "WAKE_THRESHOLD": "0.7",
                "SAMPLE_RATE": "16000",
                "MEMORY_DB_PATH": "unit.db",
            },
        ):
            cfg = load_config()
        self.assertEqual(cfg.anthropic_api_key, "sk-ant-test-unit-not-a-real-key")
        self.assertEqual(cfg.anthropic_model, "claude-test")
        self.assertEqual(cfg.whisper_model, "small")
        self.assertEqual(cfg.wake_threshold, 0.7)
        self.assertEqual(cfg.memory_db_path, "unit.db")


if __name__ == "__main__":
    unittest.main()
