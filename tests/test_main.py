"""Tests for the main-loop helpers (no mic, no network)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tests import install_optional_stubs

install_optional_stubs()

from brain.llm import JarvisReply, ToolCall  # noqa: E402
from main import MAX_EXCHANGES, _run_tools, _stringify_tool_result, _think  # noqa: E402


class TestMainHelpers(unittest.TestCase):
    def test_stringify_tool_result(self) -> None:
        self.assertEqual(_stringify_tool_result("already"), "already")
        self.assertEqual(_stringify_tool_result({"k": 1}), '{"k": 1}')

    def test_run_tools_success_and_failure(self) -> None:
        reply = JarvisReply(
            text="",
            tool_calls=[
                ToolCall(id="a", name="get_current_time", input={}),
                ToolCall(id="b", name="not_a_tool", input={}),
            ],
        )
        results = _run_tools(reply)
        self.assertEqual(results[0]["tool_use_id"], "a")
        self.assertTrue(results[0]["content"])
        self.assertEqual(results[1]["tool_use_id"], "b")
        self.assertIn("failed", results[1]["content"].lower())

    def test_think_returns_final_text_without_tools(self) -> None:
        fake = JarvisReply(text="pong", tool_calls=[], assistant_message={"role": "assistant", "content": "pong"})
        with patch("main.ask_jarvis", return_value=fake) as ask:
            out = _think("ping", [])
        self.assertEqual(out, "pong")
        ask.assert_called_once()

    def test_history_window_constant(self) -> None:
        self.assertEqual(MAX_EXCHANGES, 10)


if __name__ == "__main__":
    unittest.main()
