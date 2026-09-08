"""Tests for Claude wiring: system-prompt facts and one mocked API round-trip."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests import install_optional_stubs

install_optional_stubs()

import memory.store as store_mod  # noqa: E402
from brain.llm import (  # noqa: E402
    SYSTEM_PROMPT,
    _blocks_to_payload,
    _system_prompt,
    ask_jarvis,
)


class TestLlm(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self._db_env = patch.dict(os.environ, {"MEMORY_DB_PATH": self.tmp.name})
        self._db_env.start()
        store_mod._store = None

    def tearDown(self) -> None:
        store_mod._store = None
        self._db_env.stop()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_system_prompt_includes_facts(self) -> None:
        store_mod.save_fact("name", "Ada")
        prompt = _system_prompt()
        self.assertTrue(prompt.startswith(SYSTEM_PROMPT))
        self.assertIn("name: Ada", prompt)

    def test_system_prompt_without_facts_is_base(self) -> None:
        self.assertEqual(_system_prompt(), SYSTEM_PROMPT)

    def test_blocks_to_payload(self) -> None:
        blocks = [
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="t1", name="get_current_time", input={}),
        ]
        self.assertEqual(
            _blocks_to_payload(blocks),
            [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "get_current_time",
                    "input": {},
                },
            ],
        )

    def test_ask_jarvis_returns_text_from_mocked_api(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=" Hello from Jarvis ")]
        )
        with patch("brain.llm._get_client", return_value=(client, "claude-test")):
            reply = ask_jarvis("hi", history=[{"role": "user", "content": "prior"}])
        self.assertEqual(reply.text, "Hello from Jarvis")
        self.assertEqual(reply.tool_calls, [])
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-test")
        self.assertIn("system", kwargs)
        self.assertEqual(kwargs["messages"][-1], {"role": "user", "content": "hi"})
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("ANTHROPIC", str(kwargs).upper())

    def test_ask_jarvis_surfaces_tool_calls(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="remember_fact",
                    input={"key": "name", "value": "Ada"},
                )
            ]
        )
        with patch("brain.llm._get_client", return_value=(client, "claude-test")):
            reply = ask_jarvis("remember my name")
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].name, "remember_fact")
        self.assertEqual(reply.tool_calls[0].id, "toolu_1")


if __name__ == "__main__":
    unittest.main()
