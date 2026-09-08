"""Tests for tool dispatch (no live LLM)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import memory.store as store_mod
from brain.tools import (
    TOOL_FUNCS,
    TOOL_SCHEMAS,
    dispatch,
    recall_facts,
    remember_fact,
)


class TestTools(unittest.TestCase):
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

    def test_schemas_match_funcs(self) -> None:
        names = {s["name"] for s in TOOL_SCHEMAS}
        self.assertEqual(names, set(TOOL_FUNCS))

    def test_dispatch_time(self) -> None:
        text = dispatch("get_current_time", {})
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 8)


    def test_remember_and_recall_roundtrip(self) -> None:
        msg = remember_fact("drink", "earl grey")
        self.assertIn("drink", msg)
        self.assertEqual(recall_facts("drink"), {"drink": "earl grey"})
        self.assertEqual(recall_facts()["drink"], "earl grey")

    def test_dispatch_unknown_tool(self) -> None:
        with self.assertRaises(KeyError):
            dispatch("not_a_tool", {})


if __name__ == "__main__":
    unittest.main()
