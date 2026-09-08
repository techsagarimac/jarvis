"""Tests for SQLite fact memory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory.store import MemoryStore


class TestMemoryStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(self.tmp.name)

    def tearDown(self) -> None:
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_save_and_get_fact(self) -> None:
        self.store.save_fact("Name", "Ada")
        self.assertEqual(self.store.get_fact("name"), "Ada")
        self.assertEqual(self.store.get_fact("NAME"), "Ada")

    def test_get_missing_fact_is_none(self) -> None:
        self.assertIsNone(self.store.get_fact("nope"))

    def test_save_replaces_value(self) -> None:
        self.store.save_fact("tea", "green")
        self.store.save_fact("tea", "earl grey")
        self.assertEqual(self.store.get_fact("tea"), "earl grey")

    def test_get_all_facts(self) -> None:
        self.store.save_fact("name", "Ada")
        self.store.save_fact("city", "London")
        self.assertEqual(
            self.store.get_all_facts(),
            {"city": "London", "name": "Ada"},
        )

    def test_empty_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save_fact("  ", "x")


if __name__ == "__main__":
    unittest.main()
