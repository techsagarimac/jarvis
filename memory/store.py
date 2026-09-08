"""Simple SQLite-backed long-term memory.

Facts are stored as key/value rows (name, preferences, and similar).
``save_fact``, ``get_fact``, and ``get_all_facts`` are the public API;
``MemoryStore`` is a thin class wrapper around the same database.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_lock = threading.Lock()
_store: MemoryStore | None = None


def default_db_path() -> str:
    """Return ``MEMORY_DB_PATH`` from the environment, or ``jarvis.db``."""
    return os.getenv("MEMORY_DB_PATH", "jarvis.db")


def _norm_key(key: str) -> str:
    key = (key or "").strip().lower()
    if not key:
        raise ValueError("fact key must be a non-empty string")
    return key


@contextmanager
def _connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or default_db_path()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


class MemoryStore:
    """Open (or create) the SQLite file and read/write facts.

    Args:
        db_path: Filesystem path, e.g. ``jarvis.db`` in the project root.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or default_db_path()

    def save_fact(self, key: str, value: str) -> None:
        """Insert or replace a fact about the user."""
        key = _norm_key(key)
        value = (value or "").strip()
        with _lock, _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO facts (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )

    def get_fact(self, key: str) -> str | None:
        """Return the value for ``key``, or ``None`` if it is unknown."""
        key = _norm_key(key)
        with _lock, _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def get_all_facts(self) -> dict[str, str]:
        """Return every stored fact as ``{key: value}``."""
        with _lock, _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts ORDER BY key"
            ).fetchall()
        return {str(k): str(v) for k, v in rows}

    def remember(self, key: str, value: str) -> None:
        """Insert or update a fact the assistant should recall later."""
        self.save_fact(key, value)

    def recall(self, query: str | None = None) -> list[dict[str, Any]]:
        """Return matching facts, or all facts if ``query`` is omitted."""
        facts = self.get_all_facts()
        if query:
            needle = query.lower()
            facts = {
                k: v
                for k, v in facts.items()
                if needle in k or needle in v.lower()
            }
        return [{"key": k, "value": v} for k, v in facts.items()]

    def log_turn(self, role: str, content: str) -> None:
        """Append one conversation turn (``user`` / ``assistant``) to history."""
        raise NotImplementedError

    def recent_turns(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the last ``limit`` turns, oldest first, for LLM context."""
        raise NotImplementedError

    def close(self) -> None:
        """No-op: connections are opened per call and closed immediately."""
        return None


def _default_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def save_fact(key: str, value: str) -> None:
    """Store a fact about the user (name, preferences, etc.)."""
    _default_store().save_fact(key, value)


def get_fact(key: str) -> str | None:
    """Retrieve one fact by key, or ``None`` if it has not been saved."""
    return _default_store().get_fact(key)


def get_all_facts() -> dict[str, str]:
    """Return every stored fact as a dict."""
    return _default_store().get_all_facts()
