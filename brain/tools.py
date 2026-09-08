"""Functions Claude is allowed to call, plus their JSON tool schemas.

``TOOL_SCHEMAS`` is passed to the Anthropic Messages API. ``dispatch``
runs the matching Python function and returns a JSON-serializable value.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from memory.store import get_all_facts, get_fact, save_fact


def get_current_time() -> str:
    """Return the current local date and time as a readable string."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def remember_fact(key: str, value: str) -> str:
    """Persist a fact in SQLite via ``memory.store.save_fact``."""
    save_fact(key, value)
    return f"Remembered {key!r}."


def recall_facts(query: str | None = None) -> dict[str, str]:
    """Look up facts from SQLite, optionally filtered by a substring."""
    if not query:
        return get_all_facts()
    needle = query.lower()
    exact = get_fact(query)
    if exact is not None:
        return {query.strip().lower(): exact}
    return {
        k: v
        for k, v in get_all_facts().items()
        if needle in k.lower() or needle in v.lower()
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "remember_fact",
        "description": "Remember a fact about the user or their world for later.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short label, e.g. 'preferred_name'.",
                },
                "value": {
                    "type": "string",
                    "description": "What to remember.",
                },
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall_facts",
        "description": "Look up facts previously stored with remember_fact.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional substring to filter keys or values.",
                },
            },
            "additionalProperties": False,
        },
    },
]

TOOL_FUNCS: dict[str, Callable[..., Any]] = {
    "get_current_time": get_current_time,
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
}


def dispatch(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Run the tool named ``name`` with JSON ``arguments`` from Claude.

    Raises:
        KeyError: If ``name`` is not a registered tool.
    """
    func = TOOL_FUNCS[name]
    kwargs = dict(arguments or {})
    return func(**kwargs)
