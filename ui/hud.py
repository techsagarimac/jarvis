"""Shared HUD state for the Jarvis orb UI."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_state: dict[str, Any] = {
    "mode": "idle",
    "heard": "",
    "reply": "",
}


def set_mode(
    mode: str,
    *,
    heard: str | None = None,
    reply: str | None = None,
) -> None:
    """Update the orb mode: idle, listening, recording, thinking, speaking."""
    with _lock:
        _state["mode"] = mode
        if heard is not None:
            _state["heard"] = heard
        if reply is not None:
            _state["reply"] = reply


def snapshot() -> dict[str, Any]:
    """Return a copy of the current HUD state."""
    with _lock:
        return dict(_state)
