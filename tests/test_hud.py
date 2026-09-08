"""HUD state tests."""

from __future__ import annotations

import unittest

from ui.hud import set_mode, snapshot


class TestHud(unittest.TestCase):
    def test_set_mode_and_snapshot(self) -> None:
        set_mode("speaking", heard="Hi", reply="Hello")
        state = snapshot()
        self.assertEqual(state["mode"], "speaking")
        self.assertEqual(state["heard"], "Hi")
        self.assertEqual(state["reply"], "Hello")
        set_mode("listening")
        self.assertEqual(snapshot()["mode"], "listening")
        self.assertEqual(snapshot()["heard"], "Hi")


if __name__ == "__main__":
    unittest.main()
