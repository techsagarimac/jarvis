"""UI HTTP server tests (no browser)."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import unittest

from ui.server import reclaim_port, start_ui


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestUiServer(unittest.TestCase):
    def test_reclaim_port_stops_leftover_python_listener(self) -> None:
        port = _free_port()
        leftover = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import socket, time\n"
                    "s = socket.socket()\n"
                    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                    f"s.bind(('127.0.0.1', {port}))\n"
                    "s.listen(1)\n"
                    "time.sleep(30)\n"
                ),
            ]
        )
        try:
            deadline = time.time() + 2
            while time.time() < deadline:
                with socket.socket() as probe:
                    probe.settimeout(0.2)
                    try:
                        probe.connect(("127.0.0.1", port))
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                self.fail("leftover listener did not bind")
            signaled = reclaim_port(port)
            self.assertIn(leftover.pid, signaled)
            leftover.wait(timeout=2)
            with socket.socket() as probe:
                probe.settimeout(0.4)
                with self.assertRaises(OSError):
                    probe.connect(("127.0.0.1", port))
        finally:
            leftover.kill()
            leftover.wait(timeout=2)

    def test_start_ui_reclaims_busy_port(self) -> None:
        port = _free_port()
        leftover = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import socket, time\n"
                    "s = socket.socket()\n"
                    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                    f"s.bind(('127.0.0.1', {port}))\n"
                    "s.listen(1)\n"
                    "time.sleep(30)\n"
                ),
            ]
        )
        try:
            deadline = time.time() + 2
            while time.time() < deadline:
                with socket.socket() as probe:
                    probe.settimeout(0.2)
                    try:
                        probe.connect(("127.0.0.1", port))
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                self.fail("leftover listener did not bind")
            url = start_ui(open_browser=False, port=port)
            self.assertEqual(url, f"http://127.0.0.1:{port}/")
            leftover.wait(timeout=2)
        finally:
            leftover.kill()
            leftover.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
