"""Tiny local HTTP server for the Jarvis orb UI."""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ui.hud import snapshot

UI_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/api/state":
            body = json.dumps(snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split("?", 1)[0] in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self) -> None:
        if self.path.split("?", 1)[0] in {"/", "/index.html", ""}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        if str(args[0] if args else "").startswith("GET /api/"):
            return
        super().log_message(format, *args)


class _ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _listener_pids(port: int) -> list[int]:
    """PIDs (other than this process) listening on TCP `port`."""
    try:
        raw = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    me = os.getpid()
    pids: list[int] = []
    for token in raw.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid != me and pid not in pids:
            pids.append(pid)
    return pids


def _process_name(pid: int) -> str:
    try:
        raw = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "comm="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return raw.strip()


def reclaim_port(port: int) -> list[int]:
    """Stop leftover Python listeners on `port`. Returns PIDs that were signaled."""
    signaled: list[int] = []
    for pid in _listener_pids(port):
        name = _process_name(pid).lower()
        if "python" not in name:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
        signaled.append(pid)
    if not signaled:
        return []
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if not _listener_pids(port):
            return signaled
        time.sleep(0.05)
    for pid in list(_listener_pids(port)):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return signaled


def start_ui(open_browser: bool = True, port: int = PORT) -> str:
    """Serve the orb UI in a daemon thread. Returns the local URL."""
    try:
        httpd = _ReusableServer((HOST, port), _Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        stopped = reclaim_port(port)
        if stopped:
            print(
                f"Port {port} was in use by leftover Jarvis (pid {', '.join(map(str, stopped))}); replaced it.",
                flush=True,
            )
        httpd = _ReusableServer((HOST, port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://{HOST}:{port}/"
    print(f"Jarvis UI: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    return url
