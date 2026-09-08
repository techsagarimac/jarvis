"""Fail the suite if a real Anthropic key is hardcoded anywhere in the repo."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "models", "__pycache__", "agent-tools"}
# Placeholder in .env.example / config.py, and keys used only in tests.
ALLOWED = re.compile(r"sk-ant-(your-key-here|test[-_])")
KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_-]+")


class TestNoHardcodedSecrets(unittest.TestCase):
    def test_gitignore_excludes_dotenv(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore.splitlines())

    def test_no_real_api_keys_in_source(self) -> None:
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".md", ".txt", ".example", ".yml", ".yaml", ".toml", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for match in KEY_RE.finditer(text):
                token = match.group(0)
                if ALLOWED.match(token):
                    continue
                rel = path.relative_to(ROOT)
                offenders.append(f"{rel}: {token}")
        self.assertEqual(
            offenders,
            [],
            "Hardcoded API keys found (keep them in .env only):\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
