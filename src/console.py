"""Make stdout UTF-8 on Windows consoles so Urdu letters can print."""

from __future__ import annotations

import sys


def configure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
