"""Minimal dependency-free progress bar for long data preparation loops."""

from __future__ import annotations

import sys


def show_progress(label: str, done: int, total: int, *, width: int = 40) -> None:
    """Print an in-place bar at integer-percent changes, with a final newline."""
    if total <= 0:
        return
    pct = min(100, int(100 * done / total))
    prev = min(100, int(100 * (done - 1) / total)) if done > 0 else -1
    if done not in (0, total) and pct == prev:
        return
    filled = int(width * pct / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[setup] {label}: [{bar}] {pct:3d}% ({done:,}/{total:,})",
          end="\n" if done >= total else "", file=sys.stderr, flush=True)
