"""
app/debug.py — Lightweight debug printer, gated by cfg.debug.enabled.

Usage anywhere in the pipeline:
    from .debug import dlog
    dlog("loader",  "Loaded filing HLSR-2024", {"section_count": 6})
    dlog("chunker", "Chunked 6 docs → 42 chunks")

Output format (only printed when debug.enabled = true in config.yml):
    [DEBUG][stage] message
             {"key": "value", ...}    ← only when data is supplied
"""
from __future__ import annotations

import json
from typing import Any


def dlog(stage: str, msg: str, data: Any = None) -> None:
    """
    Print a structured debug line to stdout.

    Gated by cfg.debug.enabled — completely silent when disabled.
    The config import is intentionally deferred (inside the function body)
    to avoid any circular-import risk at module load time.

    Args:
        stage: Short label identifying the pipeline stage (e.g. 'loader', 'risk_node').
        msg:   Human-readable message describing what just happened.
        data:  Optional dict / list / scalar with additional detail.
               Dicts and lists are JSON-serialised on one line.
    """
    from .config import cfg  # deferred — safe even when imported early
    if not cfg.debug.enabled:
        return

    prefix = f"[DEBUG][{stage}]"
    print(f"{prefix} {msg}")
    if data is not None:
        if isinstance(data, (dict, list)):
            try:
                print(f"         {json.dumps(data, default=str)}")
            except Exception:
                print(f"         {data}")
        else:
            print(f"         {data}")
