"""Shared NDJSON framing for task run streaming."""

from __future__ import annotations

import json
from typing import Any


def ndjson_event(event: str, payload: Any) -> str:
    """One NDJSON line: ``{"event": "...", "data": ...}``."""
    return json.dumps({"event": event, "data": payload}, default=str) + "\n"
