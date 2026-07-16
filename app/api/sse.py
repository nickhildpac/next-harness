"""Shared Server-Sent Events helpers."""

from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
