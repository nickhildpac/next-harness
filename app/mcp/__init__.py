"""Stdio MCP surface over the same ToolRegistry used by agent tasks."""

from __future__ import annotations

from typing import Any

__all__ = ["create_server", "run_stdio"]


def __getattr__(name: str) -> Any:
    if name in {"create_server", "run_stdio"}:
        from app.mcp.server import create_server, run_stdio

        return {"create_server": create_server, "run_stdio": run_stdio}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
