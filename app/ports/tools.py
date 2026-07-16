"""Tool invocation port used by the agent loop."""

from __future__ import annotations

from typing import Any, Protocol

from app.ports.llm import ToolSpec
from app.tools.registry import ToolContext, ToolResult


class ToolInvoker(Protocol):
    """Anything AgentGraph can list tools from and dispatch calls through."""

    def specs(self) -> list[ToolSpec]: ...

    def names(self) -> list[str]: ...

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        call_id: str | None = None,
    ) -> ToolResult: ...
