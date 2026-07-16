"""Hybrid tool invoker: local ``finish`` + MCP stdio for everything else."""

from __future__ import annotations

from typing import Any, Protocol

from app.mcp.identity import TASK_SCOPED_TOOLS, USER_SCOPED_TOOLS
from app.ports.llm import ToolSpec
from app.tools.builtins import all_tools
from app.tools.registry import Tool, ToolContext, ToolError, ToolResult


class McpToolTransport(Protocol):
    async def list_specs(self) -> list[ToolSpec]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> ToolResult: ...


def _finish_tool() -> Tool:
    for tool in all_tools():
        if tool.name == "finish":
            return tool
    raise RuntimeError("finish tool is missing from builtins")


class HybridToolInvoker:
    """Advertise MCP tools plus local ``finish``; dispatch accordingly."""

    def __init__(
        self,
        mcp: McpToolTransport,
        mcp_specs: list[ToolSpec],
        *,
        allowed_tools: list[str] | None = None,
        finish: Tool | None = None,
    ):
        self._mcp = mcp
        self._finish = finish or _finish_tool()
        allowed = set(allowed_tools) if allowed_tools else None
        if allowed is not None:
            allowed.add("finish")
        self._allowed = allowed

        filtered = [
            spec
            for spec in mcp_specs
            if spec.name != "finish" and (allowed is None or spec.name in allowed)
        ]
        self._mcp_specs = {spec.name: spec for spec in filtered}
        self._specs = [*self._mcp_specs.values(), self._finish.spec()]

    @classmethod
    async def create(
        cls,
        mcp: McpToolTransport,
        *,
        allowed_tools: list[str] | None = None,
        finish: Tool | None = None,
    ) -> HybridToolInvoker:
        return cls(mcp, await mcp.list_specs(), allowed_tools=allowed_tools, finish=finish)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def names(self) -> list[str]:
        return [spec.name for spec in self._specs]

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        call_id: str | None = None,
    ) -> ToolResult:
        if self._allowed is not None and name not in self._allowed:
            return ToolResult(
                name=name,
                call_id=call_id,
                ok=False,
                output=None,
                error=f"tool '{name}' is not allowed for this task",
            )

        if name == "finish":
            return await self._invoke_local(self._finish, arguments, context, call_id=call_id)

        if name not in self._mcp_specs:
            return ToolResult(
                name=name,
                call_id=call_id,
                ok=False,
                output=None,
                error=f"unknown tool '{name}'",
            )

        payload = dict(arguments or {})
        if name in USER_SCOPED_TOOLS and context.user_id and "user_id" not in payload:
            payload["user_id"] = context.user_id
        if name in TASK_SCOPED_TOOLS and context.task_id and "task_id" not in payload:
            payload["task_id"] = context.task_id

        return await self._mcp.call_tool(name, payload, call_id=call_id)

    async def _invoke_local(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        call_id: str | None,
    ) -> ToolResult:
        try:
            output = await tool.handler(arguments or {}, context)
        except ToolError as exc:
            return ToolResult(
                name=tool.name, call_id=call_id, ok=False, output=None, error=str(exc)
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the model as an error
            return ToolResult(
                name=tool.name,
                call_id=call_id,
                ok=False,
                output=None,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        return ToolResult(name=tool.name, call_id=call_id, ok=True, output=output)
