"""Tests for MCP stdio client mapping and HybridToolInvoker."""

from __future__ import annotations

from typing import Any

from mcp import types

from app.adapters.mcp_client import (
    call_tool_result_to_tool_result,
    default_mcp_server_command,
    mcp_tool_to_spec,
    resolve_mcp_server_command,
    strip_identity_from_schema,
)
from app.core.config import Settings
from app.ports.llm import ToolSpec
from app.tools.mcp_invoker import HybridToolInvoker
from app.tools.registry import ToolContext, ToolResult


class FakeMcpSession:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs = specs or [
            ToolSpec(
                name="now",
                description="current time",
                parameters={"type": "object", "properties": {}},
            ),
            ToolSpec(
                name="list_notes",
                description="list notes",
                parameters={
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                },
            ),
            ToolSpec(
                name="search_task_documents",
                description="search docs",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ]
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.responses: dict[str, ToolResult] = {}

    async def list_specs(self) -> list[ToolSpec]:
        return list(self._specs)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> ToolResult:
        self.calls.append((name, arguments))
        if name in self.responses:
            result = self.responses[name]
            return ToolResult(
                name=result.name,
                call_id=call_id,
                ok=result.ok,
                output=result.output,
                error=result.error,
            )
        return ToolResult(name=name, call_id=call_id, ok=True, output={"ok": True})


def test_strip_identity_from_schema_removes_user_and_task():
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer"},
            "user_id": {"type": "string"},
            "task_id": {"type": "string"},
        },
        "required": ["limit", "user_id"],
    }
    cleaned = strip_identity_from_schema(schema)
    assert "user_id" not in cleaned["properties"]
    assert "task_id" not in cleaned["properties"]
    assert cleaned["required"] == ["limit"]
    assert "user_id" in schema["properties"]  # original untouched


def test_mcp_tool_to_spec_strips_identity_fields():
    tool = types.Tool(
        name="list_notes",
        description="list",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "user_id": {"type": "string"},
            },
        },
    )
    spec = mcp_tool_to_spec(tool)
    assert spec.name == "list_notes"
    assert "user_id" not in spec.parameters["properties"]
    assert "limit" in spec.parameters["properties"]


def test_call_tool_result_maps_error():
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="boom")],
        isError=True,
    )
    mapped = call_tool_result_to_tool_result("now", result, call_id="c1")
    assert mapped.ok is False
    assert mapped.error == "boom"
    assert mapped.call_id == "c1"


def test_call_tool_result_prefers_structured_content():
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text='{"ignored": true}')],
        structuredContent={"utc": "2026-01-01T00:00:00Z"},
        isError=False,
    )
    mapped = call_tool_result_to_tool_result("now", result)
    assert mapped.ok is True
    assert mapped.output == {"utc": "2026-01-01T00:00:00Z"}


def test_call_tool_result_parses_json_text_when_no_structured():
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text='{"seen": 1}')],
        isError=False,
    )
    mapped = call_tool_result_to_tool_result("echo", result)
    assert mapped.ok is True
    assert mapped.output == {"seen": 1}


def test_resolve_mcp_server_command_defaults_and_override():
    assert resolve_mcp_server_command(Settings()) == default_mcp_server_command()
    settings = Settings(mcp_server_command=["/bin/custom", "-m", "app.mcp"])
    assert resolve_mcp_server_command(settings) == ["/bin/custom", "-m", "app.mcp"]


async def test_hybrid_invoker_keeps_finish_local_and_routes_mcp():
    mcp = FakeMcpSession()
    mcp.responses["now"] = ToolResult(name="now", call_id=None, ok=True, output={"utc": "x"})
    invoker = HybridToolInvoker(mcp, await mcp.list_specs())

    names = set(invoker.names())
    assert "finish" in names
    assert "now" in names
    assert "list_notes" in names

    finish = await invoker.invoke("finish", {"summary": "done"}, ToolContext(), call_id="f1")
    assert finish.ok is True
    assert finish.output == {"summary": "done"}
    assert mcp.calls == []

    now = await invoker.invoke("now", {}, ToolContext(), call_id="n1")
    assert now.ok is True
    assert now.output == {"utc": "x"}
    assert mcp.calls == [("now", {})]


async def test_hybrid_invoker_injects_identity_from_context():
    mcp = FakeMcpSession()
    invoker = HybridToolInvoker(mcp, await mcp.list_specs())
    context = ToolContext(user_id="alice", task_id="task-1")

    await invoker.invoke("list_notes", {"limit": 5}, context)
    await invoker.invoke("search_task_documents", {"query": "fruit"}, context)

    assert mcp.calls[0] == ("list_notes", {"limit": 5, "user_id": "alice"})
    assert mcp.calls[1] == (
        "search_task_documents",
        {"query": "fruit", "task_id": "task-1"},
    )


async def test_hybrid_invoker_honors_allowed_tools():
    mcp = FakeMcpSession()
    invoker = HybridToolInvoker(
        mcp,
        await mcp.list_specs(),
        allowed_tools=["now"],
    )
    assert set(invoker.names()) == {"now", "finish"}
    denied = await invoker.invoke("list_notes", {}, ToolContext())
    assert denied.ok is False
    assert "not allowed" in (denied.error or "")
