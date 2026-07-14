"""Low-level MCP Server wrapping ``build_default_registry()``."""

from __future__ import annotations

import json
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from app.core.config import Settings, get_settings
from app.mcp.context import McpRuntime
from app.mcp.identity import (
    EXCLUDED_TOOLS,
    IdentityError,
    augment_schema,
    resolve_identity,
)
from app.tools.registry import ToolRegistry, build_default_registry


def mcp_tools(registry: ToolRegistry) -> list[types.Tool]:
    """Advertise registry tools (minus finish) using each Tool.spec()."""
    advertised: list[types.Tool] = []
    for tool in registry.tools():
        if tool.name in EXCLUDED_TOOLS:
            continue
        spec = tool.spec()
        advertised.append(
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=augment_schema(spec.name, spec.parameters),
            )
        )
    return advertised


async def handle_call_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    registry: ToolRegistry,
    settings: Settings,
    runtime: McpRuntime,
) -> types.CallToolResult:
    """Dispatch a single MCP tool call through the app ToolRegistry."""
    if name in EXCLUDED_TOOLS:
        return _error_result(f"tool '{name}' is not available via MCP")
    if registry.get(name) is None:
        return _error_result(f"unknown tool '{name}'")

    try:
        remaining, user_id, task_id = resolve_identity(name, arguments, settings)
    except IdentityError as exc:
        return _error_result(str(exc))

    result = await runtime.invoke(
        registry,
        name,
        remaining,
        user_id=user_id,
        task_id=task_id,
    )
    if not result.ok:
        return _error_result(result.error or f"tool '{name}' failed")

    structured: dict[str, Any]
    if isinstance(result.output, dict):
        structured = result.output
    else:
        structured = {"output": result.output}

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(result.output, default=str),
            )
        ],
        structuredContent=structured,
        isError=False,
    )


def create_server(
    *,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
    runtime: McpRuntime | None = None,
) -> Server:
    """Build an MCP Server whose tools mirror the app ToolRegistry."""
    registry = registry or build_default_registry()
    settings = settings or get_settings()
    runtime = runtime or McpRuntime(settings)
    server = Server("next-harness")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return mcp_tools(registry)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await handle_call_tool(
            name,
            arguments,
            registry=registry,
            settings=settings,
            runtime=runtime,
        )

    return server


def _error_result(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        isError=True,
    )


async def run_stdio(
    *,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
) -> None:
    """Serve tools over stdio until the client disconnects."""
    settings = settings or get_settings()
    async with McpRuntime(settings) as runtime:
        server = create_server(registry=registry, settings=settings, runtime=runtime)
        init_options = server.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
