"""MCP client adapters for AgentGraph tool dispatch (stdio + Streamable HTTP)."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from app.core.config import Settings
from app.ports.llm import ToolSpec
from app.tools.registry import ToolResult

logger = logging.getLogger(__name__)

_IDENTITY_SCHEMA_KEYS = frozenset({"user_id", "task_id"})


def default_mcp_server_command() -> list[str]:
    return [sys.executable, "-m", "app.mcp"]


def resolve_mcp_server_command(settings: Settings) -> list[str]:
    if settings.mcp_server_command:
        return list(settings.mcp_server_command)
    return default_mcp_server_command()


def build_mcp_child_env(
    *,
    user_id: str | None = None,
    task_id: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Full parent env plus optional MCP identity overrides for the child process."""
    env = {key: value for key, value in os.environ.items() if isinstance(value, str)}
    if user_id:
        env["MCP_USER_ID"] = user_id
    if task_id:
        env["MCP_TASK_ID"] = task_id
    if extra:
        env.update(extra)
    return env


def strip_identity_from_schema(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Remove MCP identity fields so the agent manifest stays model-facing."""
    if not parameters:
        return {"type": "object", "properties": {}}
    schema = deepcopy(parameters)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key in _IDENTITY_SCHEMA_KEYS:
            properties.pop(key, None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [item for item in required if item not in _IDENTITY_SCHEMA_KEYS]
    return schema


def mcp_tool_to_spec(tool: types.Tool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        parameters=strip_identity_from_schema(
            tool.inputSchema if isinstance(tool.inputSchema, dict) else None
        ),
    )


def _text_from_content(content: list[Any] | None) -> str:
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def call_tool_result_to_tool_result(
    name: str,
    result: types.CallToolResult,
    *,
    call_id: str | None = None,
) -> ToolResult:
    """Map an MCP ``CallToolResult`` onto the agent ``ToolResult`` shape."""
    if result.isError:
        return ToolResult(
            name=name,
            call_id=call_id,
            ok=False,
            output=None,
            error=_text_from_content(result.content) or f"tool '{name}' failed",
        )

    output: Any
    if result.structuredContent is not None:
        output = result.structuredContent
    else:
        text = _text_from_content(result.content)
        if not text:
            output = None
        else:
            try:
                output = json.loads(text)
            except json.JSONDecodeError:
                output = text

    return ToolResult(name=name, call_id=call_id, ok=True, output=output)


class McpStdioSession:
    """Async context manager wrapping an MCP stdio client session."""

    def __init__(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        if not command:
            raise ValueError("MCP server command must be non-empty")
        self._command = list(command)
        self._cwd = cwd
        self._env = env
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> McpStdioSession:
        return cls(
            command=resolve_mcp_server_command(settings),
            cwd=settings.mcp_server_cwd,
            env=build_mcp_child_env(user_id=user_id, task_id=task_id),
        )

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("McpStdioSession is not started")
        return self._session

    async def __aenter__(self) -> McpStdioSession:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            params = StdioServerParameters(
                command=self._command[0],
                args=self._command[1:],
                cwd=self._cwd,
                env=self._env,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._stack = stack
            self._session = session
            logger.info(
                "mcp_stdio_session_started",
                extra={"command": self._command, "cwd": self._cwd},
            )
            return self
        except BaseException:
            await stack.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is None:
            return None
        return await stack.__aexit__(exc_type, exc, tb)

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session.list_tools()
        return list(result.tools)

    async def list_specs(self) -> list[ToolSpec]:
        return [mcp_tool_to_spec(tool) for tool in await self.list_tools()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> ToolResult:
        result = await self.session.call_tool(name, arguments or {})
        return call_tool_result_to_tool_result(name, result, call_id=call_id)


class McpStreamableHttpSession:
    """Async context manager wrapping an MCP Streamable HTTP client session."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        if not url:
            raise ValueError("MCP Streamable HTTP URL must be non-empty")
        self._url = url
        self._headers = dict(headers or {})
        self._external_client = http_client
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        auth_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> McpStreamableHttpSession:
        headers: dict[str, str] = {}
        token = auth_token or settings.mcp_http_auth_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return cls(url=settings.mcp_streamable_url, headers=headers, http_client=http_client)

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("McpStreamableHttpSession is not started")
        return self._session

    async def __aenter__(self) -> McpStreamableHttpSession:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if self._external_client is None:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(headers=self._headers, timeout=60.0)
                )
            else:
                client = self._external_client
            read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                streamable_http_client(self._url, http_client=client)
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._stack = stack
            self._session = session
            logger.info("mcp_streamable_http_session_started", extra={"url": self._url})
            return self
        except BaseException:
            await stack.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is None:
            return None
        return await stack.__aexit__(exc_type, exc, tb)

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session.list_tools()
        return list(result.tools)

    async def list_specs(self) -> list[ToolSpec]:
        return [mcp_tool_to_spec(tool) for tool in await self.list_tools()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> ToolResult:
        result = await self.session.call_tool(name, arguments or {})
        return call_tool_result_to_tool_result(name, result, call_id=call_id)
