from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.ports.embeddings import EmbeddingsClient
from app.ports.llm import ToolSpec
from app.ports.llm import LLMClient
from app.ports.vectorstore import VectorStore


class ToolError(RuntimeError):
    """Raised by tool handlers to surface a structured, model-visible error."""


@dataclass
class ToolContext:
    """Scoped services made available to tool handlers by the agent runtime."""

    session: AsyncSession | None = None
    http_client: httpx.AsyncClient | None = None
    user_id: str | None = None
    task_id: str | None = None
    settings: Settings | None = None
    llm: LLMClient | None = None
    embeddings: EmbeddingsClient | None = None
    vectorstore: VectorStore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


@dataclass(frozen=True)
class ToolResult:
    name: str
    call_id: str | None
    ok: bool
    output: Any
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "call_id": self.call_id,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    async def invoke(
        self, name: str, arguments: dict[str, Any], context: ToolContext, *, call_id: str | None = None
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                name=name, call_id=call_id, ok=False, output=None, error=f"unknown tool '{name}'"
            )
        try:
            output = await tool.handler(arguments or {}, context)
        except ToolError as exc:
            return ToolResult(name=name, call_id=call_id, ok=False, output=None, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — surfaced to the model as an error
            return ToolResult(
                name=name,
                call_id=call_id,
                ok=False,
                output=None,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        return ToolResult(name=name, call_id=call_id, ok=True, output=output)


def build_default_registry() -> ToolRegistry:
    """Register the built-in tools available to every agent run."""
    from app.tools import builtins

    return ToolRegistry(builtins.all_tools())
