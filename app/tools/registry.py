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


ToolExecutor = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]

_VALID_JSON_SCHEMA_TYPES = {
    "object",
    "array",
    "string",
    "number",
    "integer",
    "boolean",
    "null",
}


def validate_input_schema(schema: dict[str, Any], *, strict: bool = False) -> None:
    """Reject malformed tool input schemas early, at registration time.

    Only checks the shape conventions this codebase relies on (object schema
    with a properties map and a required list drawn from those properties),
    not full JSON Schema compliance. When ``strict`` is set, also enforces the
    stricter shape OpenAI's structured-output mode requires: every declared
    property must be listed as required (use a nullable type for genuinely
    optional fields) and ``additionalProperties`` must be ``False``.
    """
    if not isinstance(schema, dict):
        raise ValueError("input_schema must be a dict")
    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in _VALID_JSON_SCHEMA_TYPES:
        raise ValueError(f"input_schema has unknown type '{schema_type}'")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise ValueError("input_schema.properties must be a dict")
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_name, str):
                raise ValueError("input_schema.properties keys must be strings")
            if not isinstance(prop_schema, dict):
                raise ValueError(f"input_schema.properties.{prop_name} must be a dict")
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise ValueError("input_schema.required must be a list of strings")
        known = set((properties or {}).keys())
        unknown = [r for r in required if r not in known]
        if unknown:
            raise ValueError(f"input_schema.required references unknown properties: {unknown}")
    if strict:
        if schema.get("additionalProperties") is not False:
            raise ValueError("strict input_schema must set additionalProperties: False")
        known = set((properties or {}).keys())
        missing = sorted(known - set(required or []))
        if missing:
            raise ValueError(f"strict input_schema must require every property: {missing}")


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    executor: ToolExecutor
    consequential: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        validate_input_schema(self.input_schema, strict=self.strict)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.input_schema,
            strict=self.strict,
        )


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

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        call_id: str | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                name=name, call_id=call_id, ok=False, output=None, error=f"unknown tool '{name}'"
            )
        try:
            output = await tool.executor(arguments or {}, context)
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

    return ToolRegistry(builtins.REGISTRY)
