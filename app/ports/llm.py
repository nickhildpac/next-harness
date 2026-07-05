from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationParams:
    model: str
    temperature: float
    top_p: float
    timeout_seconds: float


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool advertisement handed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A structured request from the model to invoke a tool."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    def resolve_model(self, params: GenerationParams) -> str:
        """The model this client will actually use for the given params."""
        ...

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        ...

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        ...

    async def health(self) -> bool:
        ...
