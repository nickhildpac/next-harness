from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


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
    strict: bool = True


@dataclass(frozen=True)
class ToolCall:
    """A structured request from the model to invoke a tool."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ToolChoice:
    """How the model should be constrained to use tools on a given turn."""

    mode: str  # "auto" | "none" | "required" | "force"
    tool_name: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    # Present on assistant messages that made native tool calls.
    tool_calls: tuple[ToolCall, ...] | None = None
    # Present on role="tool" messages, linking the result back to its call.
    tool_call_id: str | None = None


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

    def supports_native_tools(self) -> bool:
        """Whether this client can take ``tools``/``tool_choice`` in ``chat`` and return
        structured ``LLMResult.tool_calls`` instead of relying on the fenced-JSON text
        protocol. Adapters that don't override this default to False.
        """
        return False

    async def chat(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResult:
        ...

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        ...

    async def health(self) -> bool:
        ...
