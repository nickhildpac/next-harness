from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


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
class LLMResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


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

