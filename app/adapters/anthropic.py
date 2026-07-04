from collections.abc import AsyncIterator

import httpx
from anthropic import AsyncAnthropic
from anthropic import NOT_GIVEN

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.services.tokens import TokenCounter


class AnthropicUnavailable(RuntimeError):
    pass


def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    turns: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            turns.append({"role": m.role, "content": m.content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns


class AnthropicClient(LLMClient):
    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.token_counter = token_counter
        self._http = http_client
        self._client_instance: AsyncAnthropic | None = None

    def _client(self) -> AsyncAnthropic:
        if self._client_instance is None:
            if not self.settings.anthropic_api_key:
                raise AnthropicUnavailable("Anthropic API key is not configured")
            self._client_instance = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._client_instance

    def resolve_model(self, params: GenerationParams) -> str:
        return self.settings.anthropic_model

    async def health(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        system, turns = _split_system(messages)
        response = await self._client().messages.create(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.anthropic_max_tokens,
            system=system if system is not None else NOT_GIVEN,
            messages=turns,
            temperature=params.temperature,
            top_p=params.top_p,
            timeout=params.timeout_seconds,
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return LLMResult(
            content=content,
            model=response.model or self.settings.anthropic_model,
            input_tokens=(usage.input_tokens if usage else None)
            or self.token_counter.count_messages(messages),
            output_tokens=(usage.output_tokens if usage else None)
            or self.token_counter.count(content),
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        system, turns = _split_system(messages)
        async with self._client().messages.stream(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.anthropic_max_tokens,
            system=system if system is not None else NOT_GIVEN,
            messages=turns,
            temperature=params.temperature,
            top_p=params.top_p,
            timeout=params.timeout_seconds,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
