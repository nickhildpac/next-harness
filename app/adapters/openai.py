from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.services.tokens import TokenCounter


class OpenAIUnavailable(RuntimeError):
    pass


class OpenAIClient(LLMClient):
    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.token_counter = token_counter
        self._http = http_client
        self._client_instance: AsyncOpenAI | None = None

    def _client(self) -> AsyncOpenAI:
        if self._client_instance is None:
            if not self.settings.openai_api_key:
                raise OpenAIUnavailable("OpenAI API key is not configured")
            self._client_instance = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url or None,
            )
        return self._client_instance

    def resolve_model(self, params: GenerationParams) -> str:
        return self.settings.openai_model

    async def health(self) -> bool:
        return bool(self.settings.openai_api_key)

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        response = await self._client().chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=params.temperature,
            top_p=params.top_p,
            timeout=params.timeout_seconds,
        )
        choice = response.choices[0] if response.choices else None
        content = (choice.message.content if choice and choice.message else "") or ""
        usage = response.usage
        return LLMResult(
            content=content,
            model=response.model or self.settings.openai_model,
            input_tokens=(usage.prompt_tokens if usage else None)
            or self.token_counter.count_messages(messages),
            output_tokens=(usage.completion_tokens if usage else None)
            or self.token_counter.count(content),
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        stream = await self._client().chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=params.temperature,
            top_p=params.top_p,
            timeout=params.timeout_seconds,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
