from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import types

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.services.tokens import TokenCounter


class GeminiUnavailable(RuntimeError):
    pass


def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
            continue
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, contents


class GeminiClient(LLMClient):
    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.token_counter = token_counter
        self._http = http_client
        self._client_instance: genai.Client | None = None

    def _client(self) -> genai.Client:
        if self._client_instance is None:
            if not self.settings.gemini_api_key:
                raise GeminiUnavailable("Gemini API key is not configured")
            self._client_instance = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client_instance

    def _config(self, system: str | None, params: GenerationParams) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=params.temperature,
            top_p=params.top_p,
        )

    def resolve_model(self, params: GenerationParams) -> str:
        return self.settings.gemini_model

    async def health(self) -> bool:
        return bool(self.settings.gemini_api_key)

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        system, contents = _split_system(messages)
        response = await self._client().aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=contents,
            config=self._config(system, params),
        )
        content = response.text or ""
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        return LLMResult(
            content=content,
            model=getattr(response, "model_version", None) or self.settings.gemini_model,
            input_tokens=input_tokens or self.token_counter.count_messages(messages),
            output_tokens=output_tokens or self.token_counter.count(content),
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        system, contents = _split_system(messages)
        stream = await self._client().aio.models.generate_content_stream(
            model=self.settings.gemini_model,
            contents=contents,
            config=self._config(system, params),
        )
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
