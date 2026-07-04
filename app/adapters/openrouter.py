import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.services.tokens import TokenCounter


class OpenRouterUnavailable(RuntimeError):
    pass


class OpenRouterClient(LLMClient):
    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.token_counter = token_counter
        self.base_url = str(settings.openrouter_base_url).rstrip("/")
        self._http = http_client

    def resolve_model(self, params: GenerationParams) -> str:
        return self.settings.openrouter_model

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http is not None:
            yield self._http
        else:
            async with httpx.AsyncClient() as client:
                yield client

    async def health(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OpenRouterUnavailable)),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        response = await self._post_chat(messages, params, stream=False)
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        return LLMResult(
            content=content,
            model=data.get("model", self._model()),
            input_tokens=usage.get("prompt_tokens", self.token_counter.count_messages(messages)),
            output_tokens=usage.get("completion_tokens", self.token_counter.count(content)),
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, params, stream=True)
        async with self._client() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=params.timeout_seconds,
            ) as response:
                if response.status_code >= 500:
                    raise OpenRouterUnavailable("OpenRouter returned an unavailable status")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

    async def _post_chat(
        self, messages: list[ChatMessage], params: GenerationParams, *, stream: bool
    ) -> httpx.Response:
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, params, stream=stream),
                timeout=params.timeout_seconds,
            )
        if response.status_code >= 500:
            raise OpenRouterUnavailable("OpenRouter returned an unavailable status")
        response.raise_for_status()
        return response

    def _payload(
        self, messages: list[ChatMessage], params: GenerationParams, *, stream: bool
    ) -> dict:
        return {
            "model": self._model(),
            "messages": [message.__dict__ for message in messages],
            "stream": stream,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }

    def _headers(self) -> dict[str, str]:
        if not self.settings.openrouter_api_key:
            raise OpenRouterUnavailable("OpenRouter API key is not configured")
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            headers["X-Title"] = self.settings.openrouter_app_name
        return headers

    def _model(self) -> str:
        return self.settings.openrouter_model
