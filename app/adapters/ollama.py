import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.services.tokens import TokenCounter


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient(LLMClient):
    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.token_counter = token_counter
        self.base_url = str(settings.ollama_base_url).rstrip("/")
        self._http = http_client

    def resolve_model(self, params: GenerationParams) -> str:
        return params.model

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http is not None:
            yield self._http
        else:
            async with httpx.AsyncClient() as client:
                yield client

    async def health(self) -> bool:
        try:
            async with self._client() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5)
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OllamaUnavailable)),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, params, stream=False),
                timeout=params.timeout_seconds,
            )
        if response.status_code >= 500:
            raise OllamaUnavailable("Ollama returned an unavailable status")
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResult(
            content=content,
            model=data.get("model", params.model),
            input_tokens=self.token_counter.count_messages(messages),
            output_tokens=self.token_counter.count(content),
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        async with self._client() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=self._payload(messages, params, stream=True),
                timeout=params.timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    def _payload(
        self, messages: list[ChatMessage], params: GenerationParams, *, stream: bool
    ) -> dict:
        return {
            "model": params.model,
            "messages": [message.__dict__ for message in messages],
            "stream": stream,
            "options": {"temperature": params.temperature, "top_p": params.top_p},
        }
