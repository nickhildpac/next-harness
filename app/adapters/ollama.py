import json
from collections.abc import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.services.tokens import TokenCounter


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient(LLMClient):
    def __init__(self, settings: Settings, token_counter: TokenCounter):
        self.settings = settings
        self.token_counter = token_counter
        self.base_url = str(settings.ollama_base_url).rstrip("/")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
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
        payload = {
            "model": params.model,
            "messages": [message.__dict__ for message in messages],
            "stream": False,
            "options": {"temperature": params.temperature, "top_p": params.top_p},
        }
        async with httpx.AsyncClient(timeout=params.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
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
        payload = {
            "model": params.model,
            "messages": [message.__dict__ for message in messages],
            "stream": True,
            "options": {"temperature": params.temperature, "top_p": params.top_p},
        }
        async with httpx.AsyncClient(timeout=params.timeout_seconds) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

