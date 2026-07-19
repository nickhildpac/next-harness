import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.adapters.openai_tools import parse_openai_tool_calls, to_openai, tool_choice_openai
from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult, ToolChoice, ToolSpec
from app.services.tokens import TokenCounter


class OpenAIUnavailable(RuntimeError):
    pass


class OpenAIClient(LLMClient):
    def __init__(
        self,
        settings: Settings,
        token_counter: TokenCounter,
        http_client: httpx.AsyncClient | None = None,
        model_override: str | None = None,
    ):
        self.settings = settings
        self.token_counter = token_counter
        self._http = http_client
        self._client_instance: AsyncOpenAI | None = None
        self.model_override = model_override

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
        return self._model()

    def supports_native_tools(self) -> bool:
        return True

    async def health(self) -> bool:
        return bool(self.settings.openai_api_key)

    async def chat(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResult:
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = [to_openai(t) for t in tools]
            extra["tool_choice"] = tool_choice_openai(tool_choice or ToolChoice(mode="auto"))
        response = await self._client().chat.completions.create(
            model=self._model(),
            messages=[self._to_openai_message(m) for m in messages],
            temperature=params.temperature,
            top_p=params.top_p,
            timeout=params.timeout_seconds,
            **extra,
        )
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        content = (message.content if message else "") or ""
        usage = response.usage
        return LLMResult(
            content=content,
            model=response.model or self._model(),
            input_tokens=(usage.prompt_tokens if usage else None)
            or self.token_counter.count_messages(messages),
            output_tokens=(usage.completion_tokens if usage else None)
            or self.token_counter.count(content),
            tool_calls=parse_openai_tool_calls(message) if message else (),
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        stream = await self._client().chat.completions.create(
            model=self._model(),
            messages=[self._to_openai_message(m) for m in messages],
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

    def _model(self) -> str:
        return self.model_override or self.settings.openai_model

    def _to_openai_message(self, message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            payload["content"] = message.content or None
            payload["tool_calls"] = [
                {
                    "id": call.call_id or f"call_{i}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for i, call in enumerate(message.tool_calls)
            ]
        if message.role == "tool":
            payload["tool_call_id"] = message.tool_call_id
        return payload
