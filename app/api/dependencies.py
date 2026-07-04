from collections.abc import AsyncIterator

from fastapi import Depends, Request
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.ollama import OllamaClient
from app.adapters.openrouter import OpenRouterClient
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.ports.llm import LLMClient
from app.services.conversations import ConversationService
from app.services.tokens import TokenCounter


def get_llm_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    provider = _requested_provider(request) or settings.llm_provider
    return build_llm_client(settings, provider)


def build_llm_client(settings: Settings, provider: str | None = None) -> LLMClient:
    token_counter = TokenCounter()
    requested = (provider or settings.llm_provider).lower()
    if requested == "ollama":
        return OllamaClient(settings, token_counter)
    if requested in {"openrouter", "auto"} and _has_openrouter_key(settings):
        return OpenRouterClient(settings, token_counter)
    return OllamaClient(settings, token_counter)


def _requested_provider(request: Request) -> str | None:
    provider = request.headers.get("X-LLM-Provider") or request.query_params.get("llm_provider")
    if provider and provider.lower() in {"openrouter", "ollama", "auto"}:
        return provider.lower()
    return None


def _has_openrouter_key(settings: Settings) -> bool:
    key = settings.openrouter_api_key
    if isinstance(key, SecretStr):
        return bool(key.get_secret_value())
    return bool(key)


async def get_conversation_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> AsyncIterator[ConversationService]:
    yield ConversationService(session, settings, llm)
