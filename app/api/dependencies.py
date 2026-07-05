from collections.abc import AsyncIterator

import httpx
from fastapi import Depends, HTTPException, Request, status
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.anthropic import AnthropicClient
from app.adapters.gemini import GeminiClient
from app.adapters.ollama import OllamaClient
from app.adapters.openai import OpenAIClient
from app.adapters.openrouter import OpenRouterClient
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.ports.llm import LLMClient
from app.services.conversations import ConversationService
from app.services.notes import NoteService
from app.services.tasks import TaskService
from app.services.tokens import TokenCounter
from app.services.translations import TranslationService

SUPPORTED_PROVIDERS = {"openrouter", "ollama", "auto", "openai", "anthropic", "gemini"}


def get_llm_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    provider = _requested_provider(request) or settings.llm_provider
    http_client = getattr(request.app.state, "http_client", None)
    return build_llm_client(settings, provider, http_client=http_client)


def build_llm_client(
    settings: Settings,
    provider: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> LLMClient:
    token_counter = TokenCounter()
    requested = (provider or settings.llm_provider).lower()

    if requested == "ollama":
        return OllamaClient(settings, token_counter, http_client)

    # These providers have no documented silent-fallback contract (unlike openrouter/auto,
    # which CLAUDE.md explicitly promises falls back to Ollama). Requesting one of them
    # without its key configured must fail loudly instead of quietly serving a different
    # provider/model with no signal to the caller.
    if requested == "openai":
        if _has_key(settings.openai_api_key):
            return OpenAIClient(settings, token_counter, http_client)
        raise _unconfigured_provider_error("openai", "OPENAI_API_KEY")
    if requested == "anthropic":
        if _has_key(settings.anthropic_api_key):
            return AnthropicClient(settings, token_counter, http_client)
        raise _unconfigured_provider_error("anthropic", "ANTHROPIC_API_KEY")
    if requested == "gemini":
        if _has_key(settings.gemini_api_key):
            return GeminiClient(settings, token_counter, http_client)
        raise _unconfigured_provider_error("gemini", "GEMINI_API_KEY")

    # requested == "openrouter" or "auto": documented to fall back to Ollama when unconfigured.
    if _has_key(settings.openrouter_api_key):
        return OpenRouterClient(settings, token_counter, http_client)

    return OllamaClient(settings, token_counter, http_client)


def _unconfigured_provider_error(provider_id: str, env_var: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"LLM provider '{provider_id}' was requested but {env_var} is not configured.",
    )


def _requested_provider(request: Request) -> str | None:
    provider = request.headers.get("X-LLM-Provider") or request.query_params.get("llm_provider")
    if provider and provider.lower() in SUPPORTED_PROVIDERS:
        return provider.lower()
    return None


def _has_key(value: str | SecretStr | None) -> bool:
    if isinstance(value, SecretStr):
        return bool(value.get_secret_value())
    return bool(value)


async def get_conversation_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> AsyncIterator[ConversationService]:
    yield ConversationService(session, settings, llm)


async def get_note_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> AsyncIterator[NoteService]:
    yield NoteService(session, settings, llm)


async def get_translation_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> AsyncIterator[TranslationService]:
    yield TranslationService(session, settings, llm)


async def get_task_service(
    request: Request,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> AsyncIterator[TaskService]:
    http_client = getattr(request.app.state, "http_client", None)
    yield TaskService(session, settings, llm, http_client)
