from collections.abc import AsyncIterator

import httpx
from fastapi import Depends, HTTPException, Request, status
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.anthropic import AnthropicClient
from app.adapters.chroma import ChromaVectorStore
from app.adapters.gemini import GeminiClient
from app.adapters.ollama import OllamaClient
from app.adapters.openai import OpenAIClient
from app.adapters.openai_embeddings import OpenAIEmbeddingsClient
from app.adapters.openrouter import OpenRouterClient
from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_session
from app.ports.embeddings import EmbeddingsClient
from app.ports.llm import LLMClient
from app.ports.vectorstore import VectorStore
from app.services.conversations import ConversationService
from app.services.auth import AuthService, decode_access_token
from app.services.notes import NoteService
from app.services.rag import RagService
from app.services.tasks import TaskService
from app.services.tokens import TokenCounter
from app.services.translations import TranslationService

SUPPORTED_PROVIDERS = {"openrouter", "ollama", "auto", "openai", "anthropic", "gemini"}


async def get_auth_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[AuthService]:
    yield AuthService(session, settings)


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> User:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )
    user_id = decode_access_token(token, settings.auth_secret_key)
    user = await AuthService(session, settings).user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user


def get_llm_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    provider = _requested_provider(request) or settings.llm_provider
    http_client = getattr(request.app.state, "http_client", None)
    return build_llm_client(settings, provider, http_client=http_client)


def get_task_llm_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    http_client = getattr(request.app.state, "http_client", None)
    return build_llm_client(
        settings,
        settings.task_llm_provider,
        http_client=http_client,
        openai_model_override=settings.task_openai_model,
    )


def build_llm_client(
    settings: Settings,
    provider: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    openai_model_override: str | None = None,
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
            return OpenAIClient(
                settings,
                token_counter,
                http_client,
                model_override=openai_model_override,
            )
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


def get_embeddings_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EmbeddingsClient:
    # Lenient on purpose: the API key is only checked when an embed call actually happens,
    # so conversations with use_documents off never fail on a missing key.
    http_client = getattr(request.app.state, "http_client", None)
    return OpenAIEmbeddingsClient(settings, http_client)


def get_vector_store(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> VectorStore:
    # One embedded Chroma client per process, cached like http_client.
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        store = ChromaVectorStore(settings)
        request.app.state.vector_store = store
    return store


async def get_rag_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    embeddings: EmbeddingsClient = Depends(get_embeddings_client),
    vectorstore: VectorStore = Depends(get_vector_store),
) -> AsyncIterator[RagService]:
    yield RagService(session, settings, embeddings, vectorstore)


async def get_conversation_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    embeddings: EmbeddingsClient = Depends(get_embeddings_client),
    vectorstore: VectorStore = Depends(get_vector_store),
) -> AsyncIterator[ConversationService]:
    rag = RagService(session, settings, embeddings, vectorstore)
    yield ConversationService(session, settings, llm, rag=rag)


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
    llm: LLMClient = Depends(get_task_llm_client),
    embeddings: EmbeddingsClient = Depends(get_embeddings_client),
    vectorstore: VectorStore = Depends(get_vector_store),
) -> AsyncIterator[TaskService]:
    http_client = getattr(request.app.state, "http_client", None)
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    mcp_auth_token = token if scheme.lower() == "bearer" and token else None
    if not mcp_auth_token:
        mcp_auth_token = settings.mcp_http_auth_token

    mcp_http_client: httpx.AsyncClient | None = None
    transport = getattr(request.app.state, "mcp_asgi_transport", None)
    streamable_url = settings.mcp_streamable_url.lower()
    use_in_process_mcp = (
        transport is not None
        and settings.mcp_transport == "streamable_http"
        and any(host in streamable_url for host in ("127.0.0.1", "localhost", "mcp.local"))
    )
    if use_in_process_mcp:
        headers = {}
        if mcp_auth_token:
            headers["Authorization"] = f"Bearer {mcp_auth_token}"
        mcp_http_client = httpx.AsyncClient(
            transport=transport,
            base_url="http://mcp.local",
            headers=headers,
            timeout=60.0,
        )
        # Point the MCP client at the in-process mount regardless of env port.
        settings = settings.model_copy(update={"mcp_streamable_url": "http://mcp.local/mcp/"})

    try:
        yield TaskService(
            session,
            settings,
            llm,
            http_client,
            embeddings,
            vectorstore,
            mcp_auth_token=mcp_auth_token,
            mcp_http_client=mcp_http_client,
        )
    finally:
        if mcp_http_client is not None:
            await mcp_http_client.aclose()

