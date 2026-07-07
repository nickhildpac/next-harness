from app.main import app
from app.api.routes.health import health
from app.core.config import Settings


async def test_health_endpoint():
    response = await health(Settings())

    assert response["status"] == "ok"


def test_openapi_contains_conversation_routes():
    paths = app.openapi()["paths"]

    assert "/conversations" in paths
    assert "/conversations/{conversation_id}/messages" in paths
    assert "/conversations/{conversation_id}/rag" in paths
    assert "/conversations/{conversation_id}/documents" in paths
    assert "/conversations/{conversation_id}/documents/{document_id}" in paths
    assert "/tones" in paths
    assert "/providers" in paths


def test_compatibility_payload_aliases():
    from app.schemas.conversation import ConversationCreate, ConversationToneUpdate, MessageCreate

    conversation = ConversationCreate.model_validate({"tone": "Friendly"})
    tone = ConversationToneUpdate.model_validate({"tone": "Direct"})
    message = MessageCreate.model_validate({"text": "Hello"})

    assert conversation.user_id == "anonymous"
    assert conversation.tone.tone_name == "friendly"
    assert tone.tone_name == "concise"
    assert message.content == "Hello"
    assert message.user_id == "anonymous"


def test_openrouter_provider_is_used_when_key_is_configured():
    from app.adapters.openrouter import OpenRouterClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    client = build_llm_client(
        Settings(llm_provider="openrouter", openrouter_api_key="test-key")
    )

    assert isinstance(client, OpenRouterClient)


def test_openrouter_provider_falls_back_to_ollama_without_key():
    from app.adapters.ollama import OllamaClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    client = build_llm_client(Settings(llm_provider="openrouter", openrouter_api_key=""))

    assert isinstance(client, OllamaClient)


def test_ollama_request_override_is_honored_even_with_openrouter_key():
    from app.adapters.ollama import OllamaClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    client = build_llm_client(
        Settings(llm_provider="openrouter", openrouter_api_key="test-key"),
        provider="ollama",
    )

    assert isinstance(client, OllamaClient)


def test_openai_provider_used_when_key_configured():
    from app.adapters.openai import OpenAIClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    client = build_llm_client(
        Settings(openai_api_key="test-key"),
        provider="openai",
    )

    assert isinstance(client, OpenAIClient)


def test_task_llm_client_uses_openai_even_when_default_is_openrouter():
    from types import SimpleNamespace

    from app.adapters.openai import OpenAIClient
    from app.api.dependencies import get_task_llm_client
    from app.core.config import Settings
    from app.ports.llm import GenerationParams

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(http_client=None)))
    client = get_task_llm_client(
        request,
        Settings(
            llm_provider="openrouter",
            openrouter_api_key="test-key",
            openai_api_key="test-key",
            task_openai_model="gpt-4.1-mini",
        ),
    )

    assert isinstance(client, OpenAIClient)
    assert client.resolve_model(
        GenerationParams(model="ignored", temperature=0.1, top_p=0.9, timeout_seconds=1)
    ) == "gpt-4.1-mini"


def test_task_llm_provider_can_be_configured_independently():
    from types import SimpleNamespace

    from app.adapters.openrouter import OpenRouterClient
    from app.api.dependencies import get_task_llm_client
    from app.core.config import Settings

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(http_client=None)))
    client = get_task_llm_client(
        request,
        Settings(
            llm_provider="ollama",
            task_llm_provider="openrouter",
            openrouter_api_key="test-key",
        ),
    )

    assert isinstance(client, OpenRouterClient)


def test_anthropic_provider_used_when_key_configured():
    from app.adapters.anthropic import AnthropicClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    client = build_llm_client(
        Settings(anthropic_api_key="test-key"),
        provider="anthropic",
    )

    assert isinstance(client, AnthropicClient)


def test_gemini_provider_used_when_key_configured():
    from app.adapters.gemini import GeminiClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    client = build_llm_client(
        Settings(gemini_api_key="test-key"),
        provider="gemini",
    )

    assert isinstance(client, GeminiClient)


def test_unconfigured_cloud_provider_raises_instead_of_silently_substituting():
    from fastapi import HTTPException
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    # openai has no documented fallback contract (unlike openrouter/auto), so requesting it
    # without OPENAI_API_KEY must fail loudly rather than silently serving OpenRouter's model.
    try:
        build_llm_client(
            Settings(openai_api_key=None, openrouter_api_key="test-key"),
            provider="openai",
        )
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "openai" in exc.detail
        assert "OPENAI_API_KEY" in exc.detail


def test_unconfigured_cloud_provider_raises_even_with_no_fallback_available():
    from fastapi import HTTPException
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    try:
        build_llm_client(
            Settings(anthropic_api_key=None, openrouter_api_key=None),
            provider="anthropic",
        )
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "anthropic" in exc.detail
        assert "ANTHROPIC_API_KEY" in exc.detail


def test_unconfigured_gemini_provider_raises():
    from fastapi import HTTPException
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    try:
        build_llm_client(Settings(gemini_api_key=None), provider="gemini")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "gemini" in exc.detail
        assert "GEMINI_API_KEY" in exc.detail


def test_openrouter_and_auto_still_fall_back_silently_when_unconfigured():
    """Only openrouter/auto keep the documented silent-fallback-to-Ollama contract."""
    from app.adapters.ollama import OllamaClient
    from app.api.dependencies import build_llm_client
    from app.core.config import Settings

    settings = Settings(openrouter_api_key=None)

    assert isinstance(
        build_llm_client(settings, provider="openrouter"), OllamaClient
    )
    assert isinstance(build_llm_client(settings, provider="auto"), OllamaClient)


def test_providers_endpoint_reports_availability():
    from app.api.routes.providers import list_providers
    from app.core.config import Settings

    import asyncio

    result = asyncio.run(
        list_providers(Settings(openai_api_key="k", anthropic_api_key=None))
    )
    by_id = {p["id"]: p for p in result}
    assert by_id["openai"]["available"] is True
    assert by_id["anthropic"]["available"] is False
    assert by_id["ollama"]["available"] is True
    assert set(by_id.keys()) == {"openrouter", "openai", "anthropic", "gemini", "ollama"}
