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
    assert "/tones" in paths


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
