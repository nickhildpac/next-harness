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
