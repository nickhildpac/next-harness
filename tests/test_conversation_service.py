import json

import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.ports.llm import GenerationParams
from app.schemas.conversation import ConversationCreate, MessageCreate
from app.services.conversations import ConversationService

from conftest import FakeLLM


class OpenAIClient(FakeLLM):
    async def chat(self, messages, params: GenerationParams):
        raise RuntimeError(
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
            "sk-proj-secret. You can find your API key at "
            "https://platform.openai.com/account/api-keys.', 'code': 'invalid_api_key'}}"
        )

    async def stream_chat(self, messages, params: GenerationParams):
        raise RuntimeError(
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
            "sk-proj-secret. You can find your API key at "
            "https://platform.openai.com/account/api-keys.', 'code': 'invalid_api_key'}}"
        )
        yield ""


async def test_user_message_appears_once_in_llm_context(session):
    llm = FakeLLM()
    service = ConversationService(session, Settings(), llm)
    conversation = await service.create(ConversationCreate())

    await service.send_message(conversation.id, MessageCreate(content="hello"))

    context = llm.calls[0]
    user_turns = [m for m in context if m.role == "user" and m.content == "hello"]
    assert len(user_turns) == 1


async def test_stream_records_resolved_model(session):
    llm = FakeLLM()
    service = ConversationService(session, Settings(), llm)
    conversation = await service.create(ConversationCreate())

    events = [event async for event in service.stream_message(conversation.id, MessageCreate(content="hello"))]

    assert any("delta" in event for event in events)
    messages = await service.list_messages(conversation.id, limit=10, offset=0)
    assistant = [m for m in messages.items if m.role == "assistant"]
    assert assistant and assistant[0].model == "fake-model"
    assert assistant[0].content == "hi there"


async def test_unknown_stored_tone_falls_back_instead_of_500(session):
    llm = FakeLLM()
    service = ConversationService(session, Settings(), llm)
    conversation = await service.create(ConversationCreate())
    db_conversation = await service.repo.get(conversation.id)
    db_conversation.tone_name = "retired-tone"
    await session.commit()

    response = await service.send_message(conversation.id, MessageCreate(content="hello"))

    assert response.assistant_message.content == "hi there"


async def test_send_message_reports_openai_errors(session):
    service = ConversationService(session, Settings(), OpenAIClient())
    conversation = await service.create(ConversationCreate())

    with pytest.raises(HTTPException) as exc:
        await service.send_message(conversation.id, MessageCreate(content="hello"))

    assert exc.value.status_code == 503
    assert exc.value.detail == "OpenAI request failed: invalid API key"


async def test_stream_message_reports_openai_errors(session):
    service = ConversationService(session, Settings(), OpenAIClient())
    conversation = await service.create(ConversationCreate())

    events = [
        event async for event in service.stream_message(conversation.id, MessageCreate(content="hello"))
    ]
    error_event = next(event for event in events if event.startswith("event: error"))
    payload = json.loads(error_event.split("data: ", 1)[1])

    assert payload["error"] == "OpenAI request failed: invalid API key"
