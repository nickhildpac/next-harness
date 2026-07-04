import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.schemas.conversation import ConversationCreate, MessageCreate, SuggestRequest
from app.services.conversations import ConversationService

from conftest import FakeLLM


def make_service(session, llm) -> ConversationService:
    return ConversationService(session, Settings(), llm)


async def make_duo(service: ConversationService):
    return await service.create(
        ConversationCreate.model_validate(
            {"participants": ["alice", "bob"], "title": "planning", "tone": "friendly"}
        )
    )


async def test_create_duo_conversation(session):
    service = make_service(session, FakeLLM())

    conversation = await make_duo(service)

    assert conversation.kind == "duo"
    assert conversation.user_id == "alice"
    assert conversation.second_user_id == "bob"


async def test_participants_must_be_two_distinct_users():
    with pytest.raises(ValueError):
        ConversationCreate.model_validate({"participants": ["alice", "alice"]})
    with pytest.raises(ValueError):
        ConversationCreate.model_validate({"participants": ["alice"]})


async def test_duo_message_persists_without_llm_reply(session):
    llm = FakeLLM()
    service = make_service(session, llm)
    conversation = await make_duo(service)

    response = await service.send_message(
        conversation.id, MessageCreate(user_id="bob", content="hey alice")
    )

    assert response.assistant_message is None
    assert response.user_message.user_id == "bob"
    assert llm.calls == []


async def test_duo_message_rejects_non_participants(session):
    service = make_service(session, FakeLLM())
    conversation = await make_duo(service)

    with pytest.raises(HTTPException) as exc:
        await service.send_message(
            conversation.id, MessageCreate(user_id="mallory", content="hi")
        )

    assert exc.value.status_code == 400


async def test_suggest_maps_roles_around_target_user(session):
    llm = FakeLLM(reply="Sounds good, see you at 6!")
    service = make_service(session, llm)
    conversation = await make_duo(service)
    await service.send_message(conversation.id, MessageCreate(user_id="alice", content="dinner tonight?"))
    await service.send_message(conversation.id, MessageCreate(user_id="bob", content="sure, where?"))
    await service.send_message(conversation.id, MessageCreate(user_id="alice", content="tapas at 6?"))

    suggestion = await service.suggest_reply(conversation.id, SuggestRequest(for_user="bob"))

    assert suggestion.content == "Sounds good, see you at 6!"
    assert suggestion.model == "fake-model"
    assert suggestion.message is None
    context = llm.calls[0]
    assert context[0].role == "system" and "'bob'" in context[0].content
    by_content = {m.content: m.role for m in context}
    assert by_content["dinner tonight?"] == "user"
    assert by_content["sure, where?"] == "assistant"
    assert by_content["tapas at 6?"] == "user"
    # No suggestion is persisted unless asked for.
    messages = await service.list_messages(conversation.id, limit=10, offset=0)
    assert len(messages.items) == 3


async def test_suggest_persist_stores_message_as_target_user(session):
    llm = FakeLLM(reply="On my way.")
    service = make_service(session, llm)
    conversation = await make_duo(service)
    await service.send_message(conversation.id, MessageCreate(user_id="alice", content="you close?"))

    suggestion = await service.suggest_reply(
        conversation.id, SuggestRequest.model_validate({"as_user": "bob", "persist": True})
    )

    assert suggestion.message is not None
    assert suggestion.message.user_id == "bob"
    assert suggestion.message.model == "fake-model"
    messages = await service.list_messages(conversation.id, limit=10, offset=0)
    assert [m.content for m in messages.items] == ["you close?", "On my way."]


async def test_suggest_honors_tone_override(session):
    llm = FakeLLM()
    service = make_service(session, llm)
    conversation = await make_duo(service)

    await service.suggest_reply(
        conversation.id,
        SuggestRequest.model_validate({"for_user": "alice", "tone_override": "concise"}),
    )

    system_prompt = llm.calls[0][0].content
    assert "concise" in system_prompt.lower()


async def test_suggest_rejected_for_assistant_conversations(session):
    service = make_service(session, FakeLLM())
    conversation = await service.create(ConversationCreate())

    with pytest.raises(HTTPException) as exc:
        await service.suggest_reply(conversation.id, SuggestRequest(for_user="anonymous"))

    assert exc.value.status_code == 400


async def test_suggest_rejects_non_participants(session):
    service = make_service(session, FakeLLM())
    conversation = await make_duo(service)

    with pytest.raises(HTTPException) as exc:
        await service.suggest_reply(conversation.id, SuggestRequest(for_user="mallory"))

    assert exc.value.status_code == 400
