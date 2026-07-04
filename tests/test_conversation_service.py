from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.base import Base
from app.ports.llm import ChatMessage, GenerationParams, LLMResult
from app.schemas.conversation import ConversationCreate, MessageCreate
from app.services.conversations import ConversationService


class FakeLLM:
    def __init__(self):
        self.calls: list[list[ChatMessage]] = []

    def resolve_model(self, params: GenerationParams) -> str:
        return "fake-model"

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(content="hi there", model="fake-model", input_tokens=1, output_tokens=2)

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        for chunk in ["hi ", "there"]:
            yield chunk

    async def health(self) -> bool:
        return True


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


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
