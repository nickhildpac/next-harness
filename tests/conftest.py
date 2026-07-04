from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.ports.llm import ChatMessage, GenerationParams, LLMResult


class FakeLLM:
    def __init__(self, reply: str = "hi there"):
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def resolve_model(self, params: GenerationParams) -> str:
        return "fake-model"

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(content=self.reply, model="fake-model", input_tokens=1, output_tokens=2)

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
