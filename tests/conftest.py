import math
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.ports.llm import ChatMessage, GenerationParams, LLMResult
from app.ports.vectorstore import ChunkRecord, RetrievedChunk


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


class FakeEmbeddings:
    """Deterministic bag-of-words embeddings: texts sharing words rank higher on cosine."""

    DIMENSIONS = 16

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    async def health(self) -> bool:
        return True

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for word in text.lower().split():
            vector[hash(word) % self.DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeVectorStore:
    """In-memory VectorStore implementing cosine similarity over stored records."""

    def __init__(self):
        self.records: dict[str, ChunkRecord] = {}

    async def upsert(self, records: list[ChunkRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    async def query(
        self, embedding: list[float], *, scope_type: str, scope_id: str, top_k: int
    ) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(
                id=record.id,
                text=record.text,
                score=self._cosine(embedding, record.embedding),
                document_id=record.document_id,
                chunk_index=record.chunk_index,
                page=record.page,
            )
            for record in self.records.values()
            if record.scope_type == scope_type and record.scope_id == scope_id
        ]
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]

    async def delete_document(self, document_id: str) -> None:
        self.records = {
            record_id: record
            for record_id, record in self.records.items()
            if record.document_id != document_id
        }

    async def delete_conversation(self, conversation_id: str) -> None:
        await self.delete_scope("conversation", conversation_id)

    async def delete_scope(self, scope_type: str, scope_id: str) -> None:
        self.records = {
            record_id: record
            for record_id, record in self.records.items()
            if not (record.scope_type == scope_type and record.scope_id == scope_id)
        }

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()
