import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Message, MessageRole
from app.schemas.conversation import ConversationCreate, ConversationRagUpdate, MessageCreate
from app.services.conversations import ConversationService
from app.services.rag import RagService
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore


class BoomEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embeddings down")

    async def health(self) -> bool:
        return False


def make_services(session, settings: Settings | None = None, embeddings=None):
    settings = settings or Settings()
    llm = FakeLLM()
    rag = RagService(session, settings, embeddings or FakeEmbeddings(), FakeVectorStore())
    service = ConversationService(session, settings, llm, rag=rag)
    return service, rag, llm


async def ingest_fruit_doc(rag: RagService, conversation_id: str) -> None:
    await rag.ingest(
        conversation_id,
        filename="fruit.txt",
        content_type="text/plain",
        data=b"apples oranges bananas make a great fruit salad",
    )


async def test_flag_off_chat_is_unchanged(session):
    service, rag, llm = make_services(session)
    conversation = await service.create(ConversationCreate())

    response = await service.send_message(conversation.id, MessageCreate(content="hello"))

    assert rag.embeddings.calls == []
    assert [message.role for message in llm.calls[0]] == ["system", "user"]
    assert response.citations == []
    assert response.assistant_message.citations is None


async def test_flag_on_injects_context_and_returns_citations(session):
    service, rag, llm = make_services(session)
    conversation = await service.create(ConversationCreate(use_documents=True))
    await ingest_fruit_doc(rag, conversation.id)

    response = await service.send_message(
        conversation.id, MessageCreate(content="fruit salad with apples")
    )

    context = llm.calls[0]
    assert context[1].role == "system"
    assert "[1]" in context[1].content and "fruit.txt" in context[1].content
    assert context[-1].role == "user"

    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.marker == 1
    assert citation.filename == "fruit.txt"
    assert citation.snippet.startswith("apples oranges")

    assistant_row = await session.scalar(
        select(Message).where(
            Message.conversation_id == conversation.id, Message.role == MessageRole.assistant
        )
    )
    assert assistant_row.citations and assistant_row.citations[0]["filename"] == "fruit.txt"
    assert response.assistant_message.citations[0].filename == "fruit.txt"


async def test_rag_toggle_endpoint_updates_flag(session):
    service, rag, llm = make_services(session)
    conversation = await service.create(ConversationCreate())
    assert conversation.use_documents is False

    updated = await service.update_rag(conversation.id, ConversationRagUpdate(use_documents=True))
    assert updated.use_documents is True

    await ingest_fruit_doc(rag, conversation.id)
    response = await service.send_message(
        conversation.id, MessageCreate(content="fruit salad with apples")
    )
    assert response.citations


async def test_tiny_rag_budget_skips_injection(session):
    service, rag, llm = make_services(session, Settings(rag_token_budget=1))
    conversation = await service.create(ConversationCreate(use_documents=True))
    await ingest_fruit_doc(rag, conversation.id)

    response = await service.send_message(
        conversation.id, MessageCreate(content="fruit salad with apples")
    )

    assert response.citations == []
    assert [message.role for message in llm.calls[0]] == ["system", "user"]


async def test_stream_emits_citations_before_deltas_when_flag_on(session):
    service, rag, llm = make_services(session)
    conversation = await service.create(ConversationCreate(use_documents=True))
    await ingest_fruit_doc(rag, conversation.id)

    events = [
        event
        async for event in service.stream_message(
            conversation.id, MessageCreate(content="fruit salad with apples")
        )
    ]

    citation_index = next(
        i for i, event in enumerate(events) if event.startswith("event: citations")
    )
    first_delta_index = next(
        i for i, event in enumerate(events) if event.startswith("event: delta")
    )
    assert citation_index < first_delta_index
    assert events[-2].startswith("event: done")
    assert events[-1] == "data: [DONE]\n\n"

    assistant_row = await session.scalar(
        select(Message).where(
            Message.conversation_id == conversation.id, Message.role == MessageRole.assistant
        )
    )
    assert assistant_row.citations and assistant_row.citations[0]["filename"] == "fruit.txt"


async def test_stream_has_no_citations_event_when_flag_off(session):
    service, rag, llm = make_services(session)
    conversation = await service.create(ConversationCreate())

    events = [
        event
        async for event in service.stream_message(conversation.id, MessageCreate(content="hello"))
    ]

    assert not any(event.startswith("event: citations") for event in events)
    assert rag.embeddings.calls == []


async def test_flag_on_without_documents_chats_normally(session):
    # Embeddings would blow up if called; with no documents they must not be.
    service, rag, llm = make_services(session, embeddings=BoomEmbeddings())
    conversation = await service.create(ConversationCreate(use_documents=True))

    response = await service.send_message(conversation.id, MessageCreate(content="hello"))

    assert response.citations == []
    assert [message.role for message in llm.calls[0]] == ["system", "user"]


async def test_retrieval_failure_fails_loudly(session):
    service, rag, llm = make_services(session)
    conversation = await service.create(ConversationCreate(use_documents=True))
    await ingest_fruit_doc(rag, conversation.id)
    rag.embeddings = BoomEmbeddings()  # embeddings go down after ingestion

    with pytest.raises(HTTPException) as exc:
        await service.send_message(conversation.id, MessageCreate(content="hello"))
    assert exc.value.status_code == 503

    rows = list(
        await session.scalars(select(Message).where(Message.conversation_id == conversation.id))
    )
    assert rows == []  # nothing persisted on failure

    events = [
        event
        async for event in service.stream_message(conversation.id, MessageCreate(content="hello"))
    ]
    assert any(event.startswith("event: error") for event in events)
