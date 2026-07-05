import io

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Document, DocumentChunk
from app.ports.vectorstore import ChunkRecord
from app.repositories.conversations import ConversationRepository
from app.services.rag import RagService
from app.services.tokens import TokenCounter
from tests.conftest import FakeEmbeddings, FakeVectorStore


def build_pdf(pages: list[str]) -> bytes:
    """Assemble a minimal valid PDF with one line of text per page."""
    page_numbers = [3 + index * 2 for index in range(len(pages))]
    font_number = 3 + len(pages) * 2
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects: list[tuple[int, str]] = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>"),
    ]
    for number, text in zip(page_numbers, pages):
        objects.append(
            (
                number,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_number} 0 R >> >> "
                f"/Contents {number + 1} 0 R >>",
            )
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append((number + 1, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))
    objects.append((font_number, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in objects:
        offsets[number] = out.tell()
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref_position = out.tell()
    count = len(objects) + 1
    out.write(f"xref\n0 {count}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for number in sorted(offsets):
        out.write(f"{offsets[number]:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF".encode()
    )
    return out.getvalue()


async def make_conversation(session, *, use_documents: bool = False):
    repo = ConversationRepository(session)
    conversation = await repo.create(
        user_id="alice",
        title=None,
        tone_name="professional",
        custom_persona=None,
        use_documents=use_documents,
    )
    await session.commit()
    return conversation


def make_rag(session, settings: Settings | None = None) -> RagService:
    return RagService(session, settings or Settings(), FakeEmbeddings(), FakeVectorStore())


async def test_ingest_txt_creates_chunks_and_vectors(session):
    conversation = await make_conversation(session)
    rag = make_rag(session, Settings(rag_chunk_size=60, rag_chunk_overlap=10))
    text = " ".join(f"word{i}" for i in range(60))

    document = await rag.ingest(
        conversation.id, filename="notes.txt", content_type="text/plain", data=text.encode()
    )

    rows = list(
        await session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
    )
    assert document.chunk_count == len(rows) > 1
    assert [row.chunk_index for row in sorted(rows, key=lambda r: r.chunk_index)] == list(
        range(len(rows))
    )
    assert all(row.page is None for row in rows)
    assert all(row.conversation_id == conversation.id for row in rows)
    assert set(rag.vectorstore.records) == {row.id for row in rows}


async def test_ingest_pdf_records_page_numbers(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)

    document = await rag.ingest(
        conversation.id,
        filename="report.pdf",
        content_type="application/pdf",
        data=build_pdf(["Alpha facts on the first page", "Beta facts on the second page"]),
    )

    rows = list(
        await session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
    )
    assert [row.page for row in rows] == [1, 2]
    assert "Alpha" in rows[0].text and "Beta" in rows[1].text


async def test_ingest_rejects_unsupported_extension(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)

    with pytest.raises(HTTPException) as exc:
        await rag.ingest(conversation.id, filename="deck.docx", content_type=None, data=b"x")
    assert exc.value.status_code == 400


async def test_ingest_rejects_oversize_upload(session):
    conversation = await make_conversation(session)
    rag = make_rag(session, Settings(rag_max_upload_bytes=10))

    with pytest.raises(HTTPException) as exc:
        await rag.ingest(conversation.id, filename="notes.txt", content_type=None, data=b"x" * 11)
    assert exc.value.status_code == 413


async def test_ingest_rejects_empty_text(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)

    with pytest.raises(HTTPException) as exc:
        await rag.ingest(conversation.id, filename="empty.txt", content_type=None, data=b"   ")
    assert exc.value.status_code == 400


async def test_ingest_unknown_conversation_404(session):
    rag = make_rag(session)

    with pytest.raises(HTTPException) as exc:
        await rag.ingest("missing", filename="notes.txt", content_type=None, data=b"hello")
    assert exc.value.status_code == 404


async def test_delete_document_removes_rows_and_vectors(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)
    document = await rag.ingest(
        conversation.id, filename="notes.txt", content_type=None, data=b"delete me soon"
    )

    await rag.delete_document(conversation.id, document.id)

    assert await session.scalar(select(Document).where(Document.id == document.id)) is None
    assert (
        await session.scalar(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        is None
    )
    assert rag.vectorstore.records == {}

    with pytest.raises(HTTPException) as exc:
        await rag.delete_document(conversation.id, document.id)
    assert exc.value.status_code == 404


async def test_retrieve_returns_relevant_chunk_with_metadata(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)
    await rag.ingest(
        conversation.id,
        filename="fruit.txt",
        content_type=None,
        data=b"apples oranges bananas make a great fruit salad",
    )
    await rag.ingest(
        conversation.id,
        filename="cars.txt",
        content_type=None,
        data=b"engines wheels highways and fast cars",
    )

    chunks = await rag.retrieve(conversation.id, "fruit salad with apples", token_budget=10_000)

    assert chunks
    assert chunks[0].filename == "fruit.txt"
    assert chunks == sorted(chunks, key=lambda chunk: chunk.score, reverse=True)


async def test_retrieve_drops_orphan_vector_hits(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)
    await rag.vectorstore.upsert(
        [
            ChunkRecord(
                id="orphan",
                text="ghost chunk with no sql row",
                embedding=(await rag.embeddings.embed(["ghost chunk with no sql row"]))[0],
                conversation_id=conversation.id,
                document_id="ghost-doc",
                chunk_index=0,
                page=None,
            )
        ]
    )

    chunks = await rag.retrieve(conversation.id, "ghost chunk", token_budget=10_000)

    assert chunks == []


async def test_retrieve_trims_lowest_scored_chunks_to_budget(session):
    conversation = await make_conversation(session)
    rag = make_rag(session)
    await rag.ingest(
        conversation.id,
        filename="fruit.txt",
        content_type=None,
        data=b"apples oranges bananas make a great fruit salad",
    )
    await rag.ingest(
        conversation.id,
        filename="cars.txt",
        content_type=None,
        data=b"engines wheels highways and fast cars",
    )
    counter = TokenCounter()
    both = await rag.retrieve(conversation.id, "fruit salad with apples", token_budget=10_000)
    assert len(both) == 2
    cost_one = counter.count(rag.format_context(both[:1])) + 4
    cost_both = counter.count(rag.format_context(both)) + 4

    kept = await rag.retrieve(
        conversation.id, "fruit salad with apples", token_budget=cost_both - 1
    )
    assert len(kept) == 1
    assert kept[0].id == both[0].id  # the lowest-scored chunk was dropped

    assert (
        await rag.retrieve(conversation.id, "fruit salad with apples", token_budget=cost_one - 1)
        == []
    )
    assert await rag.retrieve(conversation.id, "fruit salad", token_budget=0) == []
