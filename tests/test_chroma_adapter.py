from app.adapters.chroma import ChromaVectorStore
from app.core.config import Settings
from app.ports.vectorstore import ChunkRecord


def record(record_id: str, embedding: list[float], **overrides) -> ChunkRecord:
    fields = {
        "id": record_id,
        "text": f"text for {record_id}",
        "embedding": embedding,
        "conversation_id": "conv-1",
        "document_id": "doc-1",
        "chunk_index": 0,
        "page": None,
    }
    fields.update(overrides)
    return ChunkRecord(**fields)


async def test_chroma_upsert_query_delete_round_trip(tmp_path):
    store = ChromaVectorStore(Settings(chroma_persist_dir=str(tmp_path)))
    await store.upsert(
        [
            record("a", [1.0, 0.0, 0.0, 0.0], chunk_index=0, page=3),
            record("b", [0.0, 1.0, 0.0, 0.0], chunk_index=1),
            record("other", [1.0, 0.0, 0.0, 0.0], conversation_id="conv-2", document_id="doc-2"),
        ]
    )

    hits = await store.query([1.0, 0.0, 0.0, 0.0], conversation_id="conv-1", top_k=5)

    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score
    assert hits[0].page == 3
    assert hits[1].page is None
    assert hits[0].document_id == "doc-1"

    await store.delete_document("doc-1")
    assert await store.query([1.0, 0.0, 0.0, 0.0], conversation_id="conv-1", top_k=5) == []
    # The other conversation's vectors are untouched.
    assert await store.query([1.0, 0.0, 0.0, 0.0], conversation_id="conv-2", top_k=5) != []

    await store.delete_conversation("conv-2")
    assert await store.query([1.0, 0.0, 0.0, 0.0], conversation_id="conv-2", top_k=5) == []
