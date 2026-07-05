from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChunkRecord:
    """Upsert payload: one embedded document chunk keyed by its DocumentChunk row id."""

    id: str
    text: str
    embedding: list[float]
    scope_type: str
    scope_id: str
    document_id: str
    chunk_index: int
    page: int | None


@dataclass(frozen=True)
class RetrievedChunk:
    """A query hit; `score` is a similarity in [0, 1] where higher is more relevant."""

    id: str
    text: str
    score: float
    document_id: str
    chunk_index: int
    page: int | None


class VectorStore(Protocol):
    """Vector index for document chunks, scoped by conversation or task.

    The default adapter is embedded Chroma (`app/adapters/chroma.py`). Deployments already on
    Postgres can swap in a pgvector implementation (embedding column on document_chunks) by
    implementing this protocol; nothing above the port needs to change.
    """

    async def upsert(self, records: list[ChunkRecord]) -> None: ...

    async def query(
        self, embedding: list[float], *, scope_type: str, scope_id: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Top-k most similar chunks in the scope, sorted by score descending."""
        ...

    async def delete_document(self, document_id: str) -> None: ...

    async def delete_conversation(self, conversation_id: str) -> None: ...

    async def delete_scope(self, scope_type: str, scope_id: str) -> None: ...
