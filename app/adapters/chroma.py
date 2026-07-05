import asyncio

from app.core.config import Settings
from app.ports.vectorstore import ChunkRecord, RetrievedChunk, VectorStore

_COLLECTION_NAME = "document_chunks"


class ChromaVectorStore(VectorStore):
    """Embedded Chroma index; one PersistentClient per process (cache the instance on app.state).

    The chromadb client is synchronous, so every operation runs in a worker thread.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
            self._collection = client.get_or_create_collection(
                _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    async def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        await asyncio.to_thread(self._upsert_sync, records)

    def _upsert_sync(self, records: list[ChunkRecord]) -> None:
        collection = self._get_collection()
        collection.upsert(
            ids=[record.id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.text for record in records],
            metadatas=[
                {
                    "conversation_id": record.conversation_id,
                    "document_id": record.document_id,
                    "chunk_index": record.chunk_index,
                    # Chroma metadata values cannot be None.
                    "page": record.page if record.page is not None else -1,
                }
                for record in records
            ],
        )

    async def query(
        self, embedding: list[float], *, conversation_id: str, top_k: int
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._query_sync, embedding, conversation_id, top_k)

    def _query_sync(
        self, embedding: list[float], conversation_id: str, top_k: int
    ) -> list[RetrievedChunk]:
        collection = self._get_collection()
        available = collection.count()
        if available == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, available),
            where={"conversation_id": conversation_id},
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        chunks = [
            RetrievedChunk(
                id=chunk_id,
                text=text,
                score=1.0 - distance,
                document_id=metadata["document_id"],
                chunk_index=int(metadata["chunk_index"]),
                page=int(metadata["page"]) if metadata.get("page", -1) != -1 else None,
            )
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(
            lambda: self._get_collection().delete(where={"document_id": document_id})
        )

    async def delete_conversation(self, conversation_id: str) -> None:
        await asyncio.to_thread(
            lambda: self._get_collection().delete(where={"conversation_id": conversation_id})
        )
