import asyncio
import io
import logging
from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import AgentTask, Document
from app.ports.embeddings import EmbeddingsClient
from app.ports.vectorstore import ChunkRecord, VectorStore
from app.repositories.conversations import ConversationRepository
from app.repositories.documents import DocumentRepository
from app.schemas.document import Citation
from app.services.tokens import TokenCounter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
EMBED_BATCH_SIZE = 256


@dataclass(frozen=True)
class CitedChunk:
    """A retrieved chunk joined back to its SQL row, with the filename needed for citations."""

    id: str
    text: str
    score: float
    document_id: str
    filename: str
    chunk_index: int
    page: int | None


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        embeddings: EmbeddingsClient,
        vectorstore: VectorStore,
    ):
        self.session = session
        self.settings = settings
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.repo = DocumentRepository(session)
        self.conversations = ConversationRepository(session)
        self.token_counter = TokenCounter()

    async def ingest(
        self,
        conversation_id: str,
        *,
        filename: str,
        content_type: str | None,
        data: bytes,
        owner_user_id: str | None = None,
    ) -> Document:
        await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        return await self._ingest(
            scope_type="conversation",
            scope_id=conversation_id,
            filename=filename,
            content_type=content_type,
            data=data,
        )

    async def ingest_task_document(
        self, task_id: str, *, filename: str, content_type: str | None, data: bytes
    ) -> Document:
        await self._task_or_404(task_id)
        return await self._ingest(
            scope_type="task",
            scope_id=task_id,
            filename=filename,
            content_type=content_type,
            data=data,
        )

    async def _ingest(
        self,
        *,
        scope_type: str,
        scope_id: str,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> Document:
        if len(data) > self.settings.rag_max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Document exceeds the configured upload size limit.",
            )
        pages = await self._extract_pages(filename, data)
        chunks = self._chunk_pages(pages)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document contains no extractable text.",
            )
        document = await self.repo.add_document(
            conversation_id=scope_id if scope_type == "conversation" else None,
            task_id=scope_id if scope_type == "task" else None,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
        )
        rows = await self.repo.add_chunks(document, chunks)
        try:
            vectors = await self._embed([row.text for row in rows])
            await self.vectorstore.upsert(
                [
                    ChunkRecord(
                        id=row.id,
                        text=row.text,
                        embedding=vector,
                        scope_type=scope_type,
                        scope_id=scope_id,
                        document_id=document.id,
                        chunk_index=row.chunk_index,
                        page=row.page,
                    )
                    for row, vector in zip(rows, vectors)
                ]
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            await self._delete_vectors(document.id)
            raise
        return document

    async def list_documents(
        self, conversation_id: str, owner_user_id: str | None = None
    ) -> list[Document]:
        await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        return await self.repo.list_by_conversation(conversation_id)

    async def list_task_documents(self, task_id: str) -> list[Document]:
        await self._task_or_404(task_id)
        return await self.repo.list_by_task(task_id)

    async def delete_document(
        self, conversation_id: str, document_id: str, owner_user_id: str | None = None
    ) -> None:
        await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        document = await self.repo.get(document_id, conversation_id)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        await self.repo.delete(document)
        await self.session.commit()
        # SQL is the source of truth: retrieval drops vector hits without a matching row,
        # so a failed vector delete cannot resurface this document as a citation.
        await self._delete_vectors(document_id)

    async def retrieve(
        self, conversation_id: str, query: str, token_budget: int
    ) -> list[CitedChunk]:
        return await self._retrieve(
            scope_type="conversation",
            scope_id=conversation_id,
            query=query,
            token_budget=token_budget,
        )

    async def retrieve_task(
        self, task_id: str, query: str, token_budget: int
    ) -> list[CitedChunk]:
        return await self._retrieve(
            scope_type="task",
            scope_id=task_id,
            query=query,
            token_budget=token_budget,
        )

    async def _retrieve(
        self, *, scope_type: str, scope_id: str, query: str, token_budget: int
    ) -> list[CitedChunk]:
        if token_budget <= 0:
            return []
        # No documents means nothing to retrieve; skip the embedding round-trip so a
        # doc-less conversation with the flag on still chats normally (even without a key).
        if scope_type == "conversation":
            has_documents = await self.repo.has_documents(scope_id)
        else:
            has_documents = await self.repo.has_task_documents(scope_id)
        if not has_documents:
            return []
        try:
            query_vector = (await self._embed([query]))[0]
            hits = await self.vectorstore.query(
                query_vector,
                scope_type=scope_type,
                scope_id=scope_id,
                top_k=self.settings.rag_top_k,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document retrieval is unavailable.",
            ) from exc
        if not hits:
            return []
        rows = {row.id: row for row in await self.repo.chunks_by_ids([hit.id for hit in hits])}
        filenames = await self.repo.filenames_by_document_ids(
            list({hit.document_id for hit in hits})
        )
        candidates = [
            CitedChunk(
                id=hit.id,
                text=rows[hit.id].text,
                score=hit.score,
                document_id=hit.document_id,
                filename=filenames.get(hit.document_id, "unknown"),
                chunk_index=hit.chunk_index,
                page=hit.page,
            )
            for hit in sorted(hits, key=lambda h: h.score, reverse=True)
            if hit.id in rows  # drop vector hits orphaned by a partial delete
        ]
        # Keep highest-scored chunks while the formatted context message fits the budget.
        kept: list[CitedChunk] = []
        for candidate in candidates:
            formatted = self.format_context(kept + [candidate])
            if self.token_counter.count(formatted) + 4 > token_budget:
                break
            kept.append(candidate)
        return kept

    def format_context(self, chunks: list[CitedChunk]) -> str:
        blocks = []
        for marker, chunk in enumerate(chunks, start=1):
            source = chunk.filename if chunk.page is None else f"{chunk.filename}, p. {chunk.page}"
            blocks.append(f"[{marker}] ({source})\n{chunk.text}")
        header = (
            "Use the following document excerpts to answer when relevant. "
            "Cite sources inline as [n]."
        )
        return header + "\n\n" + "\n\n".join(blocks)

    def build_citations(self, chunks: list[CitedChunk]) -> list[Citation]:
        return [
            Citation(
                marker=marker,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                score=round(chunk.score, 4),
                snippet=chunk.text[:200],
            )
            for marker, chunk in enumerate(chunks, start=1)
        ]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                vectors.extend(await self.embeddings.embed(texts[start : start + EMBED_BATCH_SIZE]))
            return vectors
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embeddings provider is unavailable or not configured.",
            ) from exc

    async def _conversation_or_404(
        self, conversation_id: str, owner_user_id: str | None = None
    ) -> None:
        conversation = await self.conversations.get(conversation_id)
        if conversation is None or (
            owner_user_id is not None and conversation.user_id != owner_user_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

    async def _task_or_404(self, task_id: str) -> None:
        if await self.session.scalar(select(AgentTask.id).where(AgentTask.id == task_id)) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    async def _delete_vectors(self, document_id: str) -> None:
        try:
            await self.vectorstore.delete_document(document_id)
        except Exception:
            logger.exception("vector_delete_failed", extra={"document_id": document_id})

    async def _extract_pages(self, filename: str, data: bytes) -> list[tuple[int | None, str]]:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported document type. Supported: .pdf, .txt, .md",
            )
        if extension == ".pdf":
            return await asyncio.to_thread(self._extract_pdf_pages, data)
        return [(None, data.decode("utf-8", errors="replace"))]

    @staticmethod
    def _extract_pdf_pages(data: bytes) -> list[tuple[int | None, str]]:
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return [
                (number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)
            ]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read the PDF file.",
            ) from exc

    def _chunk_pages(self, pages: list[tuple[int | None, str]]) -> list[dict]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )
        chunks: list[dict] = []
        # Split per page so every chunk carries an unambiguous page number.
        for page, text in pages:
            for piece in splitter.split_text(text):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    {
                        "chunk_index": len(chunks),
                        "page": page,
                        "text": piece,
                        "token_count": self.token_counter.count(piece),
                    }
                )
        return chunks
