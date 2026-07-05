from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_document(
        self,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
        filename: str,
        content_type: str | None,
        size_bytes: int,
    ) -> Document:
        document = Document(
            conversation_id=conversation_id,
            task_id=task_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def add_chunks(self, document: Document, chunks: list[dict]) -> list[DocumentChunk]:
        rows = [
            DocumentChunk(
                document_id=document.id,
                conversation_id=document.conversation_id,
                task_id=document.task_id,
                chunk_index=chunk["chunk_index"],
                page=chunk.get("page"),
                text=chunk["text"],
                token_count=chunk["token_count"],
            )
            for chunk in chunks
        ]
        self.session.add_all(rows)
        document.chunk_count = len(rows)
        await self.session.flush()
        return rows

    async def get(self, document_id: str, conversation_id: str) -> Document | None:
        stmt = select(Document).where(
            Document.id == document_id, Document.conversation_id == conversation_id
        )
        return await self.session.scalar(stmt)

    async def get_for_task(self, document_id: str, task_id: str) -> Document | None:
        stmt = select(Document).where(Document.id == document_id, Document.task_id == task_id)
        return await self.session.scalar(stmt)

    async def has_documents(self, conversation_id: str) -> bool:
        stmt = select(Document.id).where(Document.conversation_id == conversation_id).limit(1)
        return await self.session.scalar(stmt) is not None

    async def has_task_documents(self, task_id: str) -> bool:
        stmt = select(Document.id).where(Document.task_id == task_id).limit(1)
        return await self.session.scalar(stmt) is not None

    async def list_by_conversation(self, conversation_id: str) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.conversation_id == conversation_id)
            .order_by(Document.created_at.asc(), Document.id.asc())
        )
        return list(await self.session.scalars(stmt))

    async def list_by_task(self, task_id: str) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.task_id == task_id)
            .order_by(Document.created_at.asc(), Document.id.asc())
        )
        return list(await self.session.scalars(stmt))

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)
        await self.session.flush()

    async def chunks_by_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        if not chunk_ids:
            return []
        stmt = select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        return list(await self.session.scalars(stmt))

    async def filenames_by_document_ids(self, document_ids: list[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        stmt = select(Document.id, Document.filename).where(Document.id.in_(document_ids))
        result = await self.session.execute(stmt)
        return {row.id: row.filename for row in result}
