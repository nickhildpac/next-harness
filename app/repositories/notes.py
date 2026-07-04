from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Note

_ALLOWED_UPDATE_FIELDS = {"title", "content", "style_name", "custom_instructions"}


class NoteRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        user_id: str,
        title: str | None,
        content: str,
        style_name: str,
        custom_instructions: str | None,
    ) -> Note:
        note = Note(
            user_id=user_id,
            title=title,
            content=content,
            style_name=style_name,
            custom_instructions=custom_instructions,
        )
        self.session.add(note)
        await self.session.flush()
        return note

    async def get(self, note_id: str) -> Note | None:
        stmt = select(Note).where(Note.id == note_id, Note.is_archived.is_(False))
        return await self.session.scalar(stmt)

    async def list_for_user(self, user_id: str) -> list[Note]:
        stmt = (
            select(Note)
            .where(Note.user_id == user_id, Note.is_archived.is_(False))
            .order_by(Note.updated_at.desc(), Note.created_at.desc())
        )
        return list(await self.session.scalars(stmt))

    async def apply_updates(self, note: Note, updates: dict[str, Any]) -> Note:
        for field, value in updates.items():
            if field in _ALLOWED_UPDATE_FIELDS:
                setattr(note, field, value)
        await self.session.flush()
        return note

    async def replace_content(self, note: Note, content: str) -> Note:
        note.content = content
        await self.session.flush()
        return note

    async def archive(self, note: Note) -> None:
        note.is_archived = True
        await self.session.flush()
