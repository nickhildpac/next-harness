from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Translation


class TranslationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        user_id: str,
        title: str | None,
        source_text: str,
        target_language: str,
    ) -> Translation:
        translation = Translation(
            user_id=user_id,
            title=title,
            source_text=source_text,
            target_language=target_language,
        )
        self.session.add(translation)
        await self.session.flush()
        return translation

    async def get(self, translation_id: str) -> Translation | None:
        stmt = select(Translation).where(
            Translation.id == translation_id,
            Translation.is_archived.is_(False),
        )
        return await self.session.scalar(stmt)

    async def list_for_user(self, user_id: str) -> list[Translation]:
        stmt = (
            select(Translation)
            .where(Translation.user_id == user_id, Translation.is_archived.is_(False))
            .order_by(Translation.updated_at.desc(), Translation.created_at.desc())
        )
        return list(await self.session.scalars(stmt))

    async def update_result(
        self,
        translation: Translation,
        translated_text: str,
        romanized_text: str,
        model: str,
    ) -> None:
        translation.translated_text = translated_text
        translation.romanized_text = romanized_text
        translation.model = model
        await self.session.flush()

    async def archive(self, translation: Translation) -> None:
        translation.is_archived = True
        await self.session.flush()
