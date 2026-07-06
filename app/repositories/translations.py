from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Translation, TranslationTurn


class TranslationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(
        self,
        *,
        user_id: str,
        title: str | None,
        target_language: str,
    ) -> Translation:
        translation = Translation(
            user_id=user_id,
            title=title,
            target_language=target_language,
        )
        self.session.add(translation)
        await self.session.flush()
        return translation

    async def create_turn(
        self,
        *,
        session: Translation,
        turn_index: int,
        source_text: str,
        target_language: str,
    ) -> TranslationTurn:
        turn = TranslationTurn(
            translation_id=session.id,
            turn_index=turn_index,
            source_text=source_text,
            target_language=target_language,
        )
        self.session.add(turn)
        await self.session.flush()
        return turn

    async def get(self, translation_id: str) -> Translation | None:
        stmt = select(Translation).where(
            Translation.id == translation_id,
            Translation.is_archived.is_(False),
        )
        return await self.session.scalar(stmt)

    async def get_with_turns(self, translation_id: str) -> Translation | None:
        stmt = (
            select(Translation)
            .options(selectinload(Translation.turns))
            .where(Translation.id == translation_id, Translation.is_archived.is_(False))
        )
        return await self.session.scalar(stmt)

    async def list_for_user(self, user_id: str) -> list[Translation]:
        stmt = (
            select(Translation)
            .options(selectinload(Translation.turns))
            .where(Translation.user_id == user_id, Translation.is_archived.is_(False))
            .order_by(Translation.updated_at.desc(), Translation.created_at.desc())
        )
        return list(await self.session.scalars(stmt))

    async def turn_count(self, translation_id: str) -> int:
        stmt = select(func.count()).select_from(TranslationTurn).where(
            TranslationTurn.translation_id == translation_id
        )
        return int(await self.session.scalar(stmt) or 0)

    async def update_turn_result(
        self,
        turn: TranslationTurn,
        translated_text: str,
        romanized_text: str,
        model: str,
    ) -> None:
        turn.translated_text = translated_text
        turn.romanized_text = romanized_text
        turn.model = model
        await self.session.flush()

    async def touch_session(self, session: Translation) -> None:
        await self.session.flush()

    async def archive(self, translation: Translation) -> None:
        translation.is_archived = True
        await self.session.flush()
