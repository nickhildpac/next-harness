from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, ConversationSummary, Message, MessageRole


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        user_id: str,
        title: str | None,
        tone_name: str,
        custom_persona: str | None,
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id, title=title, tone_name=tone_name, custom_persona=custom_persona
        )
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get(self, conversation_id: str, user_id: str | None = None) -> Conversation | None:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages), selectinload(Conversation.summary))
            .where(Conversation.id == conversation_id, Conversation.is_archived.is_(False))
        )
        if user_id:
            stmt = stmt.where(Conversation.user_id == user_id)
        return await self.session.scalar(stmt)

    async def archive(self, conversation: Conversation) -> None:
        conversation.is_archived = True
        await self.session.flush()

    async def update_tone(
        self, conversation: Conversation, *, tone_name: str, custom_persona: str | None
    ) -> Conversation:
        conversation.tone_name = tone_name
        conversation.custom_persona = custom_persona
        await self.session.flush()
        return conversation

    async def add_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        role: MessageRole,
        content: str,
        token_count: int,
        model: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            token_count=token_count,
            model=model,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def list_messages(
        self, conversation_id: str, *, limit: int, offset: int
    ) -> tuple[list[Message], int]:
        total_stmt = select(func.count()).select_from(Message).where(
            Message.conversation_id == conversation_id
        )
        total = await self.session.scalar(total_stmt)
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        messages = list(await self.session.scalars(stmt))
        return messages, int(total or 0)

    async def recent_messages(self, conversation_id: str, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id, Message.role != MessageRole.summary)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(list(await self.session.scalars(stmt))))

    async def unsummarized_messages(self, conversation: Conversation) -> list[Message]:
        covered_until = conversation.summary.covered_until if conversation.summary else None
        stmt = select(Message).where(
            Message.conversation_id == conversation.id,
            Message.role.in_([MessageRole.user, MessageRole.assistant]),
        )
        if covered_until:
            stmt = stmt.where(Message.created_at > covered_until)
        stmt = stmt.order_by(Message.created_at.asc())
        return list(await self.session.scalars(stmt))

    async def upsert_summary(
        self, conversation: Conversation, *, content: str, covered_until, token_count: int
    ) -> ConversationSummary:
        if conversation.summary:
            conversation.summary.content = content
            conversation.summary.covered_until = covered_until
            conversation.summary.token_count = token_count
            await self.session.flush()
            return conversation.summary
        summary = ConversationSummary(
            conversation_id=conversation.id,
            content=content,
            covered_until=covered_until,
            token_count=token_count,
        )
        self.session.add(summary)
        await self.session.flush()
        return summary

