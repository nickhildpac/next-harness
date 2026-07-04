from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    summary = "summary"


class ConversationKind(str, Enum):
    assistant = "assistant"  # one user chatting with the LLM
    duo = "duo"  # two users chatting; the LLM drafts replies on request


class Conversation(Base, IdMixin, TimestampMixin):
    __tablename__ = "conversations"

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    second_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(
        String(32), default=ConversationKind.assistant.value, nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tone_name: Mapped[str] = mapped_column(String(64), default="professional", nullable=False)
    custom_persona: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    summary: Mapped["ConversationSummary | None"] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", uselist=False
    )


class Message(Base, IdMixin, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    role: Mapped[MessageRole] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ConversationSummary(Base, IdMixin, TimestampMixin):
    __tablename__ = "conversation_summaries"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    covered_until: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="summary")

