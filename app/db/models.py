from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
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


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Conversation(Base, IdMixin, TimestampMixin):
    __tablename__ = "conversations"

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    second_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    participant_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    participant_second_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(
        String(32), default=ConversationKind.assistant.value, nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tone_name: Mapped[str] = mapped_column(String(64), default="professional", nullable=False)
    custom_persona: Mapped[str | None] = mapped_column(Text, nullable=True)
    use_documents: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    summary: Mapped["ConversationSummary | None"] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", uselist=False
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
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
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)

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


class Document(Base, IdMixin, TimestampMixin):
    __tablename__ = "documents"

    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=True
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True, nullable=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    conversation: Mapped[Conversation | None] = relationship(back_populates="documents")
    task: Mapped["AgentTask | None"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )


class DocumentChunk(Base, IdMixin, TimestampMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (Index("ix_document_chunks_doc_index", "document_id", "chunk_index"),)

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Denormalized so retrieval can join hits to rows without going through documents.
    conversation_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-based; None for txt/md
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Note(Base, IdMixin, TimestampMixin):
    __tablename__ = "notes"
    __table_args__ = (Index("ix_notes_user_updated", "user_id", "updated_at"),)

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    style_name: Mapped[str] = mapped_column(String(64), default="default", nullable=False)
    custom_instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Translation(Base, IdMixin, TimestampMixin):
    __tablename__ = "translations"
    __table_args__ = (Index("ix_translations_user_updated", "user_id", "updated_at"),)

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_language: Mapped[str] = mapped_column(String(64), nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    turns: Mapped[list["TranslationTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="TranslationTurn.turn_index",
    )


class TranslationTurn(Base, IdMixin, TimestampMixin):
    __tablename__ = "translation_turns"
    __table_args__ = (Index("ix_translation_turns_session_index", "translation_id", "turn_index"),)

    translation_id: Mapped[str] = mapped_column(
        ForeignKey("translations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_language: Mapped[str] = mapped_column(String(64), nullable=False)
    translated_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    romanized_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    session: Mapped[Translation] = relationship(back_populates="turns")


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskStepKind(str, Enum):
    thought = "thought"
    tool_call = "tool_call"
    tool_result = "tool_result"
    final = "final"
    error = "error"


class AgentTask(Base, IdMixin, TimestampMixin):
    __tablename__ = "agent_tasks"
    __table_args__ = (Index("ix_agent_tasks_user_updated", "user_id", "updated_at"),)

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=TaskStatus.pending.value, nullable=False, index=True
    )
    max_steps: Mapped[int] = mapped_column(Integer, default=8, nullable=False)
    steps_taken: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    allowed_tools: Mapped[list | None] = mapped_column(JSON, nullable=True)

    steps: Mapped[list["AgentTaskStep"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AgentTaskStep.step_index",
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class AgentTaskStep(Base, IdMixin, TimestampMixin):
    __tablename__ = "agent_task_steps"
    __table_args__ = (Index("ix_agent_task_steps_task_index", "task_id", "step_index"),)

    task_id: Mapped[str] = mapped_column(
        ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    task: Mapped[AgentTask] = relationship(back_populates="steps")
