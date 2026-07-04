import json
import logging
from collections.abc import AsyncIterator

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Conversation, MessageRole
from app.orchestration.chat_graph import ChatGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.repositories.conversations import ConversationRepository
from app.schemas.conversation import (
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationResponse,
    ConversationToneUpdate,
    MessageCreate,
    MessageResponse,
    PaginatedMessages,
)
from app.services.memory import MemoryService
from app.services.tokens import TokenCounter
from app.services.tones import ToneService

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, session: AsyncSession, settings: Settings, llm: LLMClient):
        self.session = session
        self.settings = settings
        self.llm = llm
        self.repo = ConversationRepository(session)
        self.token_counter = TokenCounter()
        self.tones = ToneService(settings)
        self.memory = MemoryService(
            self.repo,
            llm,
            self.token_counter,
            settings.context_token_budget,
            settings.summary_trigger_tokens,
            settings.window_turn_count,
            settings.default_model,
            settings.request_timeout_seconds,
        )
        self.graph = ChatGraph(llm)

    async def create(self, payload: ConversationCreate) -> ConversationResponse:
        conversation = await self.repo.create(
            user_id=payload.user_id,
            title=payload.title,
            tone_name=payload.tone.tone_name,
            custom_persona=payload.tone.custom_persona,
        )
        await self.session.commit()
        return ConversationResponse.model_validate(conversation)

    async def list_all(self) -> list[ConversationResponse]:
        conversations = await self.repo.list_all()
        return [ConversationResponse.model_validate(conversation) for conversation in conversations]

    async def get(self, conversation_id: str) -> ConversationDetail:
        conversation = await self._conversation_or_404(conversation_id)
        return ConversationDetail(
            **ConversationResponse.model_validate(conversation).model_dump(),
            messages=[self._message_response(message) for message in conversation.messages],
            summary=conversation.summary.content if conversation.summary else None,
        )

    async def update_tone(
        self, conversation_id: str, payload: ConversationToneUpdate
    ) -> ConversationResponse:
        conversation = await self._conversation_or_404(conversation_id)
        updated = await self.repo.update_tone(
            conversation, tone_name=payload.tone_name, custom_persona=payload.custom_persona
        )
        await self.session.commit()
        return ConversationResponse.model_validate(updated)

    async def archive(self, conversation_id: str) -> None:
        conversation = await self._conversation_or_404(conversation_id)
        await self.repo.archive(conversation)
        await self.session.commit()

    async def list_messages(self, conversation_id: str, limit: int, offset: int) -> PaginatedMessages:
        await self._conversation_or_404(conversation_id)
        messages, total = await self.repo.list_messages(conversation_id, limit=limit, offset=offset)
        return PaginatedMessages(
            items=[self._message_response(message) for message in messages],
            limit=limit,
            offset=offset,
            total=total,
        )

    async def send_message(self, conversation_id: str, payload: MessageCreate) -> ChatResponse:
        conversation = await self._conversation_or_404(conversation_id)
        user_id = self._message_user_id(conversation, payload)
        user_message = await self.repo.add_message(
            conversation_id=conversation.id,
            user_id=user_id,
            role=MessageRole.user,
            content=payload.content,
            token_count=self.token_counter.count(payload.content),
        )
        tone = self.tones.resolve(payload.tone_override, conversation.tone_name, conversation.custom_persona)
        context = await self.memory.context_messages(conversation, tone.system_template)
        context.append(ChatMessage(role="user", content=payload.content))
        self._guard_context(context)
        result = await self._generate(context, tone.temperature, tone.top_p)
        assistant_message = await self.repo.add_message(
            conversation_id=conversation.id,
            user_id=user_id,
            role=MessageRole.assistant,
            content=result.content,
            token_count=result.output_tokens,
            model=result.model,
        )
        await self.memory.summarize_if_needed(conversation)
        await self.session.commit()
        logger.info(
            "message_completed",
            extra={
                "conversation_id": conversation.id,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        return ChatResponse(
            conversation_id=conversation.id,
            user_message=self._message_response(user_message),
            assistant_message=self._message_response(assistant_message),
            token_usage={"input": result.input_tokens, "output": result.output_tokens},
        )

    async def stream_message(
        self, conversation_id: str, payload: MessageCreate
    ) -> AsyncIterator[str]:
        try:
            conversation = await self._conversation_or_404(conversation_id)
            user_id = self._message_user_id(conversation, payload)
            user_message = await self.repo.add_message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.user,
                content=payload.content,
                token_count=self.token_counter.count(payload.content),
            )
            tone = self.tones.resolve(
                payload.tone_override, conversation.tone_name, conversation.custom_persona
            )
            context = await self.memory.context_messages(conversation, tone.system_template)
            context.append(ChatMessage(role="user", content=payload.content))
            self._guard_context(context)
            params = GenerationParams(
                model=self.settings.default_model,
                temperature=tone.temperature,
                top_p=tone.top_p,
                timeout_seconds=self.settings.request_timeout_seconds,
            )
            chunks: list[str] = []
            yield self._sse("message", {"role": "user", "id": user_message.id})
            async for chunk in self.llm.stream_chat(context, params):
                chunks.append(chunk)
                yield self._sse("delta", {"delta": chunk})
            content = "".join(chunks)
            assistant_message = await self.repo.add_message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.assistant,
                content=content,
                token_count=self.token_counter.count(content),
                model=self.settings.default_model,
            )
            await self.memory.summarize_if_needed(conversation)
            await self.session.commit()
            yield self._sse(
                "done",
                {
                    "user_message_id": user_message.id,
                    "assistant_message_id": assistant_message.id,
                    "output_tokens": assistant_message.token_count,
                },
            )
            yield "data: [DONE]\n\n"
        except HTTPException as exc:
            await self.session.rollback()
            yield self._sse("error", {"error": exc.detail})
            yield "data: [DONE]\n\n"
        except Exception:
            await self.session.rollback()
            logger.exception("stream_message_failed", extra={"conversation_id": conversation_id})
            yield self._sse("error", {"error": "Local LLM is unavailable or timed out."})
            yield "data: [DONE]\n\n"

    async def _generate(self, messages: list[ChatMessage], temperature: float, top_p: float) -> LLMResult:
        params = GenerationParams(
            model=self.settings.default_model,
            temperature=temperature,
            top_p=top_p,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        try:
            return await self.graph.run(messages, params)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Local LLM is unavailable or timed out.",
            ) from exc

    def _guard_context(self, messages: list[ChatMessage]) -> None:
        tokens = self.token_counter.count_messages(messages)
        if tokens > self.settings.context_token_budget:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Message exceeds configured context budget.",
            )

    async def _conversation_or_404(
        self, conversation_id: str, user_id: str | None = None
    ) -> Conversation:
        conversation = await self.repo.get(conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return conversation

    def _message_response(self, message) -> MessageResponse:
        return MessageResponse(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role.value if hasattr(message.role, "value") else message.role,
            content=message.content,
            token_count=message.token_count,
            model=message.model,
            created_at=message.created_at,
        )

    def _message_user_id(self, conversation: Conversation, payload: MessageCreate) -> str:
        if payload.user_id == "anonymous":
            return conversation.user_id
        return payload.user_id

    def _sse(self, event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"
