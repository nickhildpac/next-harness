import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Conversation, ConversationKind, Message, MessageRole
from app.orchestration.chat_graph import ChatGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.repositories.conversations import ConversationRepository
from app.schemas.conversation import (
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationRagUpdate,
    ConversationResponse,
    ConversationToneUpdate,
    MessageCreate,
    MessageResponse,
    PaginatedMessages,
    SuggestRequest,
    SuggestResponse,
)
from app.schemas.document import Citation
from app.services.memory import MemoryService
from app.services.llm_errors import llm_error_detail
from app.services.rag import CitedChunk, RagService
from app.services.tokens import TokenCounter
from app.services.tones import ToneService

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        llm: LLMClient,
        rag: RagService | None = None,
    ):
        self.session = session
        self.settings = settings
        self.llm = llm
        self.rag = rag
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

    async def create(
        self, payload: ConversationCreate, owner_user_id: str | None = None
    ) -> ConversationResponse:
        if payload.participants:
            participant_user_id, participant_second_user_id = payload.participants
            user_id = owner_user_id or participant_user_id
            second_user_id = participant_second_user_id
            kind = ConversationKind.duo.value
        else:
            user_id, second_user_id = owner_user_id or payload.user_id, None
            participant_user_id = participant_second_user_id = None
            kind = ConversationKind.assistant.value
        conversation = await self.repo.create(
            user_id=user_id,
            title=payload.title,
            tone_name=payload.tone.tone_name,
            custom_persona=payload.tone.custom_persona,
            second_user_id=second_user_id,
            participant_user_id=participant_user_id,
            participant_second_user_id=participant_second_user_id,
            kind=kind,
            use_documents=payload.use_documents,
        )
        await self.session.commit()
        return ConversationResponse.model_validate(conversation)

    async def list_all(self, owner_user_id: str | None = None) -> list[ConversationResponse]:
        conversations = (
            await self.repo.list_for_user(owner_user_id)
            if owner_user_id is not None
            else await self.repo.list_all()
        )
        return [ConversationResponse.model_validate(conversation) for conversation in conversations]

    async def get(self, conversation_id: str, owner_user_id: str | None = None) -> ConversationDetail:
        conversation = await self._conversation_or_404(
            conversation_id, with_messages=True, owner_user_id=owner_user_id
        )
        return ConversationDetail(
            **ConversationResponse.model_validate(conversation).model_dump(),
            messages=[self._message_response(message) for message in conversation.messages],
            summary=conversation.summary.content if conversation.summary else None,
        )

    async def update_tone(
        self, conversation_id: str, payload: ConversationToneUpdate, owner_user_id: str | None = None
    ) -> ConversationResponse:
        conversation = await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        updated = await self.repo.update_tone(
            conversation, tone_name=payload.tone_name, custom_persona=payload.custom_persona
        )
        await self.session.commit()
        return ConversationResponse.model_validate(updated)

    async def update_rag(
        self, conversation_id: str, payload: ConversationRagUpdate, owner_user_id: str | None = None
    ) -> ConversationResponse:
        conversation = await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        updated = await self.repo.update_use_documents(
            conversation, use_documents=payload.use_documents
        )
        await self.session.commit()
        return ConversationResponse.model_validate(updated)

    async def archive(self, conversation_id: str, owner_user_id: str | None = None) -> None:
        conversation = await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        await self.repo.archive(conversation)
        await self.session.commit()

    async def delete_message(
        self, conversation_id: str, message_id: str, owner_user_id: str | None = None
    ) -> None:
        await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        deleted = await self.repo.delete_message(message_id, conversation_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
        await self.session.commit()

    async def list_messages(
        self, conversation_id: str, limit: int, offset: int, owner_user_id: str | None = None
    ) -> PaginatedMessages:
        await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        messages, total = await self.repo.list_messages(conversation_id, limit=limit, offset=offset)
        return PaginatedMessages(
            items=[self._message_response(message) for message in messages],
            limit=limit,
            offset=offset,
            total=total,
        )

    async def send_message(
        self, conversation_id: str, payload: MessageCreate, owner_user_id: str | None = None
    ) -> ChatResponse:
        conversation = await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        if conversation.kind == ConversationKind.duo:
            return await self._send_duo_message(conversation, payload)
        user_id = owner_user_id or self._message_user_id(conversation, payload)
        tone = self.tones.resolve(payload.tone_override, conversation.tone_name, conversation.custom_persona)
        # Build context before persisting the new message so it appears exactly once,
        # and so a context-budget rejection leaves nothing behind in the DB.
        context = await self.memory.context_messages(conversation, tone.system_template)
        context.append(ChatMessage(role="user", content=payload.content))
        self._guard_context(context)
        rag_chunks = await self._retrieve_documents(conversation, payload.content, context)
        citations = self.rag.build_citations(rag_chunks) if rag_chunks else []
        user_message = await self.repo.add_message(
            conversation_id=conversation.id,
            user_id=user_id,
            role=MessageRole.user,
            content=payload.content,
            token_count=self.token_counter.count(payload.content),
        )
        result = await self._generate(context, tone.temperature, tone.top_p)
        assistant_message = await self.repo.add_message(
            conversation_id=conversation.id,
            user_id=user_id,
            role=MessageRole.assistant,
            content=result.content,
            token_count=result.output_tokens,
            model=result.model,
            citations=[citation.model_dump() for citation in citations] or None,
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
            citations=citations,
        )

    async def stream_message(
        self, conversation_id: str, payload: MessageCreate, owner_user_id: str | None = None
    ) -> AsyncIterator[str]:
        try:
            conversation = await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
            if conversation.kind == ConversationKind.duo:
                response = await self._send_duo_message(conversation, payload)
                yield self._sse("message", {"role": "user", "id": response.user_message.id})
                yield self._sse("done", {"user_message_id": response.user_message.id})
                yield "data: [DONE]\n\n"
                return
            user_id = owner_user_id or self._message_user_id(conversation, payload)
            tone = self.tones.resolve(
                payload.tone_override, conversation.tone_name, conversation.custom_persona
            )
            context = await self.memory.context_messages(conversation, tone.system_template)
            context.append(ChatMessage(role="user", content=payload.content))
            self._guard_context(context)
            rag_chunks = await self._retrieve_documents(conversation, payload.content, context)
            citations = self.rag.build_citations(rag_chunks) if rag_chunks else []
            citation_payload = [citation.model_dump() for citation in citations] or None
            user_message = await self.repo.add_message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=MessageRole.user,
                content=payload.content,
                token_count=self.token_counter.count(payload.content),
            )
            # Commit before streaming so the user message survives a client disconnect.
            await self.session.commit()
            params = GenerationParams(
                model=self.settings.default_model,
                temperature=tone.temperature,
                top_p=tone.top_p,
                timeout_seconds=self.settings.request_timeout_seconds,
            )
            model = self.llm.resolve_model(params)
            chunks: list[str] = []
            yield self._sse("message", {"role": "user", "id": user_message.id})
            if citations:
                yield self._sse(
                    "citations", {"citations": [citation.model_dump() for citation in citations]}
                )
            try:
                async for chunk in self.llm.stream_chat(context, params):
                    chunks.append(chunk)
                    yield self._sse("delta", {"delta": chunk})
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected mid-stream; keep whatever partial reply arrived.
                if chunks:
                    await self._persist_assistant(
                        conversation.id, user_id, "".join(chunks), model, citations=citation_payload
                    )
                raise
            content = "".join(chunks)
            assistant_message = await self._persist_assistant(
                conversation.id, user_id, content, model, commit=False, citations=citation_payload
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
        except Exception as exc:
            await self.session.rollback()
            logger.exception("stream_message_failed", extra={"conversation_id": conversation_id})
            yield self._sse("error", {"error": llm_error_detail(self.llm, exc)})
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=llm_error_detail(self.llm, exc),
            ) from exc

    async def suggest_reply(
        self, conversation_id: str, payload: SuggestRequest, owner_user_id: str | None = None
    ) -> SuggestResponse:
        conversation = await self._conversation_or_404(conversation_id, owner_user_id=owner_user_id)
        if conversation.kind != ConversationKind.duo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reply suggestions are only available for two-user conversations.",
            )
        participants = self._participants(conversation)
        if payload.for_user not in participants:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="for_user must be one of the conversation participants.",
            )
        tone = self.tones.resolve(
            payload.tone_override, conversation.tone_name, conversation.custom_persona
        )
        other_user = next(p for p in participants if p != payload.for_user)
        system_prompt = (
            f"{tone.system_template}\n\n"
            f"You are drafting the next chat message on behalf of '{payload.for_user}' in a "
            f"conversation between '{participants[0]}' and '{participants[1]}'. "
            f"Reply to '{other_user}' in the first person as '{payload.for_user}', staying "
            "consistent with what they have said so far. "
            "Write only the message text, with no name prefix or commentary."
        )
        context = await self.memory.duo_context_messages(
            conversation, system_prompt, speak_as=payload.for_user
        )
        self._guard_context(context)
        result = await self._generate(context, tone.temperature, tone.top_p)
        message_response = None
        if payload.persist:
            message = await self.repo.add_message(
                conversation_id=conversation.id,
                user_id=payload.for_user,
                role=MessageRole.user,
                content=result.content,
                token_count=result.output_tokens,
                model=result.model,
            )
            await self.memory.summarize_if_needed(conversation)
            await self.session.commit()
            message_response = self._message_response(message)
        logger.info(
            "suggestion_completed",
            extra={
                "conversation_id": conversation.id,
                "for_user": payload.for_user,
                "persisted": payload.persist,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        return SuggestResponse(
            conversation_id=conversation.id,
            for_user=payload.for_user,
            content=result.content,
            model=result.model,
            message=message_response,
            token_usage={"input": result.input_tokens, "output": result.output_tokens},
        )

    async def _send_duo_message(
        self, conversation: Conversation, payload: MessageCreate
    ) -> ChatResponse:
        user_id = self._message_user_id(conversation, payload)
        if user_id not in self._participants(conversation):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id must be one of the conversation participants.",
            )
        message = await self.repo.add_message(
            conversation_id=conversation.id,
            user_id=user_id,
            role=MessageRole.user,
            content=payload.content,
            token_count=self.token_counter.count(payload.content),
        )
        await self.memory.summarize_if_needed(conversation)
        await self.session.commit()
        return ChatResponse(
            conversation_id=conversation.id,
            user_message=self._message_response(message),
            assistant_message=None,
            token_usage={"input": message.token_count, "output": 0},
        )

    def _participants(self, conversation: Conversation) -> list[str]:
        first = conversation.participant_user_id or conversation.user_id
        participants = [first]
        if conversation.participant_second_user_id:
            participants.append(conversation.participant_second_user_id)
        elif conversation.second_user_id:
            participants.append(conversation.second_user_id)
        return participants

    async def _persist_assistant(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        model: str,
        *,
        commit: bool = True,
        citations: list | None = None,
    ) -> Message:
        message = await self.repo.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=MessageRole.assistant,
            content=content,
            token_count=self.token_counter.count(content),
            model=model,
            citations=citations,
        )
        if commit:
            await self.session.commit()
        return message

    async def _retrieve_documents(
        self, conversation: Conversation, query: str, context: list[ChatMessage]
    ) -> list[CitedChunk]:
        """Inject retrieved chunks into `context` when the conversation opts in.

        Runs after `_guard_context`, so history keeps the full budget and chunks only get the
        remainder (capped by rag_token_budget) — retrieved chunks are trimmed before history.
        """
        if not conversation.use_documents or self.rag is None:
            return []
        remaining = self.settings.context_token_budget - self.token_counter.count_messages(context)
        chunks = await self.rag.retrieve(
            conversation.id, query, min(self.settings.rag_token_budget, remaining)
        )
        if chunks:
            # After the tone prompt, before summary/history; never re-trimmed.
            context.insert(1, ChatMessage(role="system", content=self.rag.format_context(chunks)))
        return chunks

    def _guard_context(self, messages: list[ChatMessage]) -> None:
        tokens = self.token_counter.count_messages(messages)
        if tokens > self.settings.context_token_budget:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Message exceeds configured context budget.",
            )

    async def _conversation_or_404(
        self,
        conversation_id: str,
        *,
        with_messages: bool = False,
        owner_user_id: str | None = None,
    ) -> Conversation:
        if with_messages:
            conversation = await self.repo.get_with_messages(conversation_id)
        else:
            conversation = await self.repo.get(conversation_id)
        if not conversation or (owner_user_id is not None and conversation.user_id != owner_user_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return conversation

    def _message_response(self, message) -> MessageResponse:
        return MessageResponse(
            id=message.id,
            conversation_id=message.conversation_id,
            user_id=message.user_id,
            role=message.role.value if hasattr(message.role, "value") else message.role,
            content=message.content,
            token_count=message.token_count,
            model=message.model,
            created_at=message.created_at,
            citations=[Citation(**citation) for citation in message.citations]
            if message.citations
            else None,
        )

    def _message_user_id(self, conversation: Conversation, payload: MessageCreate) -> str:
        if payload.user_id == "anonymous":
            return conversation.user_id
        return payload.user_id

    def _sse(self, event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"
