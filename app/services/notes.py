import logging

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Note
from app.orchestration.chat_graph import ChatGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult
from app.repositories.notes import NoteRepository
from app.schemas.note import (
    NoteCreate,
    NoteRegenerateRequest,
    NoteRegenerateResponse,
    NoteResponse,
    NoteUpdate,
)
from app.services.note_styles import NoteStyleService
from app.services.tokens import TokenCounter

logger = logging.getLogger(__name__)

_SYSTEM_RULES = (
    "\n\nOutput rules:\n"
    "- Return ONLY the full replacement markdown body.\n"
    "- Do not wrap the output in triple-backtick code fences.\n"
    "- Preserve the author's intent while incorporating the instruction fully."
)


class NoteService:
    def __init__(self, session: AsyncSession, settings: Settings, llm: LLMClient):
        self.session = session
        self.settings = settings
        self.llm = llm
        self.repo = NoteRepository(session)
        self.token_counter = TokenCounter()
        self.styles = NoteStyleService(settings)
        self.graph = ChatGraph(llm)

    async def create(self, payload: NoteCreate) -> NoteResponse:
        note = await self.repo.create(
            user_id=payload.user_id,
            title=payload.title,
            content=payload.content,
            style_name=payload.style.style_name,
            custom_instructions=payload.style.custom_instructions,
        )
        await self.session.commit()
        return NoteResponse.model_validate(note)

    async def list_for_user(self, user_id: str) -> list[NoteResponse]:
        notes = await self.repo.list_for_user(user_id)
        return [NoteResponse.model_validate(note) for note in notes]

    async def get(self, note_id: str, user_id: str) -> NoteResponse:
        note = await self._note_or_404(note_id, user_id)
        return NoteResponse.model_validate(note)

    async def update(self, note_id: str, user_id: str, payload: NoteUpdate) -> NoteResponse:
        note = await self._note_or_404(note_id, user_id)
        updates = payload.model_dump(exclude_unset=True)
        style = updates.pop("style", None)
        if style is not None:
            updates["style_name"] = style["style_name"]
            updates["custom_instructions"] = style.get("custom_instructions")
        if updates:
            await self.repo.apply_updates(note, updates)
        await self.session.commit()
        return NoteResponse.model_validate(note)

    async def delete(self, note_id: str, user_id: str) -> None:
        note = await self._note_or_404(note_id, user_id)
        await self.repo.archive(note)
        await self.session.commit()

    async def regenerate(
        self, note_id: str, payload: NoteRegenerateRequest
    ) -> NoteRegenerateResponse:
        note = await self._note_or_404(note_id, payload.user_id)
        style = self.styles.resolve(
            payload.style_override, note.style_name, note.custom_instructions
        )
        system_content = f"{style.system_template}{_SYSTEM_RULES}"
        current = note.content.strip() or "(empty)"
        user_content = (
            "Current note (markdown):\n---\n"
            f"{current}\n"
            "---\n"
            f"Instruction: {payload.prompt}\n"
            "Produce the new complete markdown body."
        )
        messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]
        self._guard_context(messages)
        result = await self._generate(messages, style.temperature, style.top_p)
        await self.repo.replace_content(note, result.content)
        await self.session.commit()
        logger.info(
            "note_regenerated",
            extra={
                "note_id": note.id,
                "style_name": note.style_name,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        return NoteRegenerateResponse(
            note_id=note.id,
            content=result.content,
            model=result.model,
            style_name=note.style_name,
            token_usage={"input": result.input_tokens, "output": result.output_tokens},
        )

    async def _generate(
        self, messages: list[ChatMessage], temperature: float, top_p: float
    ) -> LLMResult:
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
                detail="Note exceeds configured context budget.",
            )

    async def _note_or_404(self, note_id: str, user_id: str) -> Note:
        note = await self.repo.get(note_id)
        if note is None or note.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
        return note
