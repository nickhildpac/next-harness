import logging

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Translation, TranslationTurn
from app.orchestration.chat_graph import ChatGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMClient
from app.repositories.translations import TranslationRepository
from app.schemas.translation import (
    TranslateResponse,
    TranslationCreate,
    TranslationSessionResponse,
    TranslationSessionSummary,
    TranslationTurnResponse,
)
from app.services.tokens import TokenCounter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert translator in an ongoing translation chat. Given source text, produce "
    "exactly two sections:\n\n"
    "TRANSLATION:\n"
    "<the full translation in {target_language}>\n\n"
    "ROMANIZED:\n"
    "<Latin-alphabet transliteration of the translation; if {target_language} already uses "
    "the Latin script, repeat the translation here>"
)


class TranslationService:
    def __init__(self, session: AsyncSession, settings: Settings, llm: LLMClient):
        self.session = session
        self.settings = settings
        self.llm = llm
        self.repo = TranslationRepository(session)
        self.token_counter = TokenCounter()
        self.graph = ChatGraph(llm)

    async def translate(self, payload: TranslationCreate, *, commit: bool = True) -> TranslateResponse:
        if payload.session_id:
            session = await self._or_404(payload.session_id, payload.user_id, with_turns=True)
            target_language = payload.target_language or session.target_language
            prior_turns = list(session.turns)
            turn_index = len(prior_turns)
        else:
            target_language = payload.target_language or ""
            title = payload.source_text.strip()[:50] or None
            session = await self.repo.create_session(
                user_id=payload.user_id,
                title=title,
                target_language=target_language,
            )
            prior_turns = []
            turn_index = 0

        turn = await self.repo.create_turn(
            session=session,
            turn_index=turn_index,
            source_text=payload.source_text,
            target_language=target_language,
        )
        messages = _build_messages(target_language, prior_turns, payload.source_text)
        self._guard_context(messages)
        params = GenerationParams(
            model=self.settings.default_model,
            temperature=0.3,
            top_p=0.95,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        try:
            result = await self.graph.run(messages, params)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM is unavailable or timed out.",
            ) from exc

        translated_text, romanized_text = _parse_response(result.content)
        await self.repo.update_turn_result(turn, translated_text, romanized_text, result.model)
        if turn_index == 0 and not session.title:
            session.title = payload.source_text.strip()[:50] or None
        session.target_language = target_language
        await self.repo.touch_session(session)
        if commit:
            await self.session.commit()
        logger.info(
            "translation_completed",
            extra={
                "session_id": session.id,
                "turn_id": turn.id,
                "target_language": target_language,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        return TranslateResponse(
            session_id=session.id,
            turn_id=turn.id,
            translated_text=translated_text,
            romanized_text=romanized_text,
            model=result.model,
            target_language=target_language,
            token_usage={"input": result.input_tokens, "output": result.output_tokens},
        )

    async def list_for_user(self, user_id: str) -> list[TranslationSessionSummary]:
        rows = await self.repo.list_for_user(user_id)
        summaries: list[TranslationSessionSummary] = []
        for row in rows:
            last_turn = row.turns[-1] if row.turns else None
            summaries.append(
                TranslationSessionSummary(
                    id=row.id,
                    user_id=row.user_id,
                    title=row.title,
                    target_language=row.target_language,
                    preview=last_turn.translated_text if last_turn else "",
                    turn_count=len(row.turns),
                    is_archived=row.is_archived,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )
        return summaries

    async def get(self, translation_id: str, user_id: str) -> TranslationSessionResponse:
        row = await self._or_404(translation_id, user_id, with_turns=True)
        return TranslationSessionResponse(
            id=row.id,
            user_id=row.user_id,
            title=row.title,
            target_language=row.target_language,
            is_archived=row.is_archived,
            created_at=row.created_at,
            updated_at=row.updated_at,
            turns=[TranslationTurnResponse.model_validate(turn) for turn in row.turns],
        )

    async def delete(self, translation_id: str, user_id: str) -> None:
        row = await self._or_404(translation_id, user_id)
        await self.repo.archive(row)
        await self.session.commit()

    def _guard_context(self, messages: list[ChatMessage]) -> None:
        tokens = self.token_counter.count_messages(messages)
        if tokens > self.settings.context_token_budget:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Translation context exceeds configured token budget.",
            )

    async def _or_404(
        self, translation_id: str, user_id: str, *, with_turns: bool = False
    ) -> Translation:
        if with_turns:
            row = await self.repo.get_with_turns(translation_id)
        else:
            row = await self.repo.get(translation_id)
        if row is None or row.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Translation not found"
            )
        return row


def _format_turn_output(translated_text: str, romanized_text: str) -> str:
    return f"TRANSLATION:\n{translated_text}\n\nROMANIZED:\n{romanized_text}"


def _build_messages(
    target_language: str,
    prior_turns: list[TranslationTurn],
    source_text: str,
) -> list[ChatMessage]:
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT.format(target_language=target_language))
    ]
    for turn in prior_turns:
        messages.append(ChatMessage(role="user", content=turn.source_text))
        messages.append(
            ChatMessage(
                role="assistant",
                content=_format_turn_output(turn.translated_text, turn.romanized_text),
            )
        )
    messages.append(ChatMessage(role="user", content=source_text))
    return messages


def _parse_response(text: str) -> tuple[str, str]:
    """Split LLM output on TRANSLATION:/ROMANIZED: section markers."""
    translation = ""
    romanized = ""
    if "ROMANIZED:" in text:
        parts = text.split("ROMANIZED:", 1)
        romanized = parts[1].strip()
        trans_part = parts[0]
        if "TRANSLATION:" in trans_part:
            translation = trans_part.split("TRANSLATION:", 1)[1].strip()
        else:
            translation = trans_part.strip()
    elif "TRANSLATION:" in text:
        translation = text.split("TRANSLATION:", 1)[1].strip()
    else:
        translation = text.strip()
    return translation, romanized
