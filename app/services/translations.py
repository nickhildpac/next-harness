import logging

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Translation
from app.orchestration.chat_graph import ChatGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMClient
from app.repositories.translations import TranslationRepository
from app.schemas.translation import TranslateResponse, TranslationCreate, TranslationResponse
from app.services.tokens import TokenCounter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert translator. Given source text, produce exactly two sections:\n\n"
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

    async def translate(self, payload: TranslationCreate) -> TranslateResponse:
        title = payload.source_text.strip()[:50] or None
        translation = await self.repo.create(
            user_id=payload.user_id,
            title=title,
            source_text=payload.source_text,
            target_language=payload.target_language,
        )
        system_content = _SYSTEM_PROMPT.format(target_language=payload.target_language)
        user_content = (
            f"Translate the following text to {payload.target_language}:\n\n{payload.source_text}"
        )
        messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]
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
        await self.repo.update_result(translation, translated_text, romanized_text, result.model)
        await self.session.commit()
        logger.info(
            "translation_completed",
            extra={
                "translation_id": translation.id,
                "target_language": payload.target_language,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        return TranslateResponse(
            translation_id=translation.id,
            translated_text=translated_text,
            romanized_text=romanized_text,
            model=result.model,
            target_language=payload.target_language,
            token_usage={"input": result.input_tokens, "output": result.output_tokens},
        )

    async def list_for_user(self, user_id: str) -> list[TranslationResponse]:
        rows = await self.repo.list_for_user(user_id)
        return [TranslationResponse.model_validate(r) for r in rows]

    async def get(self, translation_id: str, user_id: str) -> TranslationResponse:
        row = await self._or_404(translation_id, user_id)
        return TranslationResponse.model_validate(row)

    async def delete(self, translation_id: str, user_id: str) -> None:
        row = await self._or_404(translation_id, user_id)
        await self.repo.archive(row)
        await self.session.commit()

    def _guard_context(self, messages: list[ChatMessage]) -> None:
        tokens = self.token_counter.count_messages(messages)
        if tokens > self.settings.context_token_budget:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Source text exceeds configured context budget.",
            )

    async def _or_404(self, translation_id: str, user_id: str) -> Translation:
        row = await self.repo.get(translation_id)
        if row is None or row.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Translation not found"
            )
        return row


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
