from collections.abc import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.ollama import OllamaClient
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.ports.llm import LLMClient
from app.services.conversations import ConversationService
from app.services.tokens import TokenCounter


def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return OllamaClient(settings, TokenCounter())


async def get_conversation_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> AsyncIterator[ConversationService]:
    yield ConversationService(session, settings, llm)

