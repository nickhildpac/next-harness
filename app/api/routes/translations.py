from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user, get_translation_service
from app.db.models import User
from app.schemas.translation import (
    TranslateResponse,
    TranslationCreate,
    TranslationSessionResponse,
    TranslationSessionSummary,
)
from app.services.translations import TranslationService

router = APIRouter(tags=["translations"])

LANGUAGES = [
    {"id": "arabic", "label": "Arabic"},
    {"id": "chinese", "label": "Chinese (Simplified)"},
    {"id": "french", "label": "French"},
    {"id": "german", "label": "German"},
    {"id": "hindi", "label": "Hindi"},
    {"id": "italian", "label": "Italian"},
    {"id": "japanese", "label": "Japanese"},
    {"id": "korean", "label": "Korean"},
    {"id": "portuguese", "label": "Portuguese"},
    {"id": "russian", "label": "Russian"},
    {"id": "spanish", "label": "Spanish"},
    {"id": "turkish", "label": "Turkish"},
    {"id": "urdu", "label": "Urdu"},
]


@router.get("/languages")
async def list_languages() -> list[dict[str, str]]:
    return LANGUAGES


@router.post("/translations", response_model=TranslateResponse, status_code=201)
async def create_translation(
    payload: TranslationCreate,
    current_user: User = Depends(get_current_user),
    service: TranslationService = Depends(get_translation_service),
) -> TranslateResponse:
    payload = payload.model_copy(update={"user_id": current_user.id})
    return await service.translate(payload)


@router.get("/translations", response_model=list[TranslationSessionSummary])
async def list_translations(
    current_user: User = Depends(get_current_user),
    service: TranslationService = Depends(get_translation_service),
) -> list[TranslationSessionSummary]:
    return await service.list_for_user(current_user.id)


@router.get("/translations/{translation_id}", response_model=TranslationSessionResponse)
async def get_translation(
    translation_id: str,
    current_user: User = Depends(get_current_user),
    service: TranslationService = Depends(get_translation_service),
) -> TranslationSessionResponse:
    return await service.get(translation_id, current_user.id)


@router.delete("/translations/{translation_id}", status_code=204)
async def delete_translation(
    translation_id: str,
    current_user: User = Depends(get_current_user),
    service: TranslationService = Depends(get_translation_service),
) -> None:
    await service.delete(translation_id, current_user.id)
