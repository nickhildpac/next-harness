from fastapi import APIRouter, Depends

from app.api.dependencies import get_translation_service
from app.schemas.translation import TranslateResponse, TranslationCreate, TranslationResponse
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
    service: TranslationService = Depends(get_translation_service),
) -> TranslateResponse:
    return await service.translate(payload)


@router.get("/translations", response_model=list[TranslationResponse])
async def list_translations(
    user_id: str = "anonymous",
    service: TranslationService = Depends(get_translation_service),
) -> list[TranslationResponse]:
    return await service.list_for_user(user_id)


@router.get("/translations/{translation_id}", response_model=TranslationResponse)
async def get_translation(
    translation_id: str,
    user_id: str = "anonymous",
    service: TranslationService = Depends(get_translation_service),
) -> TranslationResponse:
    return await service.get(translation_id, user_id)


@router.delete("/translations/{translation_id}", status_code=204)
async def delete_translation(
    translation_id: str,
    user_id: str = "anonymous",
    service: TranslationService = Depends(get_translation_service),
) -> None:
    await service.delete(translation_id, user_id)
