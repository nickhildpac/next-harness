from fastapi import APIRouter, Depends
from pydantic import SecretStr

from app.core.config import Settings, get_settings

router = APIRouter(tags=["providers"])


def _has_key(value: str | SecretStr | None) -> bool:
    if isinstance(value, SecretStr):
        return bool(value.get_secret_value())
    return bool(value)


@router.get("/providers")
async def list_providers(
    settings: Settings = Depends(get_settings),
) -> list[dict[str, object]]:
    return [
        {
            "id": "openrouter",
            "label": "OpenRouter",
            "available": _has_key(settings.openrouter_api_key),
            "model": settings.openrouter_model,
        },
        {
            "id": "openai",
            "label": "OpenAI",
            "available": _has_key(settings.openai_api_key),
            "model": settings.openai_model,
        },
        {
            "id": "anthropic",
            "label": "Anthropic",
            "available": _has_key(settings.anthropic_api_key),
            "model": settings.anthropic_model,
        },
        {
            "id": "gemini",
            "label": "Gemini",
            "available": _has_key(settings.gemini_api_key),
            "model": settings.gemini_model,
        },
        {
            "id": "ollama",
            "label": "Ollama (local)",
            "available": True,
            "model": settings.default_model,
        },
    ]
