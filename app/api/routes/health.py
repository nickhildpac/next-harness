from fastapi import APIRouter, Depends

from app.api.dependencies import get_llm_client
from app.core.config import Settings, get_settings
from app.ports.llm import LLMClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}


@router.get("/health/llm")
async def llm_health(llm: LLMClient = Depends(get_llm_client)) -> dict[str, str | bool]:
    available = await llm.health()
    return {"status": "ok" if available else "unavailable", "available": available}

