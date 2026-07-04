from fastapi import APIRouter

router = APIRouter(tags=["tones"])


@router.get("/tones")
async def list_tones() -> list[dict[str, str]]:
    return [
        {"id": "friendly", "label": "Friendly", "color_key": "Friendly"},
        {"id": "professional", "label": "Formal", "color_key": "Formal"},
        {"id": "humorous", "label": "Playful", "color_key": "Playful"},
        {"id": "empathetic", "label": "Empathetic", "color_key": "Empathetic"},
        {"id": "concise", "label": "Direct", "color_key": "Direct"},
    ]
