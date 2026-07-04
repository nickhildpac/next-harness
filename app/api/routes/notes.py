from fastapi import APIRouter, Depends, Query, Response, status

from app.api.dependencies import get_note_service, get_settings
from app.core.config import Settings
from app.schemas.note import (
    NoteCreate,
    NoteRegenerateRequest,
    NoteRegenerateResponse,
    NoteResponse,
    NoteUpdate,
)
from app.services.notes import NoteService

router = APIRouter(tags=["notes"])


@router.post("/notes", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(
    payload: NoteCreate,
    service: NoteService = Depends(get_note_service),
) -> NoteResponse:
    return await service.create(payload)


@router.get("/notes", response_model=list[NoteResponse])
async def list_notes(
    user_id: str = Query(default="anonymous", min_length=1, max_length=128),
    service: NoteService = Depends(get_note_service),
) -> list[NoteResponse]:
    return await service.list_for_user(user_id)


@router.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    user_id: str = Query(default="anonymous", min_length=1, max_length=128),
    service: NoteService = Depends(get_note_service),
) -> NoteResponse:
    return await service.get(note_id, user_id)


@router.patch("/notes/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    payload: NoteUpdate,
    user_id: str = Query(default="anonymous", min_length=1, max_length=128),
    service: NoteService = Depends(get_note_service),
) -> NoteResponse:
    return await service.update(note_id, user_id, payload)


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    note_id: str,
    user_id: str = Query(default="anonymous", min_length=1, max_length=128),
    service: NoteService = Depends(get_note_service),
) -> Response:
    await service.delete(note_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/notes/{note_id}/regenerate", response_model=NoteRegenerateResponse)
async def regenerate_note(
    note_id: str,
    payload: NoteRegenerateRequest,
    service: NoteService = Depends(get_note_service),
) -> NoteRegenerateResponse:
    return await service.regenerate(note_id, payload)


@router.get("/note-styles")
async def list_note_styles(
    settings: Settings = Depends(get_settings),
) -> list[dict[str, str]]:
    labels = {
        "default": "Default",
        "academic": "Academic",
        "meeting": "Meeting",
        "blog": "Blog",
    }
    return [
        {"id": name, "label": labels.get(name, name.title())}
        for name in settings.note_styles.keys()
    ]
