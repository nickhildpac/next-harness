from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_conversation_service
from app.schemas.conversation import (
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationResponse,
    ConversationToneUpdate,
    MessageCreate,
    PaginatedMessages,
)
from app.services.conversations import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    return await service.create(payload)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationResponse]:
    return await service.list_all()


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    return await service.get(conversation_id)


@router.patch("/{conversation_id}/tone", response_model=ConversationResponse)
async def update_tone(
    conversation_id: str,
    payload: ConversationToneUpdate,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    return await service.update_tone(conversation_id, payload)


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: str,
    payload: MessageCreate,
    stream: bool = Query(default=False),
    service: ConversationService = Depends(get_conversation_service),
):
    if stream:
        return StreamingResponse(
            service.stream_message(conversation_id, payload),
            media_type="text/event-stream",
        )
    return await service.send_message(conversation_id, payload)


@router.get("/{conversation_id}/messages", response_model=PaginatedMessages)
async def list_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: ConversationService = Depends(get_conversation_service),
) -> PaginatedMessages:
    return await service.list_messages(conversation_id, limit, offset)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    await service.archive(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
