from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_current_user, get_task_service
from app.db.models import User
from app.schemas.document import DocumentResponse
from app.schemas.task import TaskCreate, TaskDetail, TaskResponse, ToolInfo
from app.services.tasks import TaskDocumentUpload, TaskService

router = APIRouter(tags=["tasks"])


@router.post("/tasks", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: TaskCreate,
    stream: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    payload = payload.model_copy(update={"user_id": current_user.id})
    if stream and payload.run:
        return StreamingResponse(
            service.stream_create_and_run(payload),
            media_type="text/event-stream",
        )
    return await service.create_task(payload)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> list[TaskResponse]:
    return await service.list_tasks(current_user.id)


@router.post(
    "/tasks/{task_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_task_document(
    task_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> DocumentResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename.",
        )
    document = await service.upload_task_document(
        task_id,
        current_user.id,
        TaskDocumentUpload(
            filename=file.filename,
            content_type=file.content_type,
            data=await file.read(),
        ),
    )
    return DocumentResponse.model_validate(document)


@router.post("/tasks/{task_id}/run", response_model=TaskDetail)
async def run_task(
    task_id: str,
    stream: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    if stream:
        return StreamingResponse(
            service.stream_run_task(task_id, current_user.id),
            media_type="text/event-stream",
        )
    return await service.run_task(task_id, current_user.id)


@router.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskDetail:
    return await service.get_task(task_id, current_user.id)


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(
    service: TaskService = Depends(get_task_service),
) -> list[ToolInfo]:
    return service.available_tools()
