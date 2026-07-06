import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_current_user, get_task_service
from app.db.models import User
from app.schemas.task import TaskCreate, TaskDetail, TaskResponse, ToolInfo
from app.services.tasks import TaskDocumentUpload, TaskService

router = APIRouter(tags=["tasks"])


@router.post("/tasks", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: TaskCreate,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskDetail:
    payload = payload.model_copy(update={"user_id": current_user.id})
    return await service.create_task(payload)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> list[TaskResponse]:
    return await service.list_tasks(current_user.id)


@router.post("/tasks/with-documents", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task_with_documents(
    goal: str = Form(...),
    max_steps: int = Form(8),
    allowed_tools: str | None = Form(None),
    run: bool = Form(True),
    files: list[UploadFile] | None = File(None),
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
) -> TaskDetail:
    uploads: list[TaskDocumentUpload] = []
    for file in files or []:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file has no filename.",
            )
        uploads.append(
            TaskDocumentUpload(
                filename=file.filename,
                content_type=file.content_type,
                data=await file.read(),
            )
        )
    payload = TaskCreate(
        goal=goal,
        user_id=current_user.id,
        max_steps=max_steps,
        allowed_tools=_parse_allowed_tools(allowed_tools),
        run=run,
    )
    return await service.create_task(payload, documents=uploads)


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


def _parse_allowed_tools(value: str | None) -> list[str] | None:
    if not value:
        return None
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        parsed = value.split(",")
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="allowed_tools must be a JSON array or comma-separated list.",
        )
    return [tool.strip() for tool in parsed if isinstance(tool, str) and tool.strip()] or None
