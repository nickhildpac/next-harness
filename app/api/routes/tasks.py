from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_task_service
from app.schemas.task import TaskCreate, TaskDetail, TaskResponse, ToolInfo
from app.services.tasks import TaskService

router = APIRouter(tags=["tasks"])


@router.post("/tasks", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: TaskCreate,
    service: TaskService = Depends(get_task_service),
) -> TaskDetail:
    return await service.create_task(payload)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    user_id: str = Query(default="anonymous", min_length=1, max_length=128),
    service: TaskService = Depends(get_task_service),
) -> list[TaskResponse]:
    return await service.list_tasks(user_id)


@router.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
) -> TaskDetail:
    return await service.get_task(task_id)


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(
    service: TaskService = Depends(get_task_service),
) -> list[ToolInfo]:
    return service.available_tools()
