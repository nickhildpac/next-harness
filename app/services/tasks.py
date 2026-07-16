from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.mcp_client import McpStdioSession
from app.core.config import Settings
from app.db.models import AgentTask, TaskStatus, TaskStepKind
from app.orchestration.agent_graph import AgentGraph, AgentRun, StepRecord
from app.ports.embeddings import EmbeddingsClient
from app.ports.llm import GenerationParams, LLMClient
from app.ports.vectorstore import VectorStore
from app.repositories.documents import DocumentRepository
from app.repositories.tasks import TaskRepository
from app.schemas.task import TaskCreate, TaskDetail, TaskResponse, TaskStepResponse, ToolInfo
from app.services.rag import RagService
from app.tools.mcp_invoker import HybridToolInvoker
from app.tools.registry import Tool, ToolContext, ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskDocumentUpload:
    filename: str
    content_type: str | None
    data: bytes


class TaskService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        llm: LLMClient,
        http_client: httpx.AsyncClient | None,
        embeddings: EmbeddingsClient | None = None,
        vectorstore: VectorStore | None = None,
        *,
        use_mcp_tools: bool = True,
    ):
        self.session = session
        self.settings = settings
        self.llm = llm
        self.http_client = http_client
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.use_mcp_tools = use_mcp_tools
        self.repo = TaskRepository(session)
        self._registry = build_default_registry()

    def available_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(name=spec.name, description=spec.description, parameters=spec.parameters)
            for spec in self._registry.specs()
        ]

    async def create_task(self, payload: TaskCreate) -> TaskDetail:
        task = await self.repo.create(
            user_id=payload.user_id,
            goal=payload.goal,
            max_steps=payload.max_steps,
            allowed_tools=payload.allowed_tools,
        )
        if not payload.run:
            await self.session.commit()
            return await self._detail(task.id)
        return await self._run_task(task)

    async def run_task(self, task_id: str, user_id: str) -> TaskDetail:
        task = await self._task_for_user(task_id, user_id)
        if task.status != TaskStatus.pending.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task cannot be run from status '{task.status}'.",
            )
        return await self._run_task(task)

    async def upload_task_document(self, task_id: str, user_id: str, document: TaskDocumentUpload):
        task = await self._task_for_user(task_id, user_id)
        if task.status != TaskStatus.pending.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Documents can only be attached while task is pending, not '{task.status}'.",
            )
        uploaded = await self._ingest_documents(task.id, [document])
        if uploaded != 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document upload failed.",
            )
        documents = await DocumentRepository(self.session).list_by_task(task.id)
        return documents[-1]

    async def _run_task(self, task: AgentTask) -> TaskDetail:
        self._validate_allowed_tools(task.allowed_tools)
        uploaded_document_count = await self._task_document_count(task.id)
        await self.repo.set_status(task, TaskStatus.running)
        await self.session.commit()

        context = self._tool_context(task, uploaded_document_count=uploaded_document_count)
        if self.use_mcp_tools:
            run = await self._run_with_mcp(task, context)
        else:
            run = await self._run_with_local_registry(task, context)

        await self._persist_run(task, run)
        await self.session.commit()
        logger.info(
            "task_run_completed",
            extra={
                "task_id": task.id,
                "status": task.status,
                "turns_used": run.turns_used,
                "step_limit_hit": run.step_limit_hit,
            },
        )
        return await self._detail(task.id)

    async def _run_with_mcp(self, task: AgentTask, context: ToolContext) -> AgentRun:
        async with McpStdioSession.from_settings(
            self.settings,
            user_id=task.user_id,
            task_id=task.id,
        ) as mcp:
            invoker = await HybridToolInvoker.create(mcp, allowed_tools=task.allowed_tools)
            graph = AgentGraph(self.llm, invoker, max_steps=task.max_steps)
            return await graph.run(task.goal, self._params(), context)

    async def _run_with_local_registry(self, task: AgentTask, context: ToolContext) -> AgentRun:
        registry = self._scoped_registry(task.allowed_tools)
        graph = AgentGraph(self.llm, registry, max_steps=task.max_steps)
        return await graph.run(task.goal, self._params(), context)

    def _tool_context(self, task: AgentTask, *, uploaded_document_count: int) -> ToolContext:
        # MCP tools open their own DB session in the child process; local registry
        # path still needs the request-scoped session and DI bag for builtins.
        if self.use_mcp_tools:
            return ToolContext(
                user_id=task.user_id,
                task_id=task.id,
                metadata={"uploaded_document_count": uploaded_document_count},
            )
        return ToolContext(
            session=self.session,
            http_client=self.http_client,
            user_id=task.user_id,
            task_id=task.id,
            settings=self.settings,
            llm=self.llm,
            embeddings=self.embeddings,
            vectorstore=self.vectorstore,
            metadata={"uploaded_document_count": uploaded_document_count},
        )

    async def list_tasks(self, user_id: str) -> list[TaskResponse]:
        tasks = await self.repo.list_for_user(user_id)
        return [TaskResponse.model_validate(t) for t in tasks]

    async def get_task(self, task_id: str, user_id: str | None = None) -> TaskDetail:
        return await self._detail(task_id, user_id)

    def _params(self) -> GenerationParams:
        return GenerationParams(
            model=self.settings.default_model,
            temperature=0.2,
            top_p=0.9,
            timeout_seconds=self.settings.request_timeout_seconds,
        )

    def _validate_allowed_tools(self, allowed: list[str] | None) -> None:
        if not allowed:
            return
        allowed_set = set(allowed)
        unknown = sorted(name for name in allowed_set if self._registry.get(name) is None)
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"allowed_tools contains unknown tool(s): {', '.join(unknown)}",
            )

    def _scoped_registry(self, allowed: list[str] | None) -> ToolRegistry:
        self._validate_allowed_tools(allowed)
        if not allowed:
            return self._registry
        allowed_set = set(allowed)
        allowed_set.add("finish")  # finish is always reachable
        tools: list[Tool] = []
        for name in self._registry.names():
            if name in allowed_set:
                tool = self._registry.get(name)
                if tool is not None:
                    tools.append(tool)
        if not tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="allowed_tools did not match any registered tool.",
            )
        return ToolRegistry(tools)

    async def _persist_run(self, task: AgentTask, run: AgentRun) -> None:
        for index, step in enumerate(run.steps):
            await self.repo.add_step(
                task,
                step_index=index,
                kind=self._step_kind(step),
                tool_name=step.tool_name,
                content=step.content,
                payload=step.payload,
                ok=step.ok,
            )
        await self.repo.bump_steps(task, run.turns_used)
        if run.errored:
            await self.repo.set_status(task, TaskStatus.failed, error=run.error, model=run.model)
        elif run.completed:
            await self.repo.set_status(
                task,
                TaskStatus.completed,
                result_summary=run.final_summary,
                model=run.model,
            )
        elif run.step_limit_hit:
            await self.repo.set_status(
                task,
                TaskStatus.failed,
                error=f"step limit ({task.max_steps}) reached before task completion",
                model=run.model,
            )
        else:
            await self.repo.set_status(
                task, TaskStatus.failed, error="run ended without a result", model=run.model
            )

    async def _ingest_documents(self, task_id: str, documents: list[TaskDocumentUpload]) -> int:
        if self.embeddings is None or self.vectorstore is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document ingestion is unavailable.",
            )
        rag = RagService(self.session, self.settings, self.embeddings, self.vectorstore)
        count = 0
        for document in documents:
            await rag.ingest_task_document(
                task_id,
                filename=document.filename,
                content_type=document.content_type,
                data=document.data,
                commit=True,
            )
            count += 1
        return count

    async def _task_for_user(self, task_id: str, user_id: str) -> AgentTask:
        task = await self.repo.get(task_id)
        if task is None or task.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return task

    async def _task_document_count(self, task_id: str) -> int:
        return len(await DocumentRepository(self.session).list_by_task(task_id))

    def _step_kind(self, step: StepRecord) -> TaskStepKind:
        return TaskStepKind(step.kind)

    async def _detail(self, task_id: str, user_id: str | None = None) -> TaskDetail:
        task = await self.repo.get_with_steps(task_id)
        if task is None or (user_id is not None and task.user_id != user_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return TaskDetail(
            **TaskResponse.model_validate(task).model_dump(),
            steps=[TaskStepResponse.model_validate(s) for s in task.steps],
        )
