from collections.abc import AsyncIterator

from app.core.config import Settings
from app.ports.llm import ChatMessage, GenerationParams, LLMResult
from app.schemas.task import TaskCreate
from app.services.tasks import TaskService


class ScriptedLLM:
    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[ChatMessage]] = []

    def resolve_model(self, params: GenerationParams) -> str:
        return "scripted"

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        self.calls.append(messages)
        content = self.script.pop(0) if self.script else "(exhausted)"
        return LLMResult(content=content, model="scripted", input_tokens=1, output_tokens=2)

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:  # pragma: no cover
        if False:
            yield ""

    async def health(self) -> bool:
        return True


async def test_tools_endpoint_lists_default_registry(session):
    service = TaskService(session, Settings(), ScriptedLLM([]), http_client=None)
    tools = service.available_tools()
    names = {t.name for t in tools}
    assert {"now", "list_notes", "create_note", "finish"}.issubset(names)


async def test_run_task_persists_run_and_steps(session):
    llm = ScriptedLLM(
        [
            'Plan: get time.\n<tool_call>{"name":"now","arguments":{}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"time reported"}}</tool_call>',
        ]
    )
    service = TaskService(session, Settings(), llm, http_client=None)
    detail = await service.create_task(
        TaskCreate(goal="Report the time", user_id="alice", max_steps=4)
    )
    assert detail.status == "completed"
    assert detail.result_summary == "time reported"
    assert detail.steps_taken == 2
    assert any(s.tool_name == "now" for s in detail.steps)
    # Reload to confirm persistence
    fetched = await service.get_task(detail.id)
    assert fetched.id == detail.id
    assert fetched.status == "completed"


async def test_run_task_records_failure_on_step_limit(session):
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"now","arguments":{}}</tool_call>',
            '<tool_call>{"name":"now","arguments":{}}</tool_call>',
            '<tool_call>{"name":"now","arguments":{}}</tool_call>',
        ]
    )
    service = TaskService(session, Settings(), llm, http_client=None)
    detail = await service.create_task(
        TaskCreate(goal="Loop forever", user_id="alice", max_steps=2)
    )
    assert detail.status == "failed"
    assert detail.error and "step limit" in detail.error


async def test_scoping_allowed_tools_hides_others(session):
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"list_notes","arguments":{}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"listed"}}</tool_call>',
        ]
    )
    service = TaskService(session, Settings(), llm, http_client=None)
    detail = await service.create_task(
        TaskCreate(
            goal="List notes", user_id="alice", max_steps=4, allowed_tools=["list_notes"]
        )
    )
    assert detail.status == "completed"
    system_prompt = llm.calls[0][0].content
    assert "list_notes" in system_prompt
    assert "http_fetch" not in system_prompt


async def test_list_tasks_scoped_by_user(session):
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"finish","arguments":{"summary":"ok"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"ok"}}</tool_call>',
        ]
    )
    service = TaskService(session, Settings(), llm, http_client=None)
    await service.create_task(TaskCreate(goal="A", user_id="alice"))
    await service.create_task(TaskCreate(goal="B", user_id="bob"))
    alice = await service.list_tasks("alice")
    bob = await service.list_tasks("bob")
    assert [t.goal for t in alice] == ["A"]
    assert [t.goal for t in bob] == ["B"]


def test_task_schema_accepts_prompt_alias():
    payload = TaskCreate.model_validate({"prompt": "do the thing"})
    assert payload.goal == "do the thing"


def test_openapi_exposes_task_routes():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/tasks" in paths
    assert "/tasks/{task_id}" in paths
    assert "/tools" in paths
