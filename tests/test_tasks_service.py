from collections.abc import AsyncIterator

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Document, Note, Translation, TranslationTurn
from app.ports.llm import ChatMessage, GenerationParams, LLMResult
from app.schemas.task import TaskCreate
from app.services.tasks import TaskService
from tests.conftest import FakeEmbeddings, FakeVectorStore


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
    assert {
        "now",
        "list_notes",
        "create_note",
        "update_note",
        "translate_text",
        "ingest_task_document",
        "search_task_documents",
        "finish",
    }.issubset(names)


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


async def test_agent_can_create_note_for_task_user(session):
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"create_note","arguments":{"title":"Ideas","content":"Ship the notes tool.","style_name":"default"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"note created"}}</tool_call>',
        ]
    )
    service = TaskService(session, Settings(), llm, http_client=None)

    detail = await service.create_task(
        TaskCreate(goal="Create a note", user_id="alice", max_steps=4)
    )

    assert detail.status == "completed"
    note = await session.scalar(select(Note).where(Note.user_id == "alice"))
    assert note is not None
    assert note.title == "Ideas"
    assert note.content == "Ship the notes tool."


async def test_agent_can_translate_and_save_text(session):
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"translate_text","arguments":{"source_text":"hello","target_language":"Spanish"}}</tool_call>',
            "TRANSLATION:\nhola\n\nROMANIZED:\nhola",
            '<tool_call>{"name":"finish","arguments":{"summary":"translation saved"}}</tool_call>',
        ]
    )
    service = TaskService(session, Settings(), llm, http_client=None)

    detail = await service.create_task(
        TaskCreate(goal="Translate hello", user_id="alice", max_steps=4)
    )

    assert detail.status == "completed"
    row = await session.scalar(select(Translation).where(Translation.user_id == "alice"))
    assert row is not None
    assert row.target_language == "Spanish"
    turn = await session.scalar(select(TranslationTurn).where(TranslationTurn.translation_id == row.id))
    assert turn is not None
    assert turn.translated_text == "hola"


async def test_agent_can_ingest_and_search_task_documents(session):
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"ingest_task_document","arguments":{"filename":"facts.md","content":"Apples and bananas are fruit salad ingredients."}}</tool_call>',
            '<tool_call>{"name":"search_task_documents","arguments":{"query":"fruit salad apples"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"document searched"}}</tool_call>',
        ]
    )
    vectorstore = FakeVectorStore()
    service = TaskService(
        session,
        Settings(),
        llm,
        http_client=None,
        embeddings=FakeEmbeddings(),
        vectorstore=vectorstore,
    )

    detail = await service.create_task(
        TaskCreate(goal="Ingest and search facts", user_id="alice", max_steps=6)
    )

    assert detail.status == "completed"
    document = await session.scalar(select(Document).where(Document.task_id == detail.id))
    assert document is not None
    assert document.filename == "facts.md"
    search_step = next(s for s in detail.steps if s.tool_name == "search_task_documents")
    assert search_step.ok is True
    assert search_step.payload["output"]["count"] == 1
    assert "Apples and bananas" in search_step.payload["output"]["context"]


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
