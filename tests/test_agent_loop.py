from collections.abc import AsyncIterator

from app.orchestration.agent_graph import AgentGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMResult, ToolCall
from app.tools.registry import Tool, ToolContext, ToolRegistry


class ScriptedLLM:
    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[ChatMessage]] = []
        self.model = "scripted"

    def resolve_model(self, params: GenerationParams) -> str:
        return self.model

    async def chat(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        self.calls.append(messages)
        if not self.script:
            raise AssertionError("scripted LLM ran out of replies")
        content = self.script.pop(0)
        return LLMResult(content=content, model=self.model, input_tokens=1, output_tokens=2)

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:  # pragma: no cover — not used in agent tests
        if False:
            yield ""

    async def health(self) -> bool:
        return True


class NativeScriptedLLM:
    """Fake OpenAI-like client: replies with structured tool_calls instead of fenced text."""

    def __init__(self, script: list[tuple[str, tuple[ToolCall, ...]]]):
        self.script = list(script)
        self.calls: list[list[ChatMessage]] = []
        self.model = "native-scripted"

    def resolve_model(self, params: GenerationParams) -> str:
        return self.model

    def supports_native_tools(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        *,
        tools=None,
        tool_choice=None,
    ) -> LLMResult:
        self.calls.append(messages)
        if not self.script:
            raise AssertionError("scripted LLM ran out of replies")
        content, tool_calls = self.script.pop(0)
        return LLMResult(
            content=content,
            model=self.model,
            input_tokens=1,
            output_tokens=2,
            tool_calls=tool_calls,
        )

    async def stream_chat(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:  # pragma: no cover — not used in agent tests
        if False:
            yield ""

    async def health(self) -> bool:
        return True


async def _make_echo_tool():
    async def echo(args, ctx):
        return {"seen": args}

    return Tool(
        name="echo",
        description="Echo args",
        input_schema={"type": "object", "additionalProperties": False},
        executor=echo,
    )


async def _make_finish_tool():
    async def finish(args, _ctx):
        return {"summary": args.get("summary") or "done"}

    return Tool(
        name="finish",
        description="finish the task",
        input_schema={"type": "object", "additionalProperties": False},
        executor=finish,
    )


def _params() -> GenerationParams:
    return GenerationParams(model="scripted", temperature=0.1, top_p=0.9, timeout_seconds=30.0)


async def _registry():
    return ToolRegistry([await _make_echo_tool(), await _make_finish_tool()])


async def test_agent_completes_with_finish_tool():
    llm = ScriptedLLM(
        [
            'Plan: echo then finish.\n<tool_call>{"name":"echo","arguments":{"x":1}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"echoed once"}}</tool_call>',
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)
    run = await graph.run("test goal", _params(), ToolContext())
    assert run.completed is True
    assert run.final_summary == "echoed once"
    kinds = [s.kind for s in run.steps]
    assert kinds == ["thought", "tool_call", "tool_result", "tool_call", "tool_result", "final"]


async def test_agent_stream_emits_custom_steps_in_order():
    from app.orchestration.agent_graph import StepRecord

    llm = ScriptedLLM(
        [
            'Plan: echo then finish.\n<tool_call>{"name":"echo","arguments":{"x":1}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"echoed once"}}</tool_call>',
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)
    custom_kinds: list[str] = []
    final_run = None
    async for mode, chunk in graph.stream("test goal", _params(), ToolContext()):
        if mode == "custom":
            assert isinstance(chunk, StepRecord)
            custom_kinds.append(chunk.kind)
        elif mode == "values" and isinstance(chunk, dict) and "run" in chunk:
            final_run = chunk["run"]

    assert custom_kinds == [
        "thought",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert final_run is not None
    assert final_run.completed is True
    assert final_run.final_summary == "echoed once"
    assert [s.kind for s in final_run.steps] == custom_kinds


async def test_agent_stream_emits_mid_act_tools_before_node_ends():
    """Two tools in one reason turn should stream as separate custom chunks mid-act."""
    from app.orchestration.agent_graph import StepRecord

    llm = ScriptedLLM(
        [
            (
                '<tool_call>{"name":"echo","arguments":{"n":1}}</tool_call>\n'
                '<tool_call>{"name":"echo","arguments":{"n":2}}</tool_call>\n'
                '<tool_call>{"name":"finish","arguments":{"summary":"two echos"}}</tool_call>'
            ),
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)
    custom_kinds: list[str] = []
    async for mode, chunk in graph.stream("batch tools", _params(), ToolContext()):
        if mode == "custom" and isinstance(chunk, StepRecord):
            custom_kinds.append(chunk.kind)

    assert custom_kinds == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final",
    ]


async def test_agent_treats_no_tool_call_as_error():
    llm = ScriptedLLM(["Nothing needed, the goal is already met."])
    graph = AgentGraph(llm, await _registry(), max_steps=3)
    run = await graph.run("say hi", _params(), ToolContext())
    assert run.completed is False
    assert run.errored is True
    assert run.error == "model refused to call tools"
    assert run.steps[-1].kind == "error"
    assert "goal is already met" in (run.final_summary or "")


async def test_agent_accepts_answer_after_successful_tool_as_implicit_finish():
    """After a tool runs, a plain-text answer with no finish call completes the run."""
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"echo","arguments":{"x":1}}</tool_call>',
            "There are two saved translations: Russian and Korean.",
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)
    run = await graph.run("list my translations", _params(), ToolContext())
    assert run.completed is True
    assert run.errored is False
    assert run.final_summary == "There are two saved translations: Russian and Korean."
    assert run.steps[-1].kind == "final"


async def test_agent_retries_empty_response_then_fails_explicitly():
    llm = ScriptedLLM(["", ""])
    graph = AgentGraph(llm, await _registry(), max_steps=3)
    run = await graph.run("say hi", _params(), ToolContext())
    assert run.completed is False
    assert run.errored is True
    assert run.error == "model returned an empty response without tool calls"
    assert run.steps[-1].kind == "error"
    assert "previous response was empty" in llm.calls[1][-1].content


async def test_agent_hits_step_limit_without_finish():
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"echo","arguments":{}}</tool_call>',
            '<tool_call>{"name":"echo","arguments":{}}</tool_call>',
            '<tool_call>{"name":"echo","arguments":{}}</tool_call>',
            '<tool_call>{"name":"echo","arguments":{}}</tool_call>',
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=2)
    run = await graph.run("go forever", _params(), ToolContext())
    assert run.completed is False
    assert run.step_limit_hit is True
    assert run.steps[-1].kind == "error"


async def test_agent_records_tool_failure_and_continues():
    llm = ScriptedLLM(
        [
            '<tool_call>{"name":"missing","arguments":{}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"recovered"}}</tool_call>',
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)
    run = await graph.run("try a missing tool", _params(), ToolContext())
    assert run.completed is True
    tool_result_step = next(
        s for s in run.steps if s.kind == "tool_result" and s.tool_name == "missing"
    )
    assert tool_result_step.ok is False
    # The second LLM turn should have received the tool-result observation
    second_turn = llm.calls[1]
    assert any("unknown tool" in m.content for m in second_turn)


async def test_agent_retries_when_model_falsely_claims_no_tool():
    async def list_notes(_args, _ctx):
        return {"items": [{"title": "Ideas"}]}

    registry = ToolRegistry(
        [
            Tool(
                name="list_notes",
                description="List notes",
                input_schema={"type": "object", "additionalProperties": False},
                executor=list_notes,
            ),
            await _make_finish_tool(),
        ]
    )
    llm = ScriptedLLM(
        [
            "I do not have a tool that can list your notes.",
            '<tool_call>{"name":"list_notes","arguments":{}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"listed notes"}}</tool_call>',
        ]
    )

    graph = AgentGraph(llm, registry, max_steps=5)
    run = await graph.run("list my notes", _params(), ToolContext())

    assert run.completed is True
    assert run.final_summary == "listed notes"
    assert any(step.tool_name == "list_notes" and step.ok is True for step in run.steps)
    assert any("Correction: you do have access" in message.content for message in llm.calls[1])


async def test_agent_retries_notes_summary_goal_without_tool_calls():
    async def list_notes(args, _ctx):
        assert args == {"limit": 2}
        return {
            "items": [
                {"id": "note_1", "title": "First"},
                {"id": "note_2", "title": "Second"},
            ]
        }

    async def get_note(args, _ctx):
        return {"id": args["note_id"], "content": f"Content for {args['note_id']}"}

    async def create_note(args, _ctx):
        return {"id": "summary_note", "title": args["title"]}

    registry = ToolRegistry(
        [
            Tool(
                name="list_notes",
                description="List notes",
                input_schema={"type": "object", "additionalProperties": False},
                executor=list_notes,
            ),
            Tool(
                name="get_note",
                description="Get note",
                input_schema={"type": "object", "additionalProperties": False},
                executor=get_note,
            ),
            Tool(
                name="create_note",
                description="Create note",
                input_schema={"type": "object", "additionalProperties": False},
                executor=create_note,
            ),
            await _make_finish_tool(),
        ]
    )
    llm = ScriptedLLM(
        [
            "I can summarize those notes for you.",
            '<tool_call>{"name":"list_notes","arguments":{"limit":2}}</tool_call>',
            (
                '<tool_call>{"name":"get_note","arguments":{"note_id":"note_1"}}</tool_call>'
                '<tool_call>{"name":"get_note","arguments":{"note_id":"note_2"}}</tool_call>'
            ),
            '<tool_call>{"name":"create_note","arguments":{"title":"Summary","content":"Combined summary"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"summary":"created summary note"}}</tool_call>',
        ]
    )

    graph = AgentGraph(llm, registry, max_steps=8)
    run = await graph.run(
        "Summarize my last 2 notes and create a new one with summary",
        _params(),
        ToolContext(),
    )

    assert run.completed is True
    assert run.final_summary == "created summary note"
    assert [step.tool_name for step in run.steps if step.kind == "tool_call"] == [
        "list_notes",
        "get_note",
        "get_note",
        "create_note",
        "finish",
    ]
    first_prompt = "\n".join(message.content for message in llm.calls[0])
    second_prompt = "\n".join(message.content for message in llm.calls[1])
    assert "First call list_notes with limit 2" in first_prompt
    assert "Correction: this task requires note tools" in second_prompt


async def test_agent_notes_summary_empty_responses_do_not_complete_without_output():
    async def list_notes(args, _ctx):
        return {"items": []}

    async def get_note(args, _ctx):
        return {"id": args["note_id"], "content": ""}

    async def create_note(args, _ctx):
        return {"id": "summary_note", "title": args.get("title")}

    registry = ToolRegistry(
        [
            Tool(
                name="list_notes",
                description="List notes",
                input_schema={"type": "object", "additionalProperties": False},
                executor=list_notes,
            ),
            Tool(
                name="get_note",
                description="Get note",
                input_schema={"type": "object", "additionalProperties": False},
                executor=get_note,
            ),
            Tool(
                name="create_note",
                description="Create note",
                input_schema={"type": "object", "additionalProperties": False},
                executor=create_note,
            ),
            await _make_finish_tool(),
        ]
    )
    llm = ScriptedLLM(["", "", ""])

    graph = AgentGraph(llm, registry, max_steps=5)
    run = await graph.run(
        "Summarize last 3 notes and save summary as a new role",
        _params(),
        ToolContext(),
    )

    assert run.completed is False
    assert run.errored is True
    assert run.error == "model returned an empty response without tool calls"
    assert "First call list_notes with limit 3" in "\n".join(
        message.content for message in llm.calls[0]
    )
    assert "Correction: this task requires note tools" in "\n".join(
        message.content for message in llm.calls[1]
    )
    assert "previous response was empty" in "\n".join(message.content for message in llm.calls[2])


async def test_agent_uses_native_tool_calls_when_supported():
    """With a native-tool-calling LLM, no <tool_call> manifest text is used at all."""
    llm = NativeScriptedLLM(
        [
            ("", (ToolCall(name="echo", arguments={"x": 1}, call_id="call_1"),)),
            (
                "",
                (ToolCall(name="finish", arguments={"summary": "echoed once"}, call_id="call_2"),),
            ),
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)

    assert "<tool_call>" not in graph.system_prompt()

    run = await graph.run("test goal", _params(), ToolContext())
    assert run.completed is True
    assert run.final_summary == "echoed once"
    kinds = [s.kind for s in run.steps]
    assert kinds == ["tool_call", "tool_result", "tool_call", "tool_result", "final"]

    # Second reasoning turn's history must carry a matching tool-role reply for call_1.
    second_turn_messages = llm.calls[1]
    tool_replies = [m for m in second_turn_messages if m.role == "tool"]
    assert len(tool_replies) == 1
    assert tool_replies[0].tool_call_id == "call_1"


async def test_agent_native_implicit_finish_without_tool_calls():
    llm = NativeScriptedLLM(
        [
            ("", (ToolCall(name="echo", arguments={"x": 1}, call_id="call_1"),)),
            ("There are two saved translations: Russian and Korean.", ()),
        ]
    )
    graph = AgentGraph(llm, await _registry(), max_steps=5)
    run = await graph.run("list my translations", _params(), ToolContext())
    assert run.completed is True
    assert run.errored is False
    assert run.final_summary == "There are two saved translations: Russian and Korean."
