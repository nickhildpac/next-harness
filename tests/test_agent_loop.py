from collections.abc import AsyncIterator

from app.orchestration.agent_graph import AgentGraph
from app.ports.llm import ChatMessage, GenerationParams, LLMResult
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


async def _make_echo_tool():
    async def echo(args, ctx):
        return {"seen": args}

    return Tool(name="echo", description="Echo args", parameters={"type": "object"}, handler=echo)


async def _make_finish_tool():
    async def finish(args, _ctx):
        return {"summary": args.get("summary") or "done"}

    return Tool(
        name="finish",
        description="finish the task",
        parameters={"type": "object"},
        handler=finish,
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


async def test_agent_treats_no_tool_call_as_final_answer():
    llm = ScriptedLLM(["Nothing needed, the goal is already met."])
    graph = AgentGraph(llm, await _registry(), max_steps=3)
    run = await graph.run("say hi", _params(), ToolContext())
    assert run.completed is True
    assert "goal is already met" in run.final_summary
    assert run.steps[-1].kind == "final"


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
    tool_result_step = next(s for s in run.steps if s.kind == "tool_result" and s.tool_name == "missing")
    assert tool_result_step.ok is False
    # The second LLM turn should have received the tool-result observation
    second_turn = llm.calls[1]
    assert any("unknown tool" in m.content for m in second_turn)
