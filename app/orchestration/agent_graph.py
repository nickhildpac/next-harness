from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.ports.llm import ChatMessage, GenerationParams, LLMClient, ToolCall
from app.tools.protocol import format_tool_result, parse_tool_calls, render_tool_manifest
from app.tools.registry import ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


AGENT_SYSTEM_PREAMBLE = (
    "You are an autonomous task-completion agent. You break a goal down into small steps, "
    "call tools to gather information or take actions, and stop as soon as the goal is met. "
    "Reason briefly, call tools decisively, and call the 'finish' tool once the goal is met. "
    "Use note tools when the user asks to save or organize notes, translation tools for "
    "translation requests, and task document search before answering questions that depend on "
    "ingested document content."
)


@dataclass
class StepRecord:
    kind: Literal["thought", "tool_call", "tool_result", "final", "error"]
    tool_name: str | None = None
    content: str | None = None
    payload: dict | list | None = None
    ok: bool | None = None


@dataclass
class AgentRun:
    goal: str
    final_summary: str | None = None
    completed: bool = False
    step_limit_hit: bool = False
    errored: bool = False
    error: str | None = None
    model: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    turns_used: int = 0


class _AgentState(TypedDict, total=False):
    run: AgentRun
    messages: list[ChatMessage]
    turn: int
    params: GenerationParams
    context: "ToolContext"
    pending_calls: list[ToolCall]


class AgentGraph:
    """LangGraph loop: reason → (optional) tool_calls → observe, repeated until done."""

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_steps: int):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        graph: StateGraph = StateGraph(_AgentState)
        graph.add_node("reason", self._reason)
        graph.add_node("act", self._act)
        graph.set_entry_point("reason")
        graph.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {"act": "act", "end": END},
        )
        graph.add_conditional_edges(
            "act",
            self._route_after_act,
            {"reason": "reason", "end": END},
        )
        self.graph = graph.compile()

    def system_prompt(self) -> str:
        return f"{AGENT_SYSTEM_PREAMBLE}\n\n{render_tool_manifest(self.registry.specs())}"

    async def run(
        self,
        goal: str,
        params: GenerationParams,
        context: ToolContext,
    ) -> AgentRun:
        run = AgentRun(goal=goal)
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt()),
            ChatMessage(role="user", content=f"Goal: {goal}"),
        ]
        state: _AgentState = {
            "run": run,
            "messages": messages,
            "turn": 0,
            "params": params,
            "context": context,
            "pending_calls": [],
        }
        try:
            final_state = await self.graph.ainvoke(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_run_failed", extra={"goal": goal})
            run.errored = True
            run.error = f"{exc.__class__.__name__}: {exc}"
            return run
        return final_state["run"]

    async def _reason(self, state: _AgentState) -> _AgentState:
        run = state["run"]
        params = state["params"]
        state["turn"] = state["turn"] + 1
        run.turns_used = state["turn"]
        result = await self.llm.chat(state["messages"], params)
        run.model = result.model
        content = result.content or ""
        parsed = parse_tool_calls(content)
        state["messages"].append(ChatMessage(role="assistant", content=content))
        if parsed.thought:
            run.steps.append(StepRecord(kind="thought", content=parsed.thought))

        if not parsed.tool_calls:
            summary = parsed.thought or content.strip() or "(no output)"
            run.final_summary = summary
            run.completed = True
            run.steps.append(StepRecord(kind="final", content=summary))
            state["pending_calls"] = []
            return state

        state["pending_calls"] = list(parsed.tool_calls)
        return state

    async def _act(self, state: _AgentState) -> _AgentState:
        run = state["run"]
        context = state["context"]
        pending = state.get("pending_calls", [])
        tool_results: list[ToolResult] = []
        for call in pending:
            run.steps.append(
                StepRecord(
                    kind="tool_call",
                    tool_name=call.name,
                    payload={"arguments": call.arguments, "call_id": call.call_id},
                )
            )
            result = await self.registry.invoke(
                call.name, call.arguments, context, call_id=call.call_id
            )
            run.steps.append(
                StepRecord(
                    kind="tool_result",
                    tool_name=result.name,
                    ok=result.ok,
                    payload={"output": result.output, "error": result.error, "call_id": result.call_id},
                )
            )
            tool_results.append(result)
            if call.name == "finish" and result.ok:
                summary = (result.output or {}).get("summary") if isinstance(result.output, dict) else None
                run.final_summary = summary or "done"
                run.completed = True
                run.steps.append(StepRecord(kind="final", content=run.final_summary))

        observation = "\n".join(
            format_tool_result(r.name, r.call_id, r.output, r.error) for r in tool_results
        )
        state["messages"].append(
            ChatMessage(role="user", content=f"Tool results:\n{observation}")
        )
        state["pending_calls"] = []
        return state

    def _route_after_reason(self, state: _AgentState) -> str:
        run = state["run"]
        if run.completed or run.errored:
            return "end"
        if state.get("pending_calls"):
            return "act"
        return "end"

    def _route_after_act(self, state: _AgentState) -> str:
        run = state["run"]
        if run.completed or run.errored:
            return "end"
        if state["turn"] >= self.max_steps:
            run.step_limit_hit = True
            run.steps.append(
                StepRecord(
                    kind="error",
                    content=f"step limit ({self.max_steps}) reached before finish",
                )
            )
            return "end"
        return "reason"
