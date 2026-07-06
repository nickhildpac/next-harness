from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.ports.llm import ChatMessage, GenerationParams, LLMClient, ToolCall
from app.tools.protocol import format_tool_result, parse_tool_calls, render_tool_manifest
from app.tools.registry import ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


AGENT_SYSTEM_PREAMBLE = (
    "You are an autonomous task-completion agent. You break a goal down into small steps, "
    "call tools to gather information or take actions, and stop as soon as the goal is met. "
    "Reason briefly, call tools decisively, and call the 'finish' tool once the goal is met. "
    "If a user request matches an available tool, call that tool; do not claim you lack access "
    "to it. Use list_notes when the user asks to list, show, or inspect notes. Use get_note "
    "when note content is needed. Use note tools when the user asks to save or organize notes, "
    "translation tools for translation requests, and task document search before answering "
    "questions that depend on ingested document content."
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
    messages: list[BaseMessage]
    turn: int
    params: GenerationParams
    context: "ToolContext"
    pending_calls: list[ToolCall]
    retry_reason: bool
    tool_refusal_retried: bool
    required_tool_retried: bool
    empty_response_retried: bool


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
            {"act": "act", "reason": "reason", "end": END},
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
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt()),
            HumanMessage(content=f"Goal: {goal}"),
        ]
        goal_hint = self._goal_tool_hint(goal, metadata=context.metadata)
        if goal_hint:
            messages.append(HumanMessage(content=goal_hint))
        state: _AgentState = {
            "run": run,
            "messages": messages,
            "turn": 0,
            "params": params,
            "context": context,
            "pending_calls": [],
            "retry_reason": False,
            "tool_refusal_retried": False,
            "required_tool_retried": False,
            "empty_response_retried": False,
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
        state["retry_reason"] = False
        state["turn"] = state["turn"] + 1
        run.turns_used = state["turn"]
        result = await self.llm.chat(self._to_port_messages(state["messages"]), params)
        run.model = result.model
        content = result.content or ""
        parsed = parse_tool_calls(content)
        state["messages"].append(AIMessage(content=content))
        if parsed.thought:
            run.steps.append(StepRecord(kind="thought", content=parsed.thought))

        if not parsed.tool_calls:
            if self._should_retry_false_tool_refusal(content, state):
                state["tool_refusal_retried"] = True
                state["retry_reason"] = True
                state["messages"].append(
                    HumanMessage(
                        content=(
                            "Correction: you do have access to the listed tools. "
                            "For this task, call the appropriate tool using a <tool_call> block. "
                            f"Available tools: {', '.join(self.registry.names())}."
                        )
                    )
                )
                return state
            required_tool_hint = self._required_tool_retry_hint(run.goal, state)
            if required_tool_hint:
                state["required_tool_retried"] = True
                state["retry_reason"] = True
                state["messages"].append(HumanMessage(content=required_tool_hint))
                return state
            if self._should_retry_empty_response(content, state):
                state["empty_response_retried"] = True
                state["retry_reason"] = True
                state["messages"].append(
                    HumanMessage(
                        content=(
                            "Correction: your previous response was empty. "
                            "Either call the required tool using a <tool_call> block, "
                            "or provide a non-empty final answer if the task is already complete."
                        )
                    )
                )
                return state
            if not content.strip():
                run.errored = True
                run.error = "model returned an empty response without tool calls"
                run.steps.append(StepRecord(kind="error", content=run.error))
                state["pending_calls"] = []
                return state
            summary = parsed.thought or content.strip()
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
        state["messages"].append(HumanMessage(content=f"Tool results:\n{observation}"))
        state["pending_calls"] = []
        return state

    def _route_after_reason(self, state: _AgentState) -> str:
        run = state["run"]
        if run.completed or run.errored:
            return "end"
        if state.get("retry_reason") and state["turn"] < self.max_steps:
            return "reason"
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

    def _should_retry_false_tool_refusal(self, content: str, state: _AgentState) -> bool:
        if state.get("tool_refusal_retried") or state["turn"] >= self.max_steps:
            return False
        lowered = content.lower()
        refusal_markers = (
            "don't have a tool",
            "do not have a tool",
            "no tool",
            "lack access",
            "can't access",
            "cannot access",
        )
        return bool(self.registry.names()) and any(marker in lowered for marker in refusal_markers)

    def _should_retry_empty_response(self, content: str, state: _AgentState) -> bool:
        if state.get("empty_response_retried") or state["turn"] >= self.max_steps:
            return False
        return not content.strip()

    def _required_tool_retry_hint(self, goal: str, state: _AgentState) -> str | None:
        if state.get("required_tool_retried") or state["turn"] >= self.max_steps:
            return None
        return self._goal_tool_hint(goal, retry=True, metadata=state["context"].metadata)

    def _goal_tool_hint(
        self, goal: str, *, retry: bool = False, metadata: dict | None = None
    ) -> str | None:
        lowered = goal.lower()
        tool_names = set(self.registry.names())
        uploaded_document_count = int((metadata or {}).get("uploaded_document_count") or 0)
        if uploaded_document_count and {"list_task_documents", "search_task_documents"}.issubset(
            tool_names
        ):
            prefix = (
                "Correction: this task has uploaded documents. "
                if retry
                else "Tool-use plan for this uploaded-document task: "
            )
            return (
                f"{prefix}First call list_task_documents to inspect the uploaded files. "
                f'Then call search_task_documents with query "{goal}". '
                "Use the returned excerpts to answer or summarize the document request. "
                "Finally call finish. Do not answer without tool calls."
            )

        if "note" not in lowered:
            return None
        if not {"list_notes", "get_note", "create_note"}.issubset(tool_names):
            return None

        asks_for_summary = any(word in lowered for word in ("summarize", "summary", "summarise"))
        asks_to_create = any(phrase in lowered for phrase in ("create", "new one", "new note", "save"))
        asks_for_recent = any(word in lowered for word in ("last", "recent", "latest"))
        if not (asks_for_summary and asks_to_create and asks_for_recent):
            return None

        limit = self._recent_note_limit(goal) or 2
        prefix = (
            "Correction: this task requires note tools. "
            if retry
            else "Tool-use plan for this notes task: "
        )
        return (
            f"{prefix}First call list_notes with limit {limit}. "
            "Then call get_note for each returned note id to read full content. "
            "Then call create_note with a concise summary of those notes. "
            "Finally call finish. Do not answer without tool calls."
        )

    def _recent_note_limit(self, goal: str) -> int | None:
        match = re.search(r"\b(?:last|recent|latest)\s+(\d+)\s+notes?\b", goal, re.IGNORECASE)
        if not match:
            return None
        return max(1, min(int(match.group(1)), 20))

    def _to_port_messages(self, messages: list[BaseMessage]) -> list[ChatMessage]:
        return [
            ChatMessage(role=self._message_role(message), content=self._message_content(message))
            for message in messages
        ]

    def _message_role(self, message: BaseMessage) -> str:
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, AIMessage):
            return "assistant"
        return "user"

    def _message_content(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        return str(content)
