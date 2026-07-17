from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph, add_messages

from app.ports.llm import ChatMessage, GenerationParams, LLMClient, ToolCall
from app.ports.tools import ToolInvoker
from app.tools.protocol import format_tool_result, parse_tool_calls, render_tool_manifest
from app.tools.registry import ToolContext, ToolResult

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
    # FIX #3: Track retry turns separately. One-shot retries (tool_refusal, required_tool,
    # empty_response) increment retry_turns; actual work increments turns_used. Step limit
    # applies to turns_used only, not retry turns (retries are free corrections).
    retry_turns: int = 0


# FIX #5: Use idiomatic LangGraph pattern with Annotated reducer. Nodes return message updates,
# not the full state. This is clearer, checkpointable, and protects against concurrent mutation.
class _AgentState(TypedDict, total=False):
    run: AgentRun
    messages: Annotated[list[BaseMessage], add_messages]
    turn: int
    # FIX #3: Track work turns separately from retry turns. Work turns are used for tool calls;
    # retry turns are free corrections and don't count against the step limit.
    work_turn: int
    params: GenerationParams
    context: "ToolContext"
    pending_calls: list[ToolCall]
    retry_reason: bool
    tool_refusal_retried: bool
    required_tool_retried: bool
    empty_response_retried: bool


class AgentGraph:
    """LangGraph loop: reason → (optional) tool_calls → observe, repeated until done."""

    def __init__(self, llm: LLMClient, tools: ToolInvoker, max_steps: int):
        self.llm = llm
        self.tools = tools
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
        return f"{AGENT_SYSTEM_PREAMBLE}\n\n{render_tool_manifest(self.tools.specs())}"

    def _initial_state(
        self,
        goal: str,
        params: GenerationParams,
        context: ToolContext,
    ) -> tuple[AgentRun, _AgentState]:
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
            "work_turn": 0,  # FIX #3: Only incremented for non-retry reasoning turns.
            "params": params,
            "context": context,
            "pending_calls": [],
            "retry_reason": False,
            "tool_refusal_retried": False,
            "required_tool_retried": False,
            "empty_response_retried": False,
        }
        return run, state

    async def run(
        self,
        goal: str,
        params: GenerationParams,
        context: ToolContext,
    ) -> AgentRun:
        run, state = self._initial_state(goal, params, context)
        try:
            final_state = await self.graph.ainvoke(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_run_failed", extra={"goal": goal})
            run.errored = True
            run.error = f"{exc.__class__.__name__}: {exc}"
            return run
        return final_state["run"]

    async def stream(
        self,
        goal: str,
        params: GenerationParams,
        context: ToolContext,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield ``(mode, chunk)`` from LangGraph ``astream`` (custom + values)."""
        run, state = self._initial_state(goal, params, context)
        try:
            async for mode, chunk in self.graph.astream(state, stream_mode=["custom", "values"]):
                yield mode, chunk
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_stream_failed", extra={"goal": goal})
            run.errored = True
            run.error = f"{exc.__class__.__name__}: {exc}"
            yield "values", {"run": run}

    @staticmethod
    def _emit_step(run: AgentRun, step: StepRecord) -> StepRecord:
        run.steps.append(step)
        get_stream_writer()(step)
        return step

    async def _reason(self, state: _AgentState) -> _AgentState:
        run = state["run"]
        params = state["params"]
        is_retry = state.get("retry_reason", False)
        state["retry_reason"] = False
        state["turn"] = state["turn"] + 1
        # FIX #3: Only increment work_turn for non-retry reasoning. Retries are free.
        if not is_retry:
            state["work_turn"] = state["work_turn"] + 1
            run.turns_used = state["work_turn"]
        else:
            run.retry_turns = run.retry_turns + 1
        result = await self.llm.chat(self._to_port_messages(state["messages"]), params)
        run.model = result.model
        content = result.content or ""
        parsed = parse_tool_calls(content)
        # FIX #5: Return message updates instead of mutating state in place.
        messages_to_add: list[BaseMessage] = [AIMessage(content=content)]
        if parsed.thought:
            self._emit_step(run, StepRecord(kind="thought", content=parsed.thought))

        if not parsed.tool_calls:
            if self._should_retry_false_tool_refusal(content, state):
                state["tool_refusal_retried"] = True
                state["retry_reason"] = True
                messages_to_add.append(
                    HumanMessage(
                        content=(
                            "Correction: you do have access to the listed tools. "
                            "For this task, call the appropriate tool using a <tool_call> block. "
                            f"Available tools: {', '.join(self.tools.names())}."
                        )
                    )
                )
                state["messages"] = messages_to_add
                return state
            required_tool_hint = self._required_tool_retry_hint(run.goal, state)
            if required_tool_hint:
                state["required_tool_retried"] = True
                state["retry_reason"] = True
                messages_to_add.append(HumanMessage(content=required_tool_hint))
                state["messages"] = messages_to_add
                return state
            if self._should_retry_empty_response(content, state):
                state["empty_response_retried"] = True
                state["retry_reason"] = True
                messages_to_add.append(
                    HumanMessage(
                        content=(
                            "Correction: your previous response was empty. "
                            "Either call the required tool using a <tool_call> block, "
                            "or provide a non-empty final answer if the task is already complete."
                        )
                    )
                )
                state["messages"] = messages_to_add
                return state
            if not content.strip():
                run.errored = True
                run.error = "model returned an empty response without tool calls"
                self._emit_step(run, StepRecord(kind="error", content=run.error))
                state["pending_calls"] = []
                state["messages"] = messages_to_add
                return state
            summary = parsed.thought or content.strip()
            # FIX #7: If the agent already ran a tool successfully, a non-empty answer with no
            # further tool calls is an implicit finish (the model gathered data and reported it),
            # not a refusal. Treat it as completion so, e.g., "list my translations" doesn't fail
            # just because the model skipped an explicit finish call.
            if self._has_successful_tool_call(run):
                run.final_summary = summary
                run.completed = True
                self._emit_step(run, StepRecord(kind="final", content=summary))
                state["pending_calls"] = []
                state["messages"] = messages_to_add
                return state
            # FIX #1: Model never called any tool and retries are exhausted. This is a genuine
            # refusal — mark as error, not completion. Only finish/implicit-finish complete a run.
            run.final_summary = summary
            run.errored = True
            run.error = "model refused to call tools"
            self._emit_step(run, StepRecord(kind="error", content=f"Model refusal: {summary}"))
            state["pending_calls"] = []
            state["messages"] = messages_to_add
            return state

        state["pending_calls"] = list(parsed.tool_calls)
        state["messages"] = messages_to_add
        return state

    async def _act(self, state: _AgentState) -> _AgentState:
        run = state["run"]
        context = state["context"]
        pending = state.get("pending_calls", [])
        tool_results: list[ToolResult] = []
        for call in pending:
            self._emit_step(
                run,
                StepRecord(
                    kind="tool_call",
                    tool_name=call.name,
                    payload={"arguments": call.arguments, "call_id": call.call_id},
                ),
            )
            result = await self.tools.invoke(
                call.name, call.arguments, context, call_id=call.call_id
            )
            self._emit_step(
                run,
                StepRecord(
                    kind="tool_result",
                    tool_name=result.name,
                    ok=result.ok,
                    payload={
                        "output": result.output,
                        "error": result.error,
                        "call_id": result.call_id,
                    },
                ),
            )
            tool_results.append(result)
            # FIX #4: Short-circuit batch on successful finish to avoid wasted tool calls.
            if call.name == "finish" and result.ok:
                summary = (
                    (result.output or {}).get("summary")
                    if isinstance(result.output, dict)
                    else None
                )
                run.final_summary = summary or "done"
                run.completed = True
                self._emit_step(run, StepRecord(kind="final", content=run.final_summary))
                # Break early: don't execute remaining pending calls after finish succeeds.
                break

        observation = "\n".join(
            format_tool_result(r.name, r.call_id, r.output, r.error) for r in tool_results
        )
        # FIX #5: Return message updates instead of mutating state in place.
        state["messages"] = [HumanMessage(content=f"Tool results:\n{observation}")]
        state["pending_calls"] = []

        # Emit step-limit here (not in the router) so custom stream clients see it.
        if not run.completed and not run.errored and state["work_turn"] >= self.max_steps:
            run.step_limit_hit = True
            self._emit_step(
                run,
                StepRecord(
                    kind="error",
                    content=f"step limit ({self.max_steps}) reached before finish",
                ),
            )
        return state

    def _route_after_reason(self, state: _AgentState) -> str:
        run = state["run"]
        if run.completed or run.errored:
            return "end"
        if state.get("retry_reason") and state["turn"] < self.max_steps:
            return "reason"
        if state.get("pending_calls"):
            return "act"
        # NOTE: Unreachable under normal conditions—_reason always sets completed, errored,
        # retry_reason, or pending_calls before returning. Kept for defensive robustness.
        return "end"

    def _route_after_act(self, state: _AgentState) -> str:
        run = state["run"]
        if run.completed or run.errored or run.step_limit_hit:
            return "end"
        return "reason"

    @staticmethod
    def _has_successful_tool_call(run: AgentRun) -> bool:
        # A successful tool_result means the agent gathered real information; a subsequent
        # tool-free answer is a report of that work, not a refusal to use tools.
        return any(step.kind == "tool_result" and step.ok for step in run.steps)

    def _should_retry_false_tool_refusal(self, content: str, state: _AgentState) -> bool:
        # FIX #2/#3: Use work_turn for limit (retries are free). Use > not >= for final turn.
        if state.get("tool_refusal_retried") or state["work_turn"] > self.max_steps:
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
        return bool(self.tools.names()) and any(marker in lowered for marker in refusal_markers)

    def _should_retry_empty_response(self, content: str, state: _AgentState) -> bool:
        # FIX #2/#3: Use work_turn for limit (retries are free). Use > not >= for final turn.
        if state.get("empty_response_retried") or state["work_turn"] > self.max_steps:
            return False
        return not content.strip()

    def _required_tool_retry_hint(self, goal: str, state: _AgentState) -> str | None:
        # FIX #2/#3: Use work_turn for limit (retries are free). Use > not >= for final turn.
        if state.get("required_tool_retried") or state["work_turn"] > self.max_steps:
            return None
        return self._goal_tool_hint(goal, retry=True, metadata=state["context"].metadata)

    def _goal_tool_hint(
        self, goal: str, *, retry: bool = False, metadata: dict | None = None
    ) -> str | None:
        # FIX #6: Hardcoded heuristics for note and document tasks are domain-specific.
        # Consider extracting to config/pluggable module if tools or domains change.
        lowered = goal.lower()
        tool_names = set(self.tools.names())
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
        asks_to_create = any(
            phrase in lowered for phrase in ("create", "new one", "new note", "save")
        )
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
        # FIX #5: For multimodal content (list of dicts), serialize to JSON to avoid Python repr.
        # Currently only str messages are created, but this protects against future multimodal parts.
        if isinstance(content, list):
            import json

            return json.dumps(content)
        return str(content)
