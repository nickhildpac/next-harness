from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph, add_messages

from app.guardrails import SAFE_OUTPUT_REPLACEMENT, GuardrailResult, Guardrails
from app.ports.llm import ChatMessage, GenerationParams, LLMClient, ToolCall, ToolChoice
from app.ports.tools import ToolInvoker
from app.tools.protocol import format_tool_result, parse_tool_calls, render_tool_manifest
from app.tools.registry import ToolContext, ToolResult

logger = logging.getLogger(__name__)


def _supports_native_tools(llm: LLMClient) -> bool:
    # Duck-typed rather than a direct attribute access: fakes/stubs used in tests implement
    # LLMClient structurally without subclassing it, so they don't inherit the Protocol default.
    probe = getattr(llm, "supports_native_tools", None)
    return bool(probe()) if callable(probe) else False


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
    kind: Literal["thought", "tool_call", "tool_result", "final", "error", "guardrail"]
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
    # Set when the output wall withheld an unsafe/sensitive result. The run still
    # completes, but final_summary is replaced with a safe placeholder.
    output_blocked: bool = False
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
    # Prior thread summaries handed to the run. Held here (not pre-seeded into messages)
    # so the input wall can sanitize it in guard_input before the model sees it.
    prior_context: str | None
    pending_calls: list[ToolCall]
    retry_reason: bool
    tool_refusal_retried: bool
    required_tool_retried: bool
    empty_response_retried: bool


class AgentGraph:
    """LangGraph loop: reason → (optional) tool_calls → observe, repeated until done."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolInvoker,
        max_steps: int,
        *,
        guardrails: Guardrails | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # Nothing passes unchecked: default to active guardrails when none is supplied.
        self.guardrails = guardrails or Guardrails()
        graph: StateGraph = StateGraph(_AgentState)
        graph.add_node("guard_input", self._guard_input)
        graph.add_node("reason", self._reason)
        graph.add_node("act", self._act)
        graph.add_node("guard_output", self._guard_output)
        graph.set_entry_point("guard_input")
        graph.add_conditional_edges(
            "guard_input",
            self._route_after_guard_input,
            {"reason": "reason", "end": END},
        )
        graph.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {"act": "act", "reason": "reason", "guard_output": "guard_output", "end": END},
        )
        graph.add_conditional_edges(
            "act",
            self._route_after_act,
            {"reason": "reason", "guard_output": "guard_output", "end": END},
        )
        graph.add_edge("guard_output", END)
        self.graph = graph.compile()

    def system_prompt(self) -> str:
        if _supports_native_tools(self.llm):
            # Tools are advertised to the model via the API's native tools= field, not text.
            return AGENT_SYSTEM_PREAMBLE
        return f"{AGENT_SYSTEM_PREAMBLE}\n\n{render_tool_manifest(self.tools.specs())}"

    def _initial_state(
        self,
        goal: str,
        params: GenerationParams,
        context: ToolContext,
        *,
        prior_context: str | None = None,
    ) -> tuple[AgentRun, _AgentState]:
        run = AgentRun(goal=goal)
        # The goal AND thread-history messages are appended by the guard_input node so the
        # LLM only ever sees sanitized input (or nothing, if the input wall blocks the run).
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt())]
        state: _AgentState = {
            "run": run,
            "messages": messages,
            "turn": 0,
            "work_turn": 0,  # FIX #3: Only incremented for non-retry reasoning turns.
            "params": params,
            "context": context,
            "prior_context": prior_context,
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
        *,
        prior_context: str | None = None,
    ) -> AgentRun:
        run, state = self._initial_state(goal, params, context, prior_context=prior_context)
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
        *,
        prior_context: str | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield ``(mode, chunk)`` from LangGraph ``astream`` (custom + values)."""
        run, state = self._initial_state(goal, params, context, prior_context=prior_context)
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

    def _emit_guardrail_findings(self, run: AgentRun, wall: str, result: GuardrailResult) -> None:
        # Only surface a step when a wall actually caught something, so clean runs keep a
        # tidy trace (and the step sequence tests rely on) while flagged runs stay auditable.
        for finding in result.findings:
            self._emit_step(
                run,
                StepRecord(
                    kind="guardrail",
                    content=(
                        f"{wall} guardrail {finding.action} {finding.category}: {finding.detail}"
                    ),
                    payload={
                        "wall": wall,
                        "category": finding.category,
                        "detail": finding.detail,
                        "action": finding.action,
                    },
                    ok=finding.action != "blocked",
                ),
            )

    def _block_input(self, run: AgentRun, source: str, result: GuardrailResult) -> None:
        run.errored = True
        run.error = f"input guardrail blocked {source}: {result.blocked_reason}"
        self._emit_step(run, StepRecord(kind="error", content=run.error))

    async def _guard_input(self, state: _AgentState) -> _AgentState:
        """Wall 1: sanitize every piece of input (thread history and goal) before it reaches
        the model; block on prompt injection."""
        run = state["run"]
        context = state["context"]
        messages_to_add: list[BaseMessage] = []

        # Thread history is input the model reads, so it passes through the same wall as the
        # goal: PII is redacted and prompt injection blocks the run before the model sees it.
        prior_context = state.get("prior_context")
        if prior_context:
            history = self.guardrails.check_input(prior_context)
            self._emit_guardrail_findings(run, "input:thread_history", history)
            if not history.allowed:
                self._block_input(run, "thread history", history)
                state["messages"] = []
                return state
            messages_to_add.append(HumanMessage(content=f"Thread history:\n{history.text}"))

        result = self.guardrails.check_input(run.goal)
        self._emit_guardrail_findings(run, "input", result)
        if not result.allowed:
            self._block_input(run, "goal", result)
            state["messages"] = []
            return state
        messages_to_add.append(HumanMessage(content=f"Goal: {result.text}"))
        goal_hint = self._goal_tool_hint(result.text, metadata=context.metadata)
        if goal_hint:
            messages_to_add.append(HumanMessage(content=goal_hint))
        state["messages"] = messages_to_add
        return state

    async def _guard_output(self, state: _AgentState) -> _AgentState:
        """Wall 2: withhold unsafe/secret results and redact PII before the user sees them."""
        run = state["run"]
        result = self.guardrails.check_output(run.final_summary or "")
        self._emit_guardrail_findings(run, "output", result)
        if not result.allowed:
            run.output_blocked = True
            run.final_summary = SAFE_OUTPUT_REPLACEMENT
            self._emit_step(
                run,
                StepRecord(
                    kind="error",
                    content=f"output guardrail blocked: {result.blocked_reason}",
                ),
            )
        elif result.redacted:
            run.final_summary = result.text
        # No message delta: this node only rewrites the result and emits steps.
        state["messages"] = []
        return state

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
        native = _supports_native_tools(self.llm)
        port_messages = self._to_port_messages(state["messages"])
        if native:
            result = await self.llm.chat(
                port_messages,
                params,
                tools=self.tools.specs(),
                tool_choice=ToolChoice(mode="auto"),
            )
        else:
            result = await self.llm.chat(port_messages, params)
        run.model = result.model
        content = result.content or ""
        if native:
            thought = content.strip()
            tool_calls = result.tool_calls
        else:
            parsed = parse_tool_calls(content)
            thought = parsed.thought
            tool_calls = parsed.tool_calls
        # FIX #5: Return message updates instead of mutating state in place.
        messages_to_add: list[BaseMessage] = [self._build_assistant_message(content, tool_calls)]
        if thought:
            self._emit_step(run, StepRecord(kind="thought", content=thought))

        if not tool_calls:
            if self._should_retry_false_tool_refusal(content, state):
                state["tool_refusal_retried"] = True
                state["retry_reason"] = True
                correction = (
                    "Call the appropriate tool."
                    if native
                    else "For this task, call the appropriate tool using a <tool_call> block."
                )
                messages_to_add.append(
                    HumanMessage(
                        content=(
                            "Correction: you do have access to the listed tools. "
                            f"{correction} Available tools: {', '.join(self.tools.names())}."
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
                required_tool = (
                    "call the required tool"
                    if native
                    else "call the required tool using a <tool_call> block"
                )
                messages_to_add.append(
                    HumanMessage(
                        content=(
                            "Correction: your previous response was empty. "
                            f"Either {required_tool}, "
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
            summary = thought or content.strip()
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

        state["pending_calls"] = list(tool_calls)
        state["messages"] = messages_to_add
        return state

    async def _act(self, state: _AgentState) -> _AgentState:
        run = state["run"]
        context = state["context"]
        pending = state.get("pending_calls", [])
        native = _supports_native_tools(self.llm)
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

        # FIX #5: Return message updates instead of mutating state in place.
        if native:
            # Each native tool call needs its own tool-role reply, paired by call_id, so the
            # next turn's message history is valid for the provider's tool-calling API.
            state["messages"] = [
                ToolMessage(
                    content=json.dumps(r.output if r.ok else {"error": r.error}, default=str),
                    tool_call_id=r.call_id or "",
                    name=r.name,
                )
                for r in tool_results
            ]
        else:
            observation = "\n".join(
                format_tool_result(r.name, r.call_id, r.output, r.error) for r in tool_results
            )
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

    def _route_after_guard_input(self, state: _AgentState) -> str:
        # A blocked input marks the run errored; otherwise proceed to the first reason turn.
        return "end" if state["run"].errored else "reason"

    def _route_after_reason(self, state: _AgentState) -> str:
        run = state["run"]
        # A completed run always exits through the output wall before delivery.
        if run.completed:
            return "guard_output"
        if run.errored:
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
        # A completed run always exits through the output wall before delivery.
        if run.completed:
            return "guard_output"
        if run.errored or run.step_limit_hit:
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

    @staticmethod
    def _build_assistant_message(content: str, tool_calls: tuple[ToolCall, ...]) -> AIMessage:
        if not tool_calls:
            return AIMessage(content=content)
        return AIMessage(
            content=content,
            tool_calls=[
                {
                    "name": call.name,
                    "args": call.arguments,
                    "id": call.call_id or f"call_{i}",
                    "type": "tool_call",
                }
                for i, call in enumerate(tool_calls)
            ],
        )

    def _to_port_messages(self, messages: list[BaseMessage]) -> list[ChatMessage]:
        return [self._to_port_message(message) for message in messages]

    def _to_port_message(self, message: BaseMessage) -> ChatMessage:
        role = self._message_role(message)
        content = self._message_content(message)
        if isinstance(message, ToolMessage):
            return ChatMessage(role=role, content=content, tool_call_id=message.tool_call_id)
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = tuple(
                ToolCall(name=tc["name"], arguments=tc.get("args") or {}, call_id=tc.get("id"))
                for tc in message.tool_calls
            )
            return ChatMessage(role=role, content=content, tool_calls=tool_calls)
        return ChatMessage(role=role, content=content)

    def _message_role(self, message: BaseMessage) -> str:
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, ToolMessage):
            return "tool"
        return "user"

    def _message_content(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        # FIX #5: For multimodal content (list of dicts), serialize to JSON to avoid Python repr.
        # Currently only str messages are created, but this protects against future multimodal parts.
        if isinstance(content, list):
            return json.dumps(content)
        return str(content)
