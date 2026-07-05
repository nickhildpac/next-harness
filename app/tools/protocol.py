"""Provider-neutral tool-calling protocol.

The agent teaches the LLM to emit tool calls as fenced JSON blocks:

    <tool_call>
    {"name": "list_notes", "arguments": {"limit": 5}}
    </tool_call>

The plain text that appears outside those blocks is preserved as visible model
reasoning. This keeps the agent working with every adapter we ship — no
provider-native tool APIs required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.ports.llm import ToolCall, ToolSpec

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedTurn:
    thought: str
    tool_calls: tuple[ToolCall, ...]


def render_tool_manifest(specs: list[ToolSpec]) -> str:
    lines = ["You have access to the following tools:", ""]
    for i, spec in enumerate(specs, start=1):
        params = json.dumps(spec.parameters, sort_keys=True)
        lines.append(f"{i}. {spec.name} — {spec.description}")
        lines.append(f"   parameters: {params}")
    lines.append("")
    lines.append(
        'To call a tool, emit a block exactly like:\n'
        '<tool_call>\n'
        '{"name": "<tool_name>", "arguments": {"...": "..."}}\n'
        '</tool_call>'
    )
    lines.append(
        "You may emit multiple tool_call blocks in one turn. Any text outside the "
        "blocks is treated as your reasoning and shown to the user."
    )
    lines.append(
        'When the task is complete, call the "finish" tool with a short summary — '
        "do not repeat other tool calls after finishing."
    )
    return "\n".join(lines)


def parse_tool_calls(content: str) -> ParsedTurn:
    if not content:
        return ParsedTurn(thought="", tool_calls=())
    calls: list[ToolCall] = []
    for i, match in enumerate(TOOL_CALL_PATTERN.finditer(content)):
        body = match.group("body")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = parsed.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        call_id = parsed.get("call_id") or f"call_{i}"
        calls.append(ToolCall(name=name, arguments=arguments, call_id=call_id))
    thought = TOOL_CALL_PATTERN.sub("", content).strip()
    return ParsedTurn(thought=thought, tool_calls=tuple(calls))


def format_tool_result(name: str, call_id: str | None, output: Any, error: str | None) -> str:
    payload: dict[str, Any] = {"name": name, "call_id": call_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["output"] = output
    return json.dumps(payload, default=str)
