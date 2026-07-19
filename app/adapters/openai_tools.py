"""Native OpenAI function-calling wire format: our ToolSpec/ToolChoice <-> OpenAI's API shapes."""

from __future__ import annotations

import json
from typing import Any

from app.ports.llm import ToolCall, ToolChoice, ToolSpec


def to_openai(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        },
    }


def tool_choice_openai(tc: ToolChoice) -> Any:
    if tc.mode == "auto":
        return "auto"
    if tc.mode == "none":
        return "none"
    if tc.mode == "required":
        return "required"
    if tc.mode == "force":
        return {"type": "function", "function": {"name": tc.tool_name}}
    raise ValueError(tc.mode)


def parse_openai_tool_calls(message: Any) -> tuple[ToolCall, ...]:
    """Convert an OpenAI ``ChatCompletionMessage.tool_calls`` list into port ``ToolCall``s."""
    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        raw_arguments = raw.function.arguments or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": raw_arguments}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        calls.append(ToolCall(name=raw.function.name, arguments=arguments, call_id=raw.id))
    return tuple(calls)
