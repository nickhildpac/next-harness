from types import SimpleNamespace

from app.adapters.openai_tools import parse_openai_tool_calls, to_openai, tool_choice_openai
from app.ports.llm import ToolChoice, ToolSpec


def _spec(**overrides):
    defaults = dict(
        name="list_notes",
        description="List notes",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        strict=True,
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


def test_to_openai_wraps_spec_as_function_tool():
    payload = to_openai(_spec())
    assert payload == {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List notes",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "strict": True,
        },
    }


def test_tool_choice_openai_modes():
    assert tool_choice_openai(ToolChoice(mode="auto")) == "auto"
    assert tool_choice_openai(ToolChoice(mode="none")) == "none"
    assert tool_choice_openai(ToolChoice(mode="required")) == "required"
    assert tool_choice_openai(ToolChoice(mode="force", tool_name="finish")) == {
        "type": "function",
        "function": {"name": "finish"},
    }


def test_tool_choice_openai_rejects_unknown_mode():
    try:
        tool_choice_openai(ToolChoice(mode="bogus"))
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _raw_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_parse_openai_tool_calls_decodes_arguments():
    message = SimpleNamespace(
        tool_calls=[
            _raw_call("call_1", "list_notes", '{"limit": 5}'),
            _raw_call("call_2", "finish", '{"summary": "done"}'),
        ]
    )
    calls = parse_openai_tool_calls(message)
    assert calls[0].name == "list_notes"
    assert calls[0].arguments == {"limit": 5}
    assert calls[0].call_id == "call_1"
    assert calls[1].name == "finish"
    assert calls[1].arguments == {"summary": "done"}


def test_parse_openai_tool_calls_handles_missing_and_malformed():
    assert parse_openai_tool_calls(SimpleNamespace(tool_calls=None)) == ()
    message = SimpleNamespace(tool_calls=[_raw_call("call_1", "broken", "not json")])
    calls = parse_openai_tool_calls(message)
    assert calls[0].arguments == {"_raw": "not json"}
