from app.tools.protocol import parse_tool_calls, render_tool_manifest
from app.ports.llm import ToolSpec


def test_parse_single_tool_call():
    content = 'Thinking...\n<tool_call>{"name":"now","arguments":{}}</tool_call>'
    parsed = parse_tool_calls(content)
    assert parsed.thought == "Thinking..."
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "now"
    assert parsed.tool_calls[0].arguments == {}


def test_parse_multiple_and_ignores_garbage():
    content = (
        "Let me look these up.\n"
        '<tool_call>{"name":"list_notes","arguments":{"limit":3}}</tool_call>\n'
        "then check time.\n"
        '<tool_call>{"name":"now","arguments":{}}</tool_call>\n'
        '<tool_call>{"totally malformed json}</tool_call>'
    )
    parsed = parse_tool_calls(content)
    assert [c.name for c in parsed.tool_calls] == ["list_notes", "now"]
    assert parsed.tool_calls[0].arguments == {"limit": 3}
    assert "then check time." in parsed.thought
    # Malformed call block is stripped from the visible thought too
    assert "malformed" not in parsed.thought


def test_parse_no_calls_returns_thought_only():
    parsed = parse_tool_calls("Nothing to do here.")
    assert parsed.thought == "Nothing to do here."
    assert parsed.tool_calls == ()


def test_manifest_advertises_tools():
    specs = [
        ToolSpec(name="alpha", description="does alpha", parameters={"type": "object"}),
        ToolSpec(name="beta", description="does beta", parameters={"type": "object"}),
    ]
    manifest = render_tool_manifest(specs)
    assert "alpha" in manifest and "beta" in manifest
    assert "<tool_call>" in manifest
    assert "finish" in manifest.lower()
