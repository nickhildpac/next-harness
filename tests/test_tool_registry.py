import pytest

from app.tools.registry import Tool, ToolContext, ToolError, ToolRegistry


async def _echo(args, ctx):
    return {"echo": args, "user": ctx.user_id}


async def _boom(_args, _ctx):
    raise ToolError("planned failure")


async def _explode(_args, _ctx):
    raise RuntimeError("unexpected")


def _make_tool(name, handler):
    return Tool(name=name, description=name, parameters={"type": "object"}, handler=handler)


async def test_registry_invoke_success():
    registry = ToolRegistry([_make_tool("echo", _echo)])
    result = await registry.invoke("echo", {"a": 1}, ToolContext(user_id="alice"))
    assert result.ok is True
    assert result.output == {"echo": {"a": 1}, "user": "alice"}


async def test_registry_reports_missing_tool():
    registry = ToolRegistry()
    result = await registry.invoke("nope", {}, ToolContext())
    assert result.ok is False
    assert "unknown tool" in result.error


async def test_registry_surfaces_tool_error():
    registry = ToolRegistry([_make_tool("boom", _boom)])
    result = await registry.invoke("boom", {}, ToolContext())
    assert result.ok is False
    assert result.error == "planned failure"


async def test_registry_wraps_unexpected_exception():
    registry = ToolRegistry([_make_tool("explode", _explode)])
    result = await registry.invoke("explode", {}, ToolContext())
    assert result.ok is False
    assert "RuntimeError" in result.error


async def test_registry_rejects_duplicate_registration():
    registry = ToolRegistry([_make_tool("dup", _echo)])
    with pytest.raises(ValueError):
        registry.register(_make_tool("dup", _echo))


async def test_registry_tools_lists_registered():
    registry = ToolRegistry([_make_tool("echo", _echo), _make_tool("boom", _boom)])
    assert [t.name for t in registry.tools()] == ["echo", "boom"]
