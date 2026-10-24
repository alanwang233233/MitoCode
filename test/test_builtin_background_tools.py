"""Tests for the Background and BackgroundResult tools."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.tool.builtin.background_tools import background, background_result
from mitocode.core.tool.builtin.context import ToolContext, set_current
from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError
from mitocode.core.tool.tool_manager import ToolManager


@pytest.fixture
def manager(tmp_path):
    from mitocode.core.workspace.workspace import Workspace

    tm = ToolManager()
    tm.register(_echo, name="Echo")
    tm.register(_slow, name="Slow")
    tm.register(_boom, name="Boom")
    ws = Workspace(tmp_path, "t", 2)
    set_current(ToolContext(workspace=ws, tool_manager=tm))
    return tm


async def _echo(text: str) -> str:
    return f"echo:{text}"


async def _slow() -> str:
    await asyncio.sleep(30)
    return "done"


async def _boom() -> str:
    raise ValueError("inner failure")


def _call(coro):
    return asyncio.run(coro)


def test_background_returns_a_handle_immediately(manager):
    out = _call(background('Echo("hi")', 5000))
    assert out.startswith("bg_")


def test_background_does_not_wait_for_the_call(manager):
    out = _call(background("Slow()", 5000))
    assert out.startswith("bg_")


@pytest.mark.asyncio
async def test_background_result_reports_the_output(manager):
    handle = await background('Echo("hi")', 5000)
    await asyncio.sleep(0.05)
    assert "echo:hi" in await background_result(handle)


def test_background_result_on_an_unknown_handle_says_so(manager):
    assert "unknown" in _call(background_result("bg_42")).lower()


def test_background_rejects_an_unknown_tool(manager):
    out = _call(background('NoSuchTool("x")', 5000))
    assert out.startswith("Error:")
    assert "NoSuchTool" in out


def test_background_rejects_a_malformed_call(manager):
    assert _call(background("not a call", 5000)).startswith("Error:")


def test_background_reports_missing_arguments(manager):
    out = _call(background("Echo()", 5000))
    assert out.startswith("Error:")
    assert "text" in out


def test_background_without_a_tool_manager_says_so():
    from mitocode.core.tool.builtin.context import ToolContext, set_current

    set_current(ToolContext())
    assert _call(background('Echo("hi")', 5000)).startswith("Error:")


@pytest.mark.asyncio
async def test_background_result_marks_a_failed_call(manager):
    handle = await background("Boom()", 5000)
    await asyncio.sleep(0.05)
    out = await background_result(handle)
    assert "Error" in out or "failed" in out.lower()


def test_background_rejects_infinite_timeout():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout
    from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        _resolve_timeout(1e400)
    assert "infinite" in str(excinfo.value).lower()


def test_background_rejects_nan_timeout():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout
    from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        _resolve_timeout(float("nan"))
    assert "finite" in str(excinfo.value).lower()


def test_background_rejects_boolean_timeout():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout
    from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        _resolve_timeout(True)
    assert "boolean" in str(excinfo.value).lower()


def test_background_clamps_timeout_above_one_hour():
    from mitocode.core.tool.builtin.background_tools import (
        _resolve_timeout,
        MAX_TIMEOUT_MS,
    )

    timeout_ms, note = _resolve_timeout(10_000_000)  # 10M ms = ~2.7 hours
    assert timeout_ms == MAX_TIMEOUT_MS
    assert note is not None
    assert "clamped" in note.lower()


def test_background_clamps_negative_timeout_to_default():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout

    timeout_ms, note = _resolve_timeout(-1)
    assert timeout_ms == 300_000  # default 5 minutes


def test_background_accepts_zero_timeout_as_default():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout

    timeout_ms, note = _resolve_timeout(0)
    assert timeout_ms == 300_000  # default


def test_background_accepts_none_timeout_as_default():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout

    timeout_ms, note = _resolve_timeout(None)
    assert timeout_ms == 300_000  # default


def test_background_accepts_string_timeout():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout

    with pytest.raises(Exception) as excinfo:
        _resolve_timeout("soon")
    assert "number" in str(excinfo.value).lower()


def test_background_accepts_list_timeout():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout

    with pytest.raises(Exception) as excinfo:
        _resolve_timeout(["soon"])
    assert "number" in str(excinfo.value).lower()


def test_background_accepts_object_timeout():
    from mitocode.core.tool.builtin.background_tools import _resolve_timeout

    with pytest.raises(Exception) as excinfo:
        _resolve_timeout(object())
    assert "number" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_background_timeout_passes_through_to_registry():
    from mitocode.core.tool.builtin.background import BackgroundRegistry

    registry = BackgroundRegistry()
    received = {}

    async def record_timeout():
        received["timeout"] = 12345

    handle = registry.launch(record_timeout, caller_id=None, timeout=12345)
    # The registry should store the timeout
    # Check the record instead of _tasks since the task may complete and be removed quickly
    assert handle in registry._records
    record = registry._records[handle]
    assert record.timeout == 12345
    # Wait for the task to complete
    await asyncio.sleep(0.01)
    # Now collect to verify it finished
    record = await registry.collect(handle)
    assert record.state.name == "DONE"
