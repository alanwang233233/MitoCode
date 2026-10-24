"""Tests for the Background and BackgroundResult tools."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.core.tool.builtin.context as context_module
from mitocode.core.tool.builtin.background import TaskState
from mitocode.core.tool.builtin.background_tools import (
    MAX_TIMEOUT_MS,
    background,
    background_result,
)
from mitocode.core.tool.builtin.context import ToolContext, current
from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError
from mitocode.core.tool.tool_manager import ToolManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Register the stub tools and install an isolated current context.

    The context is installed by patching the context module's ``_CURRENT``
    rather than by ``set_current``, so pytest restores the previous value at
    teardown: ``set_current`` would leave this file's tmp_path workspace and
    stub tool manager visible to every later test in the same process.

    Args:
        tmp_path: The directory the test workspace is rooted at.
        monkeypatch: Pytest's restore-on-teardown patcher.

    Returns:
        The tool manager the context was built around.
    """
    from mitocode.core.workspace.workspace import Workspace

    tm = ToolManager()
    tm.register(_echo, name="Echo")
    tm.register(_slow, name="Slow")
    tm.register(_boom, name="Boom")
    ws = Workspace(tmp_path, "t", 2)
    monkeypatch.setattr(
        context_module, "_CURRENT", ToolContext(workspace=ws, tool_manager=tm)
    )
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


async def _wait_for_result(handle, attempts=200):
    """Poll BackgroundResult until the task leaves RUNNING, with a bound.

    A fixed sleep races the detached task: on a slow runner the assertion
    runs before the task settles and fails intermittently, on a fast one it
    just wastes time.  Poll instead, and fail with a clear sentence when the
    bound is exhausted.

    Args:
        handle: The ``bg_<n>`` handle returned by ``background``.
        attempts: How many 10 ms polls to allow before failing.

    Returns:
        The first BackgroundResult rendering that is not "still running".
    """
    for _ in range(attempts):
        out = await background_result(handle)
        if "still running" not in out:
            return out
        await asyncio.sleep(0.01)
    raise AssertionError(f"{handle} did not finish within the polling bound")


async def _collect_terminal(handle, attempts=200):
    """Poll the current context's registry until *handle* leaves RUNNING.

    Args:
        handle: The ``bg_<n>`` handle to redeem.
        attempts: How many 10 ms polls to allow before failing.

    Returns:
        The first record whose state is terminal.
    """
    for _ in range(attempts):
        record = await current().background.collect(handle)
        if record.state is not TaskState.RUNNING:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(f"{handle} did not finish within the polling bound")


def test_background_returns_a_handle_immediately(manager):
    out = _call(background('Echo("hi")', 5000))
    assert out.startswith("bg_")


@pytest.mark.asyncio
async def test_background_does_not_wait_for_the_call(manager):
    """The tool returns the handle without waiting for Slow's 30-second run.

    The one-second bound is the test: if ``background`` ever started
    awaiting the detached call, this fails long before Slow finishes --
    which a plain ``startswith("bg_")`` assertion could never catch.
    """
    out = await asyncio.wait_for(background("Slow()", 5000), timeout=1)
    assert out.startswith("bg_")


@pytest.mark.asyncio
async def test_background_result_reports_the_output(manager):
    handle = await background('Echo("hi")', 5000)
    out = await _wait_for_result(handle)
    assert "echo:hi" in out, f"the output must reach the collector: {out!r}"


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


def test_background_without_a_tool_manager_says_so(monkeypatch):
    """An unattached context is a wiring error answered in prose, not a crash."""
    monkeypatch.setattr(context_module, "_CURRENT", ToolContext())
    assert _call(background('Echo("hi")', 5000)).startswith("Error:")


@pytest.mark.asyncio
async def test_background_result_marks_a_failed_call(manager):
    handle = await background("Boom()", 5000)
    out = await _wait_for_result(handle)
    assert (
        "Error" in out or "failed" in out.lower()
    ), f"the failure must be reported, not lost: {out!r}"


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
async def test_background_passes_the_resolved_timeout_to_the_registry(manager):
    """background() forwards the resolved ceiling to the registry it launches into.

    The registry cannot prove this wiring on its own: a regression that
    launched with the raw model timeout, or skipped the clamp, would leave a
    direct ``registry.launch`` test untouched.  Drive the tool itself and
    read the record it created -- once with an ordinary ceiling, which must
    arrive as asked, and once above MAX_TIMEOUT_MS, which must arrive
    clamped with the reply saying so.
    """
    reply = await background('Echo("hi")', 12345)
    record = await _collect_terminal(reply.split(":", 1)[0])
    assert (
        record.state is TaskState.DONE
    ), "the stub tool finishes immediately; the record must settle DONE"
    assert (
        record.timeout == 12345
    ), f"the requested ceiling must reach the registry as-is, got {record.timeout!r}"

    reply = await background('Echo("hi")', 10_000_000)
    assert (
        "clamped" in reply.lower()
    ), f"the reply must say the ceiling was clamped: {reply!r}"
    clamped_record = await _collect_terminal(reply.split(":", 1)[0])
    assert clamped_record.timeout == MAX_TIMEOUT_MS, (
        f"a ceiling above the cap must arrive clamped, got "
        f"{clamped_record.timeout!r}"
    )
