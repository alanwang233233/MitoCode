"""Tests for the CLI event pump that renders orchestrator events.

Every test drives :class:`~mitocode.cli.pump.EventPump` with an
``asyncio.Queue`` filled by hand and a recording UI fake, so no test needs a
real orchestrator, a real provider or a real terminal. The queue is the only
input: the pump is the single consumer, and the tests pin down which events are
rendered, how tool results are matched to their calls, and when
``wait_idle()`` is allowed to return.
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.pump import EventPump


class _RecordingStream:
    """Stand-in for ``StreamingRenderer`` that remembers every call.

    Attributes:
        feeds: Deltas passed to ``feed``, in order.
        flushes: Number of ``flush`` calls.
    """

    def __init__(self) -> None:
        """Initialise empty recordings."""
        self.feeds = []
        self.flushes = 0

    def feed(self, delta: str) -> None:
        """Record one streamed delta.

        Args:
            delta: Text fed by the pump.
        """
        self.feeds.append(delta)

    def flush(self) -> None:
        """Record one flush of the partial line."""
        self.flushes += 1


class _RecordingUI:
    """Minimal UI fake recording everything the pump renders.

    Attributes:
        stream: Recording stream renderer.
        tool_starts: One record per ``begin_tool`` call.
        tool_results: One record per ``finish_tool`` call.
        notes: Messages passed to ``note``, in order.
        errors: Messages passed to ``error``, in order.
        prompt_active: Always false; the pump never asks for input.
        verbose: Whether full tool output was requested.
        agent_count: Number of agents shown in the toolbar.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        """Initialise the fake with empty recordings.

        Args:
            verbose: Value of the UI's verbose flag.
        """
        self.stream = _RecordingStream()
        self.tool_starts = []
        self.tool_results = []
        self.notes = []
        self.errors = []
        self.prompt_active = False
        self.verbose = verbose
        self.agent_count = 1

    def begin_tool(self, call: Text, prefix: str = "") -> None:
        """Record a tool header line.

        Args:
            call: Header text built by ``format_tool_call``.
            prefix: Muted prefix for a background agent.
        """
        self.tool_starts.append(_record(call, prefix))

    def finish_tool(self, result: Text, prefix: str = "") -> None:
        """Record a tool result line.

        Args:
            result: Preview text built by ``format_result``.
            prefix: Muted prefix for a background agent.
        """
        self.tool_results.append(_record(result, prefix))

    def note(self, text: str) -> None:
        """Record a muted note.

        Args:
            text: Note body.
        """
        self.notes.append(text)

    def error(self, text: str) -> None:
        """Record an error line.

        Args:
            text: Error body.
        """
        self.errors.append(text)


class _BrokenNoteUI(_RecordingUI):
    """UI fake whose ``note`` channel always fails."""

    def note(self, text: str) -> None:
        """Fail every note, standing in for a broken renderer.

        Args:
            text: Note the pump tried to print.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("cannot print")


def _record(text: Text, prefix: str) -> SimpleNamespace:
    """Capture one rendered line as plain text plus its prefix.

    Args:
        text: Rendered text handed to the fake UI.
        prefix: Muted prefix the pump supplied.

    Returns:
        A namespace with ``plain`` (prefix plus text), ``text`` and ``prefix``.
    """
    return SimpleNamespace(plain=f"{prefix}{text.plain}", text=text, prefix=prefix)


def _state(workspace: Path, *, main_id: int = 0, verbose: bool = False):
    """Build the slice of ``CliState`` the pump reads.

    Args:
        workspace: Workspace root used to relativise displayed paths.
        main_id: Agent id of the interactive main agent.
        verbose: Whether full tool output should be rendered.

    Returns:
        A namespace standing in for :class:`~mitocode.cli.state.CliState`.
    """
    return SimpleNamespace(
        main_id=main_id,
        agents=[
            {"id": 0, "name": "Main", "state": "idle"},
            {"id": 3, "name": "explore", "state": "running"},
        ],
        tool_call_count=0,
        total_chars=0,
        verbose=verbose,
        workspace=SimpleNamespace(path=workspace),
    )


async def _stop(pump_task: "asyncio.Task[None]") -> None:
    """Cancel the pump task and consume its cancellation.

    Args:
        pump_task: The task returned by ``asyncio.create_task(pump.run())``.
    """
    pump_task.cancel()
    await asyncio.gather(pump_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pump_renders_main_tool_cycle_and_waits_idle(tmp_path):
    """A main-agent tool cycle renders and ``wait_idle`` covers every event."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path, main_id=0)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait((0, {"type": "token", "content": "hi\n"}))
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Read",
                "args": {"file_path": "a.py"},
                "display_param": "file_path",
            },
        )
    )
    # ``Read`` results collapse to a line count, so twelve numbered output
    # lines must render as "12 lines".
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_result",
                "id": "c1",
                "success": True,
                "result": "\n".join(f"{index}: value" for index in range(1, 13)),
            },
        )
    )
    queue.put_nowait((0, {"type": "turn_end", "reason": "stop"}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert state.tool_call_count == 1
    assert ui.stream.feeds == ["hi\n"]
    assert ui.stream.flushes == 1
    assert ui.tool_starts[0].plain == "Read  a.py"
    assert ui.tool_results[-1].plain == "12 lines"
    await _stop(task)


@pytest.mark.asyncio
async def test_wait_idle_waits_until_every_queued_event_is_rendered(tmp_path):
    """``wait_idle`` cannot return while the queue still holds unread events."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    # Events are enqueued before the pump task is ever scheduled, which is
    # exactly when a naive "is the queue empty?" check could return too early.
    queue.put_nowait((0, {"type": "token", "content": "hello\n"}))
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Read",
                "args": {"file_path": "a.py"},
                "display_param": "file_path",
            },
        )
    )
    queue.put_nowait(
        (0, {"type": "tool_result", "id": "c1", "success": True, "result": "only line"})
    )
    queue.put_nowait((0, {"type": "turn_end", "reason": "stop"}))

    # The waiter is scheduled *before* the pump: it must park on the condition
    # instead of returning because the queue still has four unread events.
    idle = asyncio.create_task(pump.wait_idle())
    await asyncio.sleep(0)
    assert not idle.done()

    task = asyncio.create_task(pump.run())
    await asyncio.wait_for(idle, timeout=2)

    assert ui.stream.feeds == ["hello\n"]
    assert ui.stream.flushes == 1
    assert len(ui.tool_starts) == 1
    assert len(ui.tool_results) == 1
    assert state.total_chars == len("hello\n")
    await _stop(task)


@pytest.mark.asyncio
async def test_pump_does_not_stream_background_text(tmp_path):
    """A background agent's prose never reaches the main transcript."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait((3, {"type": "token", "content": "sub\n"}))
    queue.put_nowait((3, {"type": "turn_end", "reason": "stop"}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert "sub" not in "".join(ui.stream.feeds)
    assert ui.stream.feeds == []
    assert ui.notes == ["↳ explore finished"]
    assert state.total_chars == 0
    await _stop(task)


@pytest.mark.asyncio
async def test_background_tool_cycle_is_rendered_with_agent_prefix(tmp_path):
    """A background tool call is rendered dim-prefixed with the agent's name."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait(
        (
            3,
            {
                "type": "tool_start",
                "id": "c9",
                "name": "Grep",
                "args": {"pattern": "needle", "path": str(tmp_path)},
                "display_param": "pattern",
            },
        )
    )
    queue.put_nowait(
        (3, {"type": "tool_result", "id": "c9", "success": True, "result": "2 hits"})
    )

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert ui.tool_starts[0].plain == "↳ explore Grep  needle in ."
    assert ui.tool_results[0].plain == "↳ explore 2 hits"
    assert state.tool_call_count == 0
    await _stop(task)


@pytest.mark.asyncio
async def test_tool_results_are_matched_by_agent_and_call_id(tmp_path):
    """Two agents reusing a call id never see each other's arguments."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Edit",
                "args": {"file_path": str(tmp_path / "a.py"), "old_string": "x"},
                "display_param": "file_path",
            },
        )
    )
    queue.put_nowait(
        (
            3,
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Bash",
                "args": {"command": "ls"},
                "display_param": "command",
            },
        )
    )
    queue.put_nowait(
        (3, {"type": "tool_result", "id": "c1", "success": True, "result": "bg out"})
    )
    queue.put_nowait(
        (0, {"type": "tool_result", "id": "c1", "success": True, "result": "main out"})
    )

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert [record.prefix for record in ui.tool_starts] == ["", "↳ explore "]
    assert ui.tool_results[0].plain == "↳ explore bg out"
    # The main result was formatted with the Edit arguments, not Bash's, so it
    # renders the diff summary instead of the raw output.
    assert "Updated a.py" in ui.tool_results[1].plain
    assert state.tool_call_count == 1
    await _stop(task)


@pytest.mark.asyncio
async def test_failed_tool_result_is_rendered_with_error_style(tmp_path):
    """A falsy ``success`` or an ``Error:`` body makes the result line red."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Bash",
                "args": {"command": "false"},
                "display_param": "command",
            },
        )
    )
    queue.put_nowait(
        (0, {"type": "tool_result", "id": "c1", "success": False, "result": "boom"})
    )
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c2",
                "name": "Bash",
                "args": {"command": "nope"},
                "display_param": "command",
            },
        )
    )
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_result",
                "id": "c2",
                "success": True,
                "result": "Error: no such file",
            },
        )
    )
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c3",
                "name": "Bash",
                "args": {"command": "true"},
                "display_param": "command",
            },
        )
    )
    queue.put_nowait(
        (0, {"type": "tool_result", "id": "c3", "success": True, "result": "ok"})
    )

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert [record.text.style for record in ui.tool_results[:2]] == [
        "mito.error",
        "mito.error",
    ]
    assert ui.tool_results[2].text.style in ("", None)
    assert ui.tool_results[0].plain == "boom"
    await _stop(task)


@pytest.mark.asyncio
async def test_sync_event_updates_agents_without_rendering(tmp_path):
    """A ``sync`` event replaces the agent table silently."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    state.agents = []
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    snapshot = [{"id": 0, "name": "Main"}, {"id": 4, "name": "tester"}]
    queue.put_nowait((-1, {"type": "sync", "agents": snapshot}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert state.agents == snapshot
    assert ui.agent_count == 2
    assert ui.notes == [] and ui.errors == []
    assert ui.tool_starts == [] and ui.tool_results == []
    await _stop(task)


@pytest.mark.asyncio
async def test_main_cancellation_flushes_and_notes_interrupted(tmp_path):
    """Cancelling the main turn flushes its partial line and says so."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait((0, {"type": "token", "content": "half a line"}))
    queue.put_nowait((0, {"type": "turn_cancelled"}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert ui.stream.feeds == ["half a line"]
    assert ui.stream.flushes == 1
    assert ui.notes == ["Interrupted."]
    await _stop(task)


@pytest.mark.asyncio
async def test_background_cancellation_and_error_print_muted_lines(tmp_path):
    """Background cancellation and failures become dim prefixed notes."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait((3, {"type": "turn_cancelled"}))
    queue.put_nowait((3, {"type": "error", "message": "provider exploded"}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert ui.notes == [
        "↳ explore cancelled",
        "↳ explore error: provider exploded",
    ]
    assert ui.errors == []
    await _stop(task)


@pytest.mark.asyncio
async def test_main_error_is_printed_through_the_error_surface(tmp_path):
    """A main-agent failure reaches the UI's error channel with its prefix."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait((0, {"type": "error", "message": "bad key"}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert ui.errors == ["Error: bad key"]
    assert ui.notes == []
    await _stop(task)


@pytest.mark.asyncio
async def test_unknown_event_types_and_missing_payloads_are_ignored(tmp_path):
    """The pump ignores unknown types and never dies on a sparse event."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait((0, {"type": "wat"}))
    queue.put_nowait((0, {"type": "token"}))
    queue.put_nowait((0, {"type": "tool_start", "id": "c5", "name": "Read"}))
    queue.put_nowait((0, {"type": "tool_result", "id": "ghost", "result": "orphan"}))
    queue.put_nowait((0, {"type": "tool_result", "id": "c5", "result": "5 lines"}))
    queue.put_nowait((0, {"type": "turn_end"}))

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    # The unknown type rendered nothing, the sparse tool_start still produced a
    # header, and the unmatched result fell back to an empty argument mapping.
    assert ui.stream.feeds == []
    assert len(ui.tool_starts) == 1
    assert ui.tool_starts[0].plain == "Read"
    assert len(ui.tool_results) == 2
    assert ui.tool_results[0].plain == "orphan"
    assert ui.tool_results[1].plain == "1 lines"
    assert ui.stream.flushes == 1
    await _stop(task)


@pytest.mark.asyncio
async def test_verbose_flag_controls_result_truncation(tmp_path):
    """The pump passes the state's verbose flag and workspace root through."""
    long_result = "\n".join(f"line {index}" for index in range(20))
    for verbose in (False, True):
        queue: asyncio.Queue = asyncio.Queue()
        ui = _RecordingUI(verbose=verbose)
        state = _state(tmp_path, verbose=verbose)
        pump = EventPump(queue, ui, state)
        task = asyncio.create_task(pump.run())
        queue.put_nowait(
            (
                0,
                {
                    "type": "tool_start",
                    "id": "c1",
                    "name": "Bash",
                    "args": {"command": "seq 20"},
                    "display_param": "command",
                },
            )
        )
        queue.put_nowait(
            (
                0,
                {
                    "type": "tool_result",
                    "id": "c1",
                    "success": True,
                    "result": long_result,
                },
            )
        )

        await asyncio.wait_for(pump.wait_idle(), timeout=2)

        shown = ui.tool_results[0].plain
        if verbose:
            assert "line 19" in shown
            assert "lines" not in shown
        else:
            assert "line 19" not in shown
            assert "… +12 lines" in shown
        await _stop(task)


@pytest.mark.asyncio
async def test_cancelled_turn_forgets_remembered_tool_calls(tmp_path):
    """A cancelled turn drops its unfinished calls and names unknown agents."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _RecordingUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    task = asyncio.create_task(pump.run())
    queue.put_nowait(
        (
            0,
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Edit",
                "args": {"file_path": "a.py", "old_string": "x"},
                "display_param": "file_path",
            },
        )
    )
    queue.put_nowait((0, {"type": "turn_cancelled"}))
    # Agent 7 is absent from the snapshot, so its lines fall back to its id;
    # agent 5 is present but has no name yet and falls back the same way.
    state.agents.append({"id": 5, "name": ""})
    queue.put_nowait((5, {"type": "turn_end"}))
    queue.put_nowait(
        (
            7,
            {
                "type": "tool_start",
                "id": "c2",
                "name": "Read",
                "args": {"file_path": "b.py"},
                "display_param": "file_path",
            },
        )
    )
    queue.put_nowait(
        (7, {"type": "tool_result", "id": "c2", "success": True, "result": "one line"})
    )

    await asyncio.wait_for(pump.wait_idle(), timeout=2)

    assert (0, "c1") not in pump._calls  # pylint: disable=protected-access
    assert ui.notes[0] == "Interrupted."
    assert ui.notes[1] == "↳ agent #5 finished"
    assert ui.tool_results[-1].prefix == "↳ agent #7 "
    await _stop(task)


@pytest.mark.asyncio
async def test_a_broken_handler_is_logged_and_the_pump_keeps_running(tmp_path):
    """One failing render is logged and skipped instead of killing the pump."""
    queue: asyncio.Queue = asyncio.Queue()
    ui = _BrokenNoteUI()
    state = _state(tmp_path)
    pump = EventPump(queue, ui, state)
    logged = []
    sink = loguru_logger.add(logged.append, level="ERROR")
    try:
        task = asyncio.create_task(pump.run())
        queue.put_nowait((3, {"type": "turn_end"}))
        queue.put_nowait((0, {"type": "token", "content": "still alive\n"}))

        await asyncio.wait_for(pump.wait_idle(), timeout=2)
    finally:
        loguru_logger.remove(sink)

    assert any("could not render" in line for line in logged)
    assert ui.stream.feeds == ["still alive\n"]
    await _stop(task)
