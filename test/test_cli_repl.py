"""Tests for the interactive CLI REPL: turns, signals and exit.

Every test drives :class:`~mitocode.cli.repl.Repl` with a real
:class:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator` and a scripted
provider, and with a recording UI fake.  The event pump is real as well, because
the contract under test includes "the turn summary is printed only after every
queued event has been rendered".  No test starts a terminal, raises a real
signal, or needs a network connection: ``Ctrl+C`` is simulated by calling
:meth:`Repl.handle_sigint` or by having the fake prompt raise
``KeyboardInterrupt``, exactly as prompt_toolkit does.
"""

import asyncio
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli import repl as repl_module
from mitocode.cli.args import CliArgs
from mitocode.cli.pump import EventPump
from mitocode.cli.repl import CTRL_C_HINT, CTRL_C_WINDOW, Repl
from mitocode.cli.state import CliState
from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace


def _chunk(content=None, tool_calls=None, finish_reason=None):
    """Build one OpenAI-shaped streaming chunk.

    Args:
        content: Streamed text delta, or ``None``.
        tool_calls: Streamed tool-call deltas, or ``None``.
        finish_reason: Terminal reason, or ``None`` for a partial chunk.

    Returns:
        A ``SimpleNamespace`` shaped like an OpenAI stream chunk.
    """
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _tool_call(index, call_id, name, arguments):
    """Build one streamed tool-call delta.

    Args:
        index: Position of the call in the assistant message.
        call_id: Provider-assigned call id.
        name: Tool name; empty for argument-only chunks.
        arguments: Raw JSON argument fragment.

    Returns:
        A ``SimpleNamespace`` shaped like an OpenAI tool-call delta.
    """
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _text_turn(text, size=4):
    """Build a scripted turn that streams *text* and then stops.

    Args:
        text: Assistant prose to stream.
        size: Characters per chunk.

    Returns:
        A list of chunks ending with ``finish_reason="stop"``.
    """
    chunks = [
        _chunk(content=text[index : index + size])
        for index in range(0, len(text), size)
    ]
    chunks.append(_chunk(finish_reason="stop"))
    return chunks


def _tool_turn(text, call_id, name, arguments, size=4):
    """Build a scripted turn that streams *text* and then requests a tool.

    Args:
        text: Assistant prose to stream before the call.
        call_id: Provider-assigned call id.
        name: Tool to invoke.
        arguments: JSON argument string.
        size: Characters per argument chunk.

    Returns:
        A list of chunks ending with ``finish_reason="tool_calls"``.
    """
    chunks = [
        _chunk(content=text[index : index + size])
        for index in range(0, len(text), size)
    ]
    chunks.append(_chunk(tool_calls=[_tool_call(0, call_id, name, "")]))
    for index in range(0, len(arguments), size):
        fragment = arguments[index : index + size]
        chunks.append(_chunk(tool_calls=[_tool_call(0, call_id, None, fragment)]))
    chunks.append(_chunk(finish_reason="tool_calls"))
    return chunks


class _ScriptedProvider(_SampleChatProvider):
    """Provider that replays a list of scripted streaming turns.

    The last turn is replayed forever, so a test may script one closing turn and
    still run more than two turns without exhausting the script.

    Attributes:
        turns: Chunks of each turn, consumed in order.
        turn: Index of the next turn to replay.
    """

    def __init__(self, turns):
        """Store the scripted turns.

        Args:
            turns: One list of chunks per ``chat`` call.
        """
        self.stream = True
        self.turns = list(turns)
        self.turn = 0

    async def init(self):
        """Satisfy the provider lifecycle without doing anything."""

    @staticmethod
    async def get_model_list():
        """Return the model identifiers this provider claims.

        Returns:
            A single scripted model name.
        """
        return ["scripted-model"]

    @classmethod
    def get_settings_schema(cls):
        """Return an empty settings schema.

        Returns:
            A permissive JSON Schema fragment.
        """
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @classmethod
    def check_settings_schema(cls, settings):
        """Accept any settings mapping.

        Args:
            settings: Candidate settings.

        Returns:
            Always ``True``.
        """
        return True

    async def chat(self, model="", tools=None, messages=None, stream=True):
        """Replay the next scripted turn.

        Args:
            model: Ignored model name.
            tools: Ignored tool schemas.
            messages: Ignored conversation.
            stream: Ignored streaming flag.

        Returns:
            An async generator yielding the scripted chunks.
        """
        turn = self.turns[min(self.turn, len(self.turns) - 1)]
        self.turn += 1

        async def _stream():
            """Yield the scripted chunks, yielding control between each."""
            for chunk in turn:
                yield chunk
                await asyncio.sleep(0)

        return _stream()


class _RecordingStream:
    """Stand-in for ``StreamingRenderer`` that remembers every call.

    Attributes:
        lines: Streamed deltas, in order.
        flushes: Number of ``flush`` calls.
        events: Ordered ``(kind, payload)`` records used to check rendering order.
    """

    def __init__(self):
        """Initialise empty recordings."""
        self.lines = []
        self.flushes = 0
        self.events = []

    def feed(self, delta):
        """Record one streamed delta.

        Args:
            delta: Text fed by the pump.
        """
        self.lines.append(delta)
        self.events.append(("token", delta))

    def flush(self):
        """Record one flush of the partial line."""
        self.flushes += 1
        self.events.append(("flush", ""))


class RecordingUI:
    """UI fake replaying answers and recording every rendered line.

    Answers may be exception instances, which are raised instead of returned,
    because the REPL is the component that decides what ``EOFError`` and
    ``KeyboardInterrupt`` mean.

    Attributes:
        answers: Remaining answers, popped from the front.
        stream: Recording stream renderer.
        notes: Messages passed to ``note``.
        errors: Messages passed to ``error``.
        tool_lines: Rendered tool header lines.
        tool_results: Rendered tool result lines.
        finish_lines: Formatted turn summaries.
        finish_calls: Raw ``(seconds, tokens, tool_calls)`` tuples.
        prompt_active: Always ``False`` between calls.
        verbose: Verbose flag read by the pump.
        agent_count: Agent count read by the pump.
        events: Ordered render records, shared with the stream.
    """

    def __init__(self, answers=()):
        """Initialise the fake with scripted answers.

        Args:
            answers: Values or exceptions returned/raised by ``read_prompt``.
        """
        self.answers = list(answers)
        self.stream = _RecordingStream()
        self.notes = []
        self.errors = []
        self.tool_lines = []
        self.tool_results = []
        self.finish_lines = []
        self.finish_calls = []
        self.prompt_active = False
        self.verbose = False
        self.agent_count = 1
        self.events = self.stream.events
        self.thinking_starts = 0
        self.thinking_stops = 0

    async def read_prompt(self):
        """Return the next scripted answer.

        Returns:
            The next answer string.

        Raises:
            EOFError: When the script is exhausted or holds an exception.
            KeyboardInterrupt: When the script holds one.
        """
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def ask(self, message):
        """Return the next scripted answer, ignoring the message.

        Args:
            message: Prompt text.

        Returns:
            The next answer string.
        """
        return await self.read_prompt()

    def note(self, text):
        """Record a muted note.

        Args:
            text: Note body.
        """
        self.notes.append(text)

    def warn(self, text):
        """Record a warning.

        Args:
            text: Warning body.
        """
        self.notes.append(text)

    def error(self, text):
        """Record an error line.

        Args:
            text: Error body.
        """
        self.errors.append(text)

    def begin_tool(self, call, prefix=""):
        """Record a tool header line.

        Args:
            call: Rendered header text.
            prefix: Muted background-agent prefix.
        """
        self.tool_lines.append(f"{prefix}{call.plain}")
        self.events.append(("tool_start", call.plain))

    def finish_tool(self, result, prefix=""):
        """Record a tool result line.

        Args:
            result: Rendered result text.
            prefix: Muted background-agent prefix.
        """
        self.tool_results.append(f"{prefix}{result.plain}")
        self.events.append(("tool_result", result.plain))

    def start_thinking(self):
        """Record that the thinking spinner was requested."""
        self.thinking_starts += 1

    def stop_thinking(self):
        """Record that the thinking spinner was stopped."""
        self.thinking_stops += 1

    def feed_reasoning(self, delta):
        """Record a reasoning delta fed by the pump.

        Args:
            delta: Reasoning text handed to the UI.
        """
        self.events.append(("reasoning_feed", delta))

    def end_reasoning(self):
        """Record that the reasoning region was collapsed."""
        self.events.append(("reasoning_end", ""))

    def pause_live(self):
        """Satisfy the approval contract without rendering anything."""

    def resume_live(self):
        """Satisfy the approval contract without rendering anything."""

    def finish_turn(self, seconds, tokens, tool_calls):
        """Record one end-of-turn summary.

        Args:
            seconds: Turn duration in seconds.
            tokens: Estimated tokens used by the turn.
            tool_calls: Main-agent tool calls in the turn.
        """
        self.finish_calls.append((seconds, tokens, tool_calls))
        self.finish_lines.append(
            f"Done in {seconds}s · {tokens} tokens · {tool_calls} tool calls"
        )
        self.events.append(("finish", len(self.finish_lines)))


def _build_state(tmp_path, provider, *, initial_prompt=None):
    """Build a real ``CliState`` wired to a real orchestrator and workspace.

    Args:
        tmp_path: Workspace root.
        provider: Scripted provider every session snapshots.
        initial_prompt: Optional ``--initial-prompt`` value.

    Returns:
        The state, whose main session is created by the caller.
    """
    tool_manager = ToolManager()
    orchestrator = AgentOrchestrator(
        tool_manager=tool_manager, default_provider=provider
    )
    workspace = Workspace(tmp_path, "test-workspace", 1)
    state = CliState(
        args=CliArgs(workspace=tmp_path, initial_prompt=initial_prompt),
        workspace=workspace,
        tool_manager=tool_manager,
        provider=provider,
        orchestrator=orchestrator,
    )
    return state


def _register_read_tool(tool_manager):
    """Register a ``Read`` tool that returns the file's text.

    Args:
        tool_manager: Registry the tool is added to.
    """

    async def Read(file_path: str):
        """Read a text file from disk.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            The file's contents.
        """
        return Path(file_path).read_text(encoding="utf-8")

    tool_manager.register(Read)


async def _create_main(state, name="Main"):
    """Create the interactive main session and point ``state`` at it.

    Args:
        state: State whose ``main_id`` is updated.
        name: Display name of the session.

    Returns:
        The created session.
    """
    session = await state.orchestrator.create_session(name=name)
    state.main_id = session.agent_id
    return session


async def _stop(pump_task):
    """Cancel the pump task and consume its cancellation.

    Args:
        pump_task: Task returned by ``asyncio.create_task(pump.run())``.
    """
    pump_task.cancel()
    await asyncio.gather(pump_task, return_exceptions=True)


async def _setup(tmp_path, ui, turns, *, initial_prompt=None, tools=None):
    """Build state, pump and REPL for one test.

    Args:
        tmp_path: Workspace root.
        ui: Recording UI fake that will back the pump.
        turns: Scripted provider turns.
        initial_prompt: Optional ``--initial-prompt`` value.
        tools: Optional callable registering extra tools.

    Returns:
        ``(state, repl, pump, pump_task, session)``.
    """
    provider = _ScriptedProvider(turns)
    state = _build_state(tmp_path, provider, initial_prompt=initial_prompt)
    if tools is not None:
        tools(state.tool_manager)
    session = await _create_main(state)
    pump = EventPump(state.orchestrator.event_queue, ui, state)
    pump_task = asyncio.create_task(pump.run())
    repl = Repl(state, ui, pump, None)
    return state, repl, pump, pump_task, session


@contextmanager
def _recording_patch_stdout(**kwargs):
    """Stand-in for ``prompt_toolkit.patch_stdout`` that records its use.

    Args:
        **kwargs: Keyword arguments the REPL passed to ``patch_stdout``.
    """
    _recording_patch_stdout.calls += 1
    _recording_patch_stdout.kwargs = kwargs
    yield


_recording_patch_stdout.calls = 0
_recording_patch_stdout.kwargs = {}


@pytest.mark.asyncio
async def test_one_turn_renders_text_tool_and_summary(tmp_path):
    """A turn streams prose, runs a tool and ends with exactly one summary."""
    target = tmp_path / "notes.txt"
    target.write_text("hello from file\n", encoding="utf-8")

    ui = RecordingUI(answers=["say hi", EOFError()])
    turns = [
        _tool_turn("hello there", "call_read", "Read", '{"file_path": "%s"}' % target),
        _text_turn("all done"),
    ]

    def _tools(tool_manager):
        """Register the ``Read`` tool used by the scripted turn."""
        _register_read_tool(tool_manager)

    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, turns, tools=_tools)
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert "hello" in "".join(ui.stream.lines)
    assert any("Read" in line for line in ui.tool_lines)
    assert len(ui.finish_lines) == 1
    assert "Done in" in ui.finish_lines[-1]
    assert ui.finish_calls == [(ui.finish_calls[0][0], ui.finish_calls[0][1], 1)]
    assert state.turn_count == 1


@pytest.mark.asyncio
async def test_summary_is_printed_after_the_last_event_is_rendered(tmp_path):
    """``finish_turn`` runs only after ``pump.wait_idle`` drained the queue."""
    target = tmp_path / "notes.txt"
    target.write_text("hello from file\n", encoding="utf-8")

    ui = RecordingUI(answers=["say hi", EOFError()])
    turns = [
        _tool_turn("hello there", "call_read", "Read", '{"file_path": "%s"}' % target),
        _text_turn("all done"),
    ]

    def _tools(tool_manager):
        """Register the ``Read`` tool used by the scripted turn."""
        _register_read_tool(tool_manager)

    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, turns, tools=_tools)
    try:
        await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    kinds = [kind for kind, _ in ui.events]
    assert "tool_result" in kinds
    assert kinds[-1] == "finish"
    assert ui.events[-2][0] in ("flush", "token", "tool_result")
    assert ui.stream.flushes >= 1


@pytest.mark.asyncio
async def test_sigint_cancels_the_turn_and_keeps_the_session_usable(tmp_path):
    """A Ctrl+C during a turn cancels it without destroying the session."""
    tool_started = asyncio.Event()

    async def Slow():
        """Block until the test cancels the turn."""
        tool_started.set()
        await asyncio.sleep(30)
        return "never"

    ui = RecordingUI(answers=[])
    turns = [
        _tool_turn("working", "call_slow", "Slow", "{}"),
        _text_turn("resumed"),
    ]

    def _tools(tool_manager):
        """Register the blocking tool used by the scripted turn."""
        tool_manager.register(Slow)

    state, repl, _, pump_task, session = await _setup(tmp_path, ui, turns, tools=_tools)
    try:
        turn = asyncio.create_task(repl.run_turn("go"))
        await asyncio.wait_for(tool_started.wait(), timeout=2)
        repl.handle_sigint()
        await asyncio.wait_for(turn, timeout=3)

        assert session.state.value == "cancelled"
        assert len(ui.finish_lines) == 1
        assert repl._active_turn is None  # pylint: disable=protected-access

        await asyncio.wait_for(repl.run_turn("again"), timeout=5)
        assert session.state.value == "finished"
        assert "resumed" in "".join(ui.stream.lines)
        assert len(ui.finish_lines) == 2
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_prompt_ctrl_c_twice_exits_and_hints_first(tmp_path):
    """A first Ctrl+C at the prompt only prints the hint; the second exits."""
    ui = RecordingUI(answers=[KeyboardInterrupt(), KeyboardInterrupt()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert ui.notes == [CTRL_C_HINT]


@pytest.mark.asyncio
async def test_eof_exits_cleanly(tmp_path):
    """Ctrl+D ends the process without running a turn."""
    ui = RecordingUI(answers=[EOFError()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert ui.finish_lines == []


@pytest.mark.asyncio
async def test_slash_exit_ends_the_loop(tmp_path):
    """``/exit`` ends the loop through the command dispatcher."""
    ui = RecordingUI(answers=["/exit", EOFError()])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert state.turn_count == 0
    assert ui.finish_lines == []


@pytest.mark.asyncio
async def test_initial_prompt_runs_one_turn_before_reading_input(tmp_path):
    """``--initial-prompt`` submits one message before the first prompt."""
    turns = [_text_turn("hello from initial"), _text_turn("hello from later")]
    ui = RecordingUI(answers=["later", EOFError()])
    state, repl, _, pump_task, _ = await _setup(
        tmp_path, ui, turns, initial_prompt="start"
    )
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert len(ui.finish_lines) == 2
    assert state.turn_count == 2
    assert "hello from initial" in "".join(ui.stream.lines)


@pytest.mark.asyncio
async def test_failing_initial_prompt_does_not_end_the_process(tmp_path, monkeypatch):
    """A raised ``--initial-prompt`` turn is reported and the prompt still comes back."""
    ui = RecordingUI(answers=["after failure", EOFError()])
    _, repl, _, pump_task, _ = await _setup(
        tmp_path, ui, [_text_turn("recovered")], initial_prompt="boom"
    )

    real_run_turn = repl.run_turn
    submitted = []

    async def _explode_on_initial(text):
        """Raise on the initial prompt and delegate every later turn.

        Args:
            text: User message submitted for the turn.

        Raises:
            RuntimeError: On the first, initial-prompt call.
        """
        submitted.append(text)
        if len(submitted) == 1:
            raise RuntimeError("initial turn exploded")
        await real_run_turn(text)

    monkeypatch.setattr(repl, "run_turn", _explode_on_initial)
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert any("Turn failed: initial turn exploded" in line for line in ui.errors)
    assert submitted == ["boom", "after failure"]
    assert "recovered" in "".join(ui.stream.lines)


@pytest.mark.asyncio
async def test_initial_prompt_slash_prefix_stays_a_turn(tmp_path, monkeypatch):
    """A ``--initial-prompt`` starting with "/" is a turn, not a slash command."""
    ui = RecordingUI(answers=[EOFError()])
    state, repl, _, pump_task, _ = await _setup(
        tmp_path, ui, [_text_turn("ran as a turn")], initial_prompt="/help"
    )

    dispatch_calls = []

    async def _forbid_dispatch(text, ctx):
        """Record any dispatch call; the loop must never route the prompt here.

        Args:
            text: The submitted line that would be dispatched.
            ctx: The command context that would back the dispatch.

        Returns:
            A default result, unreachable in a passing run.
        """
        dispatch_calls.append(text)
        from mitocode.cli.commands import CommandResult

        return CommandResult()

    monkeypatch.setattr(repl_module, "dispatch", _forbid_dispatch)
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert dispatch_calls == []
    assert ui.errors == []
    assert state.turn_count == 1
    assert "ran as a turn" in "".join(ui.stream.lines)


@pytest.mark.asyncio
async def test_blank_input_is_ignored(tmp_path):
    """Empty and whitespace-only lines never start a turn."""
    ui = RecordingUI(answers=["", "   ", EOFError()])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert state.turn_count == 0
    assert ui.finish_lines == []


@pytest.mark.asyncio
async def test_unknown_command_keeps_the_loop_alive(tmp_path):
    """An unknown slash command prints one error and the REPL keeps running."""
    ui = RecordingUI(answers=["/nope", "say hi", EOFError()])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("hello")])
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert any("Unknown command /nope" in line for line in ui.errors)
    assert state.turn_count == 1


@pytest.mark.asyncio
async def test_handle_sigint_without_a_turn_needs_two_presses(tmp_path):
    """A lone SIGINT at the prompt does not shut the process down."""
    ui = RecordingUI(answers=[EOFError()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    try:
        repl._last_ctrl_c_at = (
            time.monotonic() - CTRL_C_WINDOW - 1
        )  # pylint: disable=protected-access
        repl.handle_sigint()
        assert not repl._shutdown.is_set()  # pylint: disable=protected-access

        repl.handle_sigint()
        assert repl._shutdown.is_set()  # pylint: disable=protected-access
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_run_is_wrapped_in_patch_stdout(tmp_path, monkeypatch):
    """Background output is patched into the prompt for the whole run."""
    ui = RecordingUI(answers=[EOFError()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    _recording_patch_stdout.calls = 0
    monkeypatch.setattr(repl_module, "patch_stdout", _recording_patch_stdout)
    try:
        await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert _recording_patch_stdout.calls == 1


@pytest.mark.asyncio
async def test_run_requests_raw_patch_stdout(tmp_path, monkeypatch):
    """``raw=True`` keeps Rich's ANSI styling intact above the prompt.

    ``StdoutProxy(raw=False)`` strips VT100 sequences, which corrupted the
    transcript with literal ``ESC[...`` fragments; the REPL must opt out.
    """
    ui = RecordingUI(answers=[EOFError()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    _recording_patch_stdout.calls = 0
    _recording_patch_stdout.kwargs = {}
    monkeypatch.setattr(repl_module, "patch_stdout", _recording_patch_stdout)
    try:
        await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert _recording_patch_stdout.calls == 1
    assert _recording_patch_stdout.kwargs.get("raw") is True


@pytest.mark.asyncio
async def test_run_turn_reports_token_and_tool_call_deltas(tmp_path):
    """The summary counts only what this turn produced."""
    target = tmp_path / "notes.txt"
    target.write_text("hello from file\n", encoding="utf-8")

    ui = RecordingUI(answers=[])
    turns = [
        _tool_turn("hello there", "call_read", "Read", '{"file_path": "%s"}' % target),
        _text_turn("all done"),
    ]

    def _tools(tool_manager):
        """Register the ``Read`` tool used by the scripted turn."""
        _register_read_tool(tool_manager)

    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, turns, tools=_tools)
    try:
        await asyncio.wait_for(repl.run_turn("say hi"), timeout=5)
        assert ui.finish_calls[-1][2] == 1
        assert ui.finish_calls[-1][1] > 0
        assert state.tool_call_count == 1
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_missing_main_session_still_finishes_the_turn(tmp_path):
    """A vanished main session reports an error but never loses the summary."""
    ui = RecordingUI(answers=[])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])
    try:
        await state.orchestrator.destroy_agent(state.main_id)
        await asyncio.wait_for(repl.run_turn("hello"), timeout=5)

        assert any(
            f"Main agent #{state.main_id} does not exist" in line for line in ui.errors
        )
        assert len(ui.finish_lines) == 1
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_clear_rebinds_the_turn_to_the_new_main_session(tmp_path):
    """``/clear`` replaces the session and the next turn uses the new id."""
    ui = RecordingUI(answers=["/clear", "hello", EOFError()])
    state, repl, _, pump_task, first = await _setup(
        tmp_path, ui, [_text_turn("hello after clear")]
    )
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert state.main_id != first.agent_id
    assert state.turn_count == 1
    assert "hello after clear" in "".join(ui.stream.lines)
    assert any("New session" in line for line in ui.notes)


@pytest.mark.asyncio
async def test_external_cancellation_is_not_swallowed(tmp_path):
    """Cancelling ``run_turn`` from outside propagates after the summary."""
    tool_started = asyncio.Event()

    async def Slow():
        """Block until the test cancels the turn."""
        tool_started.set()
        await asyncio.sleep(30)
        return "never"

    ui = RecordingUI(answers=[])
    turns = [_tool_turn("working", "call_slow", "Slow", "{}")]

    def _tools(tool_manager):
        """Register the blocking tool used by the scripted turn."""
        tool_manager.register(Slow)

    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, turns, tools=_tools)
    try:
        turn = asyncio.create_task(repl.run_turn("go"))
        await asyncio.wait_for(tool_started.wait(), timeout=2)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, timeout=3)
        assert len(ui.finish_lines) == 1
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_send_message_returning_none_stops_the_spinner(tmp_path, monkeypatch):
    """A vanished agent still stops thinking and prints one summary."""
    ui = RecordingUI(answers=[])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])

    async def _send_nothing(agent_id, message):
        """Stand in for an orchestrator whose session disappeared.

        Args:
            agent_id: Ignored agent id.
            message: Ignored user message.

        Returns:
            ``None``, as ``send_message`` does for a missing session.
        """
        del agent_id, message
        return None

    monkeypatch.setattr(state.orchestrator, "send_message", _send_nothing)
    try:
        await asyncio.wait_for(repl.run_turn("hello"), timeout=5)
    finally:
        await _stop(pump_task)

    assert any("does not exist" in line for line in ui.errors)
    assert ui.thinking_starts == 1
    assert ui.thinking_stops == 1
    assert len(ui.finish_lines) == 1


@pytest.mark.asyncio
async def test_failing_turn_does_not_escape_the_loop(tmp_path, monkeypatch):
    """A turn that raises is reported and the prompt comes back."""
    ui = RecordingUI(answers=["boom", EOFError()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])

    async def _explode(text):
        """Raise the failure the loop must contain.

        Args:
            text: Ignored user message.

        Raises:
            RuntimeError: Always.
        """
        del text
        raise RuntimeError("turn exploded")

    monkeypatch.setattr(repl, "run_turn", _explode)
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert any("Turn failed: turn exploded" in line for line in ui.errors)


@pytest.mark.asyncio
async def test_failing_command_does_not_escape_the_loop(tmp_path, monkeypatch):
    """A raising command handler is reported and the prompt comes back."""
    ui = RecordingUI(answers=["/help", EOFError()])
    _, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("unused")])

    async def _explode(line, ctx):
        """Raise the failure the loop must contain.

        Args:
            line: Ignored command line.
            ctx: Ignored command context.

        Raises:
            RuntimeError: Always.
        """
        del line, ctx
        raise RuntimeError("command exploded")

    monkeypatch.setattr(repl_module, "dispatch", _explode)
    try:
        code = await asyncio.wait_for(repl.run(), timeout=5)
    finally:
        await _stop(pump_task)

    assert code == 0
    assert any("Command failed: command exploded" in line for line in ui.errors)


# ---------------------------------------------------------------------------
# End-of-turn session save and title trigger
# ---------------------------------------------------------------------------


class _FakeSessionManager:
    """Recording stand-in for the manager the REPL saves through.

    Attributes:
        saves: Number of ``save_current`` calls.
        titles: Number of ``maybe_generate_title`` calls.
        save_error: Exception ``save_current`` raises when scripted.
    """

    def __init__(self):
        self.saves = 0
        self.titles = 0
        self.save_error = None

    async def save_current(self, *, force=False, report_failure=False):
        """Count one save request, raising the scripted error when set.

        Args:
            force: Ignored force flag.
            report_failure: Ignored failure-reporting flag.

        Raises:
            RuntimeError: When ``save_error`` is set.
        """
        self.saves += 1
        if self.save_error is not None:
            raise self.save_error

    def maybe_generate_title(self):
        """Count one title trigger."""
        self.titles += 1


@pytest.mark.asyncio
async def test_turn_end_saves_the_session_and_triggers_the_title(tmp_path):
    """After every turn the manager saved once and the title hook ran."""
    ui = RecordingUI(answers=[])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("hello")])
    manager = _FakeSessionManager()
    state.session_manager = manager
    try:
        await asyncio.wait_for(repl.run_turn("say hi"), timeout=5)

        assert manager.saves == 1
        assert manager.titles == 1
        assert state.turn_count == 1
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_turn_end_save_error_is_reported_not_fatal(tmp_path):
    """A raising manager save is contained by the turn-failure reporter."""
    ui = RecordingUI(answers=[])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("hello")])
    manager = _FakeSessionManager()
    manager.save_error = RuntimeError("disk exploded")
    state.session_manager = manager
    try:
        await asyncio.wait_for(repl._run_turn_reported("say hi"), timeout=5)

        assert any("Turn failed" in line for line in ui.errors)
        assert state.turn_count == 1
    finally:
        await _stop(pump_task)


@pytest.mark.asyncio
async def test_turn_end_without_a_manager_stays_quiet(tmp_path):
    """No manager wired means no save or title calls and a normal turn."""
    ui = RecordingUI(answers=[])
    state, repl, _, pump_task, _ = await _setup(tmp_path, ui, [_text_turn("hello")])
    try:
        await asyncio.wait_for(repl.run_turn("say hi"), timeout=5)

        assert state.session_manager is None
        assert len(ui.finish_lines) == 1
    finally:
        await _stop(pump_task)
