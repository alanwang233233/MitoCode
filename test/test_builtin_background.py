"""Unit tests for the detached-call registry and the delivery gate in front of it.

Two modules are covered here and the split is deliberate: ``background.py`` owns
*what happened to the work* and *who still owes a notification*, and
``delivery_gate.py`` owns the single question of *whether a prompt may be written
into a caller's session right now*.  The tests are grouped the same way, because
the two halves have genuinely different failure modes and a regression in either
looks nothing like the other.

Most of the weight sits on four defects that were found in review rather than by
a failing test, so they are the four tests that had to earn their place:

* work that suppresses its own ``CancelledError`` and returns must be recorded
  ``TIMEOUT``, never ``DONE`` — the failure the whole module exists to prevent;
* a tool raising its *own* ``TimeoutError`` must be ``FAILED``, not the
  registry's timeout — on 3.11+ ``asyncio.TimeoutError`` *is* the builtin, so
  the exception type cannot be the discriminator;
* a ``dispatch_pending`` cancelled while awaiting ``send_message`` must leave
  the handle pending again rather than claimed forever;
* two overlapping dispatches must not both deliver one handle.

Each of those was verified red against a reconstruction of the previous
behaviour, outside this repository, before being written down here.
"""

import asyncio
import inspect
import os
import sys
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.tool.builtin.background import (
    TERMINAL_STATES,
    BackgroundRegistry,
    BackgroundTask,
    TaskState,
)
from mitocode.core.tool.builtin.context import ToolContext
from mitocode.core.tool.builtin.delivery_gate import (
    DELIVERABLE_STATE_VALUES,
    PARKED_NO_SESSION,
    PARKED_RUNNING,
    PARKED_UNKNOWN_STATE,
    RUNNING_STATE_VALUE,
    park_reason,
)
from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError

# --- constants -------------------------------------------------------------
#
# Spelled out here rather than imported where the *value* is what is under test:
# a value copied out of the implementation passes on any implementation,
# including a wrong one.  Only the control constants are imported.

AGENT_ID = 1
OTHER_AGENT_ID = 2

#: A caller mid-turn, one in a state this build cannot read, and one whose
#: session has been removed: the three callers the gate parks for.
BUSY_AGENT_ID = 3
CONFUSED_AGENT_ID = 4
GONE_AGENT_ID = 5

WORK_RESULT = "12 passed in 3.21s"

#: First handle a fresh registry issues, and the shape of a delivered message.
EXPECTED_NOTIFICATION = f"Background task bg_1 finished (done):\n{WORK_RESULT}"

#: Five minutes, in milliseconds, spelled out rather than imported.
EXPECTED_DEFAULT_CEILING_MS = 300_000

#: How many failed sends a handle gets before it is retired from notification.
EXPECTED_RETRY_CAP = 5

#: Longest fragment of an unrecognised state value echoed into the drift log.
EXPECTED_TRUNCATION_CHARS = 60

#: A ceiling short enough to fire inside a test, in milliseconds.
SHORT_CEILING_MS = 20

#: A ceiling no test ever comes close to, so nothing expires by accident.
LONG_CEILING_MS = 60_000

#: How long detached work sleeps when a test only needs it to stay in flight.
#: Every test that uses this cancels or times the task out long before it ends.
LONG_SLEEP_SECONDS = 30

#: Waiting for a detached task to settle is a poll, never a fixed sleep.
POLL_DELAY_SECONDS = 0.005
POLL_ATTEMPTS = 400


# --- doubles ---------------------------------------------------------------


class _FakeSession:
    """Stand-in for an ``AgentSession``: the gate reads nothing else off it.

    The delivery gate compares ``session.state`` by its ``.value`` string and
    never imports the agent package, so a test double may supply a bare string,
    an enum member or nothing at all without this module knowing the difference.
    """

    def __init__(self, state: object = "idle") -> None:
        """Record *state* verbatim, including ``None`` and non-strings.

        Args:
            state: Whatever the double should report as ``session.state``.
        """
        self.state = state


class _FakeOrchestrator:
    """Records the deliveries the registry asks for, and can be told to fail.

    Mirrors only the two methods ``dispatch_pending`` calls — ``get_session`` and
    ``send_message`` — which is the whole surface ``AgentOrchestrator`` exposes
    to the tool package.  Hand-written rather than patched, so a test can hold a
    send open, fail it a scripted number of times, or flip a caller's state.
    """

    def __init__(self, sessions=None, *, failures=(), gate=None, entered=None) -> None:
        """Configure the double.

        Args:
            sessions: Ids mapped to the sessions ``get_session`` should return.
            failures: Exceptions raised by successive ``send_message`` calls,
                consumed in order; an empty script always succeeds.
            gate: Event awaited inside ``send_message``; ``None`` never blocks.
            entered: Event set as soon as a send reaches the gate.
        """
        self.sessions = dict(sessions or {})
        self.scripted_failures = list(failures)
        self.gate = gate
        self.entered = entered
        self.sent = []
        self.attempts = 0

    def get_session(self, agent_id: int):
        """Return the session registered under *agent_id*, or ``None``.

        Args:
            agent_id: The caller the registry wants to deliver into.

        Returns:
            Whatever this double was configured with for that id.
        """
        return self.sessions.get(agent_id)

    async def send_message(self, agent_id: int, message: str) -> None:
        """Record a delivery, after optionally blocking or failing.

        Args:
            agent_id: The caller being messaged.
            message: The text the registry built for the notification.

        Raises:
            Exception: The next scripted failure, when one is left.
        """
        self.attempts += 1
        if self.gate is not None:
            if self.entered is not None:
                self.entered.set()
            await self.gate.wait()
        if self.scripted_failures:
            raise self.scripted_failures.pop(0)
        self.sent.append((agent_id, message))


class _StartingCallerOrchestrator(_FakeOrchestrator):
    """A caller that begins a new turn the instant the registry looks it up.

    The race the delivery gate exists for: the caller was idle when it launched
    the background call and is mid-turn by the time the call finishes.
    """

    def get_session(self, agent_id: int):
        """Return the session with its state forced to running.

        Args:
            agent_id: The caller the registry wants to deliver into.

        Returns:
            The registered session, now reported as running.
        """
        session = self.sessions.get(agent_id)
        if session is not None:
            session.state = RUNNING_STATE_VALUE
        return session


class _BlockingFirstSendOrchestrator(_FakeOrchestrator):
    """Holds the *first* send open and lets every later one through.

    Blocking only the first is what makes the overlap testable without a hang: a
    second dispatch that wrongly believes the handle is still owed will reach
    ``send_message`` and return, instead of parking on the same gate forever.
    """

    async def send_message(self, agent_id: int, message: str) -> None:
        """Block on the gate only when this is the first send.

        Args:
            agent_id: The caller being messaged.
            message: The text the registry built for the notification.

        Raises:
            Exception: The next scripted failure, when one is left.
        """
        self.attempts += 1
        if self.attempts == 1 and self.gate is not None:
            if self.entered is not None:
                self.entered.set()
            await self.gate.wait()
        if self.scripted_failures:
            raise self.scripted_failures.pop(0)
        self.sent.append((agent_id, message))


class _Returns:
    """Coroutine factory returning a fixed value."""

    def __init__(self, value: object) -> None:
        """Hold the value the detached call will produce.

        Args:
            value: What the coroutine returns once awaited.
        """
        self.value = value

    async def __call__(self) -> object:
        """Return the configured value.

        Returns:
            The value given to the constructor.
        """
        return self.value


class _Raises:
    """Coroutine factory raising a fixed exception."""

    def __init__(self, exc: BaseException) -> None:
        """Hold the exception the detached call will raise.

        Args:
            exc: The exception to raise once awaited.
        """
        self.exc = exc

    async def __call__(self):
        """Raise the configured exception.

        Raises:
            BaseException: The exception given to the constructor.
        """
        raise self.exc


class _ExplosiveStr:
    """A return value whose ``__str__`` raises, as some proxy classes do."""

    def __str__(self) -> str:
        """Fail on purpose: the registry renders the value with ``str``.

        Raises:
            RuntimeError: Always, which is the whole point of this class.
        """
        raise RuntimeError("__str__ exploded")


# --- helpers ---------------------------------------------------------------


async def _drain_loop(times: int = 10) -> None:
    """Yield to the event loop enough times for scheduled callbacks to run.

    Used after cancelling detached work so their done-callbacks and the
    bookkeeping they trigger run before the loop is torn down.

    Args:
        times: How many zero-length sleeps to perform.
    """
    for _ in range(times):
        await asyncio.sleep(0)


async def _settled(registry: BackgroundRegistry, handle: str) -> BackgroundTask:
    """Wait for *handle* to leave RUNNING and return its record unspending it.

    ``pending_notifications`` is the probe rather than ``collect`` because
    ``collect`` *spends* a result: the first call that observes a terminal state
    clears the stored text and flips the record to COLLECTED.  Waiting through
    ``collect`` would destroy the state the tests below are about to assert on.

    Args:
        registry: The registry holding the handle.
        handle: The handle to wait for.

    Returns:
        A snapshot of the settled record, copied from the registry's own.

    Raises:
        AssertionError: When the handle is still running after the poll budget.
    """
    for _ in range(POLL_ATTEMPTS):
        for record in registry.pending_notifications():
            if record.handle == handle:
                return record
        await asyncio.sleep(POLL_DELAY_SECONDS)
    raise AssertionError(
        f"{handle} was still RUNNING after {POLL_ATTEMPTS * POLL_DELAY_SECONDS:.1f}s"
    )


def _settled_registry(*, caller_id=None, work=None, timeout=LONG_CEILING_MS):
    """Launch work that finishes at once, and return ``(registry, handle, ctx)``.

    Args:
        caller_id: The owning turn's agent id, or ``None`` for none.
        work: The coroutine factory; defaults to returning ``WORK_RESULT``.
        timeout: The ceiling in milliseconds.

    Returns:
        The registry, the fresh handle, and a context with no orchestrator.
    """
    registry = BackgroundRegistry()
    handle = registry.launch(
        work if work is not None else _Returns(WORK_RESULT),
        caller_id=caller_id,
        timeout=timeout,
    )
    return registry, handle, ToolContext()


def _context(orchestrator) -> ToolContext:
    """Build a context whose orchestrator is *orchestrator*.

    Args:
        orchestrator: The double to attach.

    Returns:
        A context with the orchestrator already installed.
    """
    return ToolContext(orchestrator=orchestrator)


def _handles(records) -> list:
    """Reduce a list of records to their handles, for order and identity asserts.

    Args:
        records: ``BackgroundTask`` snapshots.

    Returns:
        The handles, in the order the records were listed.
    """
    return [record.handle for record in records]


# --- the vocabulary --------------------------------------------------------


def test_task_state_has_exactly_the_six_documented_members():
    """Adding a state has to be a decision, so the vocabulary is pinned."""
    assert set(TaskState.__members__) == {
        "RUNNING",
        "DONE",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "COLLECTED",
    }, (
        "TaskState drifted from the documented lifecycle; pending_notifications "
        "walks TERMINAL_STATES and every consumer switches on these members"
    )


def test_task_state_values_are_the_wire_strings_a_notification_embeds():
    """``NOTIFICATION_TEXT`` puts ``state.value`` in front of the model."""
    assert {state.name: state.value for state in TaskState} == {
        "RUNNING": "running",
        "DONE": "done",
        "FAILED": "failed",
        "TIMEOUT": "timeout",
        "CANCELLED": "cancelled",
        "COLLECTED": "collected",
    }, "the state values are the wire strings a delivered notification quotes"


def test_running_is_the_only_non_terminal_state():
    """``pending_notifications`` walks the set, so this is load-bearing."""
    assert TaskState.RUNNING not in TERMINAL_STATES, (
        "a running task has nothing to announce, so it must not be terminal"
    )
    assert TERMINAL_STATES == {
        TaskState.DONE,
        TaskState.FAILED,
        TaskState.TIMEOUT,
        TaskState.CANCELLED,
    }, (
        "COLLECTED is absent on purpose — a collected task has already told the "
        "model, and announcing it again would be noise"
    )


def test_background_task_is_a_dataclass_with_the_fields_collect_hands_back():
    """The record is the thing ``collect`` returns and delivery reads."""
    import dataclasses

    assert dataclasses.is_dataclass(BackgroundTask), (
        "a plain class would not give callers a value-comparable snapshot"
    )
    assert tuple(f.name for f in dataclasses.fields(BackgroundTask)) == (
        "handle",
        "state",
        "result",
        "caller_id",
        "timeout",
    ), "the field order is the order the docstring and every reader assume"


# --- launch ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_returns_incrementing_handles_starting_at_bg_1():
    """Handles are the only thing that survives the process, so they are the key."""
    registry = BackgroundRegistry()
    work = _Returns(WORK_RESULT)

    handles = [
        registry.launch(work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
        for _ in range(3)
    ]

    assert handles == ["bg_1", "bg_2", "bg_3"], (
        "the counter starts at 1 and increments; a zero-based handle reads "
        f"like a placeholder, got {handles!r}"
    )
    assert len(set(handles)) == 3, "handles must be unique within one registry"


@pytest.mark.asyncio
async def test_launch_returns_before_the_detached_work_has_run():
    """The whole point: a ten-second call costs the caller one scheduling tick."""
    registry = BackgroundRegistry()
    started = asyncio.Event()

    async def slow_work() -> str:
        started.set()
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    snapshot = await registry.collect(handle)

    assert snapshot.state is TaskState.RUNNING, (
        "collect right after launch must find the work in flight"
    )
    assert snapshot.result == "", "a running task has no result to hand back yet"
    assert not started.is_set(), (
        "launch must not have run the coroutine; a ten-second operation has to "
        "cost the caller the same as a ten-millisecond one"
    )
    registry.cancel_all()
    await _drain_loop()


def test_launch_outside_a_running_loop_raises_a_builtin_tool_error():
    """A missing loop is a wiring fault, and it has its own exception type."""
    registry = BackgroundRegistry()
    calls = []

    def never_called():
        calls.append(True)
        return _Returns(WORK_RESULT)()

    with pytest.raises(MitoCodeBuiltinToolError, match="outside a running event loop"):
        registry.launch(never_called, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)

    assert calls == [], (
        "the factory must not be invoked when the launch is rejected; it is a "
        "factory precisely so that nothing is built when nothing is scheduled"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, None], ids=["zero", "none"])
async def test_launch_replaces_a_falsy_ceiling_with_the_five_minute_default(timeout):
    """A model writing ``timeout=0`` means "no ceiling", not "fail instantly"."""
    registry = BackgroundRegistry()
    started = asyncio.Event()

    async def slow_work() -> str:
        started.set()
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=timeout)
    snapshot = await registry.collect(handle)

    assert snapshot.timeout == EXPECTED_DEFAULT_CEILING_MS, (
        "a falsy ceiling must fall back to five minutes, not to zero — a zero "
        "budget would fail every call the model makes"
    )
    assert snapshot.state is TaskState.RUNNING, (
        "the substituted ceiling must not fire immediately"
    )
    registry.cancel_all()
    await _drain_loop()


@pytest.mark.asyncio
async def test_launch_keeps_a_negative_ceiling_verbatim():
    """Only a *falsy* ceiling is replaced; a negative one is a real request."""
    registry = BackgroundRegistry()

    async def slow_work() -> str:
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=-100)
    snapshot = await registry.collect(handle)

    assert snapshot.timeout == -100, (
        "the record carries the ceiling as asked so a timeout can be explained "
        "after the fact; substituting here would hide the caller's mistake"
    )
    registry.cancel_all()
    await _drain_loop()


@pytest.mark.asyncio
async def test_a_negative_ceiling_expires_the_task_immediately():
    """The millisecond conversion floors at zero rather than rejecting.

    Hand-computed: ``max(-100, 0) / 1000`` is ``0.0`` seconds, so the deadline
    future completes on the first pass through the loop and the ceiling fires.
    """
    registry = BackgroundRegistry()

    async def slow_work() -> str:
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=-100)
    record = await _settled(registry, handle)

    assert record.state is TaskState.TIMEOUT, (
        "a negative ceiling must not raise out of the tool body; the documented "
        "reading of a nonsensical budget is 'expire immediately'"
    )
    assert record.result == "timed out after -100 ms", (
        "the timeout text reports the ceiling that was actually asked for"
    )
    registry.cancel_all()
    await _drain_loop()


@pytest.mark.asyncio
async def test_a_one_millisecond_ceiling_is_not_rounded_up_to_the_default():
    """A sub-millisecond ceiling is honoured, not treated as nonsense.

    Hand-computed: ``1 ms / 1000`` is ``0.001`` seconds.  An implementation that
    guarded the conversion with a minimum would clamp this to the five-minute
    default and the task would still be running when the assertion below runs.
    """
    registry = BackgroundRegistry()

    async def slow_work() -> str:
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=1)
    record = await _settled(registry, handle)

    assert record.state is TaskState.TIMEOUT, (
        "a one-millisecond ceiling must expire within the test, so it cannot "
        f"have been rounded up to {EXPECTED_DEFAULT_CEILING_MS} ms"
    )
    assert record.result == "timed out after 1 ms", (
        "the timeout text must report the ceiling that actually elapsed"
    )
    registry.cancel_all()
    await _drain_loop()


@pytest.mark.parametrize(
    "bad_ceiling",
    ["soon", object(), ["soon"]],
    ids=["unparseable-string", "not-a-number-at-all", "list-of-strings"],
)
@pytest.mark.asyncio
async def test_launch_keeps_a_non_numeric_ceiling_verbatim_without_expiring_on_it(
    bad_ceiling,
):
    """A model can emit anything, and a broken ceiling is not a crashed tool.

    Two rejection paths have to be covered, because the conversion catches two
    exception types.  ``"soon"`` is a string ``float()`` refuses with a
    ``ValueError``; a bare object and a list are refused with a ``TypeError``
    instead.  A guard that catches only the first leaves the deadline task to
    raise out of the wrapper, and the caller is left with an orphaned traceback
    and a record that never settles at all.
    """
    registry = BackgroundRegistry()

    async def slow_work() -> str:
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=bad_ceiling)
    snapshot = await registry.collect(handle)

    assert snapshot.timeout == bad_ceiling, (
        "the ceiling is stored as asked; rewriting it would lose what the model "
        f"actually requested (got {snapshot.timeout!r})"
    )
    assert snapshot.state is TaskState.RUNNING, (
        "the call must be in flight, not failed by the ceiling it was given"
    )
    assert snapshot.result == "", (
        "a running call has nothing to report, least of all a complaint about "
        f"its own ceiling (got {snapshot.result!r})"
    )
    await asyncio.sleep(POLL_DELAY_SECONDS * 10)
    assert (await registry.collect(handle)).state is TaskState.RUNNING, (
        "an unparseable ceiling falls back to the five-minute default rather "
        "than expiring the call on the spot"
    )
    registry.cancel_all()
    await _drain_loop()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_id", [None, 7], ids=["no-owning-turn", "agent-seven"])
async def test_launch_stores_the_caller_id_on_the_handle(caller_id):
    """A ``None`` caller is collectable but never notifiable."""
    registry, handle, _ = _settled_registry(caller_id=caller_id)
    snapshot = await registry.collect(handle)

    assert snapshot.caller_id == caller_id, (
        "the owning turn travels on the record; the delivery gate skips a "
        f"handle whose caller is None, so this field decides it (got {caller_id!r})"
    )


# --- fix 1: a timeout must never be recorded as a success -----------------


@pytest.mark.asyncio
async def test_work_that_suppresses_its_own_cancellation_is_recorded_as_timed_out():
    """The one outcome this module exists to prevent.

    ``asyncio.wait_for`` hands back whatever the inner coroutine produced when
    that coroutine swallows its own ``CancelledError`` — verified on CPython
    3.10.20, 3.13.11 and 3.14.0 — so the obvious implementation records a task
    that blew its ceiling as a success.  The registry records the timeout the
    instant the deadline future wins the race, and discards the value.
    """
    registry = BackgroundRegistry()
    released = asyncio.Event()

    async def suppresses_cancellation() -> str:
        try:
            await asyncio.sleep(LONG_SLEEP_SECONDS)
        except asyncio.CancelledError:
            released.set()
            return "a value produced after the ceiling"
        return "unreachable"  # pragma: no cover - the sleep always raises here

    handle = registry.launch(
        suppresses_cancellation, caller_id=AGENT_ID, timeout=SHORT_CEILING_MS
    )
    record = await _settled(registry, handle)

    assert record.state is TaskState.TIMEOUT, (
        "the deadline fired, so TIMEOUT must win over anything the inner "
        f"produced; got {record.state!r} with {record.result!r}"
    )
    assert record.result == "timed out after 20 ms", (
        "the timeout text reports the ceiling that actually elapsed"
    )
    assert record.result != "a value produced after the ceiling", (
        "the value the callee produced after suppressing its cancellation must "
        "be discarded, not reported as the outcome"
    )
    await released.wait()
    await _drain_loop()


@pytest.mark.asyncio
async def test_a_late_failure_after_the_ceiling_is_logged_and_never_promoted():
    """Work that outlives its ceiling may still fail; the answer does not move.

    The registry cancels the work twice — once from the deadline task and once
    after the race is settled — so the work here absorbs a fixed number of
    cancellations before failing, which is the ``asyncio.shield`` case the source
    docstring describes.  It is released only after the record has settled, so
    the failure is guaranteed to arrive with ``TIMEOUT`` already written rather
    than racing it.
    """
    registry = BackgroundRegistry()
    released = asyncio.Event()

    async def survives_then_fails() -> str:
        try:
            await asyncio.sleep(LONG_SLEEP_SECONDS)
        except asyncio.CancelledError:
            # The registry cancels the work once from the deadline task and once
            # more after the race is settled; both land here, so each one is
            # absorbed and the wait retried until the test releases it.
            for _ in range(3):
                try:
                    await released.wait()
                except asyncio.CancelledError:
                    continue
            raise RuntimeError("late failure")
        return "unreachable"  # pragma: no cover - the sleep always raises here

    handle = registry.launch(
        survives_then_fails, caller_id=AGENT_ID, timeout=SHORT_CEILING_MS
    )
    try:
        settled = await _settled(registry, handle)
        assert settled.state is TaskState.TIMEOUT, (
            "the ceiling fires first; the late failure is a separate event"
        )

        lines = []
        sink = loguru_logger.add(lines.append, level="DEBUG")
        try:
            released.set()
            await _drain_loop()
        finally:
            loguru_logger.remove(sink)
        logged = "\n".join(lines)

        after = registry.pending_notifications()
        assert [record.state for record in after] == [TaskState.TIMEOUT], (
            "a failure from work that outlived its ceiling must never be promoted "
            "over the timeout; the ceiling is the registry's promise"
        )
        assert after[0].result == "timed out after 20 ms", (
            "the timeout text is unchanged by the late failure"
        )
        assert handle in logged, "the line must name the handle whose work misbehaved"
        assert "late failure" in logged, (
            "the late exception must reach the log, otherwise the drain would "
            "only be suppressing asyncio's orphan warning with nothing in its place"
        )
    finally:
        # Unconditional: this work is built not to be stoppable, so an assertion
        # that fires before the release would leave the event loop holding a task
        # nothing can finish.
        released.set()
        registry.cancel_all()
        await _drain_loop()


@pytest.mark.asyncio
async def test_a_tool_raising_its_own_timeout_error_is_recorded_as_failed():
    """``asyncio.TimeoutError`` is the builtin on 3.11+, so the type cannot decide.

    A socket or subprocess timeout raised by the tool itself is exactly the work
    this registry exists to run, and it is indistinguishable from the registry's
    own deadline by exception type.  The outcome is decided by *which future
    completed*, so the tool's own timeout is an ordinary failure on every
    supported interpreter — 3.10, where the two classes are still distinct,
    included.
    """
    registry = BackgroundRegistry()

    async def own_timeout() -> str:
        raise asyncio.TimeoutError("socket timed out")

    handle = registry.launch(own_timeout, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    record = await _settled(registry, handle)

    assert record.state is TaskState.FAILED, (
        "only the registry's own deadline may produce TIMEOUT; a timeout raised "
        f"by the callee is an ordinary failure (got {record.state!r})"
    )
    assert record.result == "Error: TimeoutError: socket timed out", (
        "the failure is rendered with its real type and message so the model "
        "can tell a tool timeout from a registry timeout"
    )
    assert "timed out after" not in record.result, (
        "the registry must not invent a ceiling the call never hit"
    )


@pytest.mark.asyncio
async def test_a_tool_raising_the_builtin_timeout_error_is_also_failed():
    """The other spelling of the same exception, for completeness of the matrix."""
    registry = BackgroundRegistry()

    async def own_timeout() -> str:
        raise TimeoutError("subprocess timed out")

    handle = registry.launch(own_timeout, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    record = await _settled(registry, handle)

    assert record.state is TaskState.FAILED, (
        "the builtin spelling must reach the same ledger entry on every "
        f"interpreter (got {record.state!r})"
    )
    assert record.result == "Error: TimeoutError: subprocess timed out", (
        "the failure must name the exception type the tool really raised"
    )


# --- cancellation is recorded, never converted into success ---------------


@pytest.mark.asyncio
async def test_a_wrapper_cancelled_from_outside_is_recorded_as_cancelled_not_done():
    """The wrapper catches ``CancelledError`` only to write the state, then re-raises.

    ``cancel_all`` writes the state itself, so the only way to reach the
    wrapper's own clause is to cancel the task directly — which is what shutdown
    and a caller changing its mind do.  ``cancel_all`` cannot be used here
    because it would pre-empt the branch under test.  The registry's ``_tasks``
    map is the only handle on the wrapper task, so it is read directly; the
    ``pylint`` pragma marks the one intentional reach into internals in this file.
    """
    registry = BackgroundRegistry()
    started = asyncio.Event()

    async def slow_work() -> str:
        started.set()
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    await started.wait()

    task = registry._tasks[handle]  # pylint: disable=protected-access
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled(), "the cancellation must be re-raised, not absorbed"
    snapshot = await registry.collect(handle)
    assert snapshot.state is TaskState.CANCELLED, (
        "a task cancelled from outside must not be reported as a success"
    )
    assert snapshot.result == "cancelled before it finished", (
        "a cancelled task carries the explanation the model is given"
    )
    assert snapshot.state is not TaskState.DONE, (
        "and it must never read as the success it did not reach"
    )


@pytest.mark.asyncio
async def test_a_wrapper_cancelled_before_its_first_line_ran_is_reconciled():
    """The safety net: a wrapper that never started still has an answer.

    Cancelling a task before the loop has run its first line means the wrapper
    body — the thing that normally settles the record — never executed, so the
    registry has to reconcile the outcome itself rather than leave the handle
    reading ``RUNNING`` forever.  Exactly one pass of the loop is allowed before
    the question is asked, because callbacks are scheduled with ``call_soon``
    and not run inline; a caller that collects immediately must not depend on
    that timing.
    """
    registry = BackgroundRegistry()
    ran = []

    async def never_runs() -> str:
        ran.append(True)
        return WORK_RESULT  # pragma: no cover - the task is cancelled first

    handle = registry.launch(never_runs, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    registry._tasks[handle].cancel()  # pylint: disable=protected-access
    await asyncio.sleep(0)

    snapshot = await registry.collect(handle)

    assert ran == [], "the wrapper must never have run a single line"
    assert snapshot.state is TaskState.CANCELLED, (
        "a handle whose wrapper was cancelled before it started must still be "
        f"answerable (got {snapshot.state!r})"
    )
    assert snapshot.result == "cancelled before it finished", (
        "an unreconciled handle must still answer with the cancelled text"
    )
    assert snapshot.state is not TaskState.DONE, (
        "and never as a success the work never got to produce"
    )


@pytest.mark.asyncio
async def test_work_that_hands_back_an_already_cancelled_future_is_cancelled():
    """The work itself may arrive already cancelled, and that is not a failure.

    ``CoroFactory`` promises an awaitable, not a coroutine, so a tool may hand
    back a future that is cancelled before the registry ever sees it.  Nothing
    about it failed and nothing about it succeeded, so it is recorded as
    cancelled.
    """

    def already_cancelled():
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        return future

    registry = BackgroundRegistry()
    handle = registry.launch(
        already_cancelled, caller_id=AGENT_ID, timeout=LONG_CEILING_MS
    )
    record = await _settled(registry, handle)

    assert record.state is TaskState.CANCELLED, (
        "a cancelled inner future is not a failure and not a success, and the "
        f"ceiling never fired (got {record.state!r})"
    )
    assert record.result == "cancelled before it finished", (
        "a cancelled future is reported with the same text as any other"
    )


# --- a factory that never handed over any work to run ---------------------


@pytest.mark.asyncio
async def test_a_factory_that_raises_is_recorded_as_failed_not_left_running():
    """A factory that never produces an awaitable takes the wrapper down early.

    ``asyncio.ensure_future(coro_factory())`` is the first statement of the
    wrapper, so a factory that raises — or hands back something that is not
    awaitable — ends the wrapper task before a single outcome has been recorded.
    Reconciling that record from outside the wrapper is what
    ``BackgroundRegistry._settle`` exists for, and it is a live path rather than
    a defensive one: the tool mini-DSL parses arbitrary literals, so a model can
    supply a value here that is not callable at all.
    """

    def raising_factory():
        raise TypeError("factory blew up")

    registry = BackgroundRegistry()
    handle = registry.launch(raising_factory, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)

    record = await _settled(registry, handle)

    assert record.state is TaskState.FAILED, (
        "a factory that never handed over work has failed, and the record must "
        f"say so rather than read RUNNING forever (got {record.state!r})"
    )
    assert record.result == "Error: TypeError: factory blew up", (
        "the failure is rendered with its real type and message, because the "
        "wrapper never got the chance to render anything itself"
    )


@pytest.mark.asyncio
async def test_a_factory_that_returns_a_non_awaitable_is_recorded_as_failed():
    """The other way a factory can be unusable: a value instead of an awaitable.

    The rendering prefix is pinned and the interpreter's own wording is not,
    because that wording belongs to ``asyncio`` rather than to this module and
    has already changed between supported versions.
    """

    def non_awaitable_factory():
        return 42

    registry = BackgroundRegistry()
    handle = registry.launch(
        non_awaitable_factory, caller_id=AGENT_ID, timeout=LONG_CEILING_MS
    )

    record = await _settled(registry, handle)

    assert record.state is TaskState.FAILED, (
        "a value where an awaitable belongs is a factory failure, and the "
        f"record must be settled rather than left RUNNING (got {record.state!r})"
    )
    assert record.result.startswith("Error: TypeError: "), (
        "the failure must be rendered with its real type so a model can branch "
        f"on it (got {record.result!r})"
    )
    assert record.result != "", "an empty result would tell the model nothing"


# --- the four outcomes the registry records -------------------------------


@pytest.mark.asyncio
async def test_a_returning_coroutine_is_recorded_as_done():
    """The ordinary happy path, and the only one that carries the callee's text."""
    registry, handle, _ = _settled_registry()
    record = await _settled(registry, handle)

    assert record.state is TaskState.DONE, (
        "a returned coroutine settles DONE, which is the ordinary path"
    )
    assert record.result == WORK_RESULT, "collect hands back the rendered return"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(42, "42"), (None, "None"), (["a", "b"], "['a', 'b']")],
    ids=["int", "none", "list"],
)
@pytest.mark.asyncio
async def test_a_non_string_return_value_is_rendered_through_str(value, expected):
    """Built-in tools return text, but the guard costs nothing.

    ``None`` is the case that matters: it has to render as ``"None"`` so that a
    tool which returned nothing is distinguishable from one that has not finished.
    """
    registry = BackgroundRegistry()
    handle = registry.launch(
        _Returns(value), caller_id=AGENT_ID, timeout=LONG_CEILING_MS
    )
    record = await _settled(registry, handle)

    assert record.state is TaskState.DONE, (
        "a value that rendered successfully still settled DONE"
    )
    assert record.result == expected, (
        f"a non-string return is rendered with str(), not repr() (got {record.result!r})"
    )


@pytest.mark.asyncio
async def test_a_return_value_whose_str_raises_is_recorded_as_done_not_failed():
    """A hostile ``__str__`` must not turn a success into a failure.

    The work completed, so the record must read ``DONE``; the rendering guard
    keeps the crash inside ``_render``.  Note the returned text is the message of
    the exception ``__str__`` raised, which is what ``safe_exception_message``
    yields — the ``_render`` docstring calls it "a placeholder naming the type",
    and the two descriptions disagree.  The behaviour is pinned as implemented so
    a change to either the code or its description is a visible edit.
    """
    registry = BackgroundRegistry()
    handle = registry.launch(
        _Returns(_ExplosiveStr()), caller_id=AGENT_ID, timeout=LONG_CEILING_MS
    )
    record = await _settled(registry, handle)

    assert record.state is TaskState.DONE, (
        "the work completed; a value that cannot be rendered is not a failure"
    )
    assert record.result == "__str__ exploded", (
        "the guard returns safe_exception_message(exc), which for a renderable "
                "message is that message rather than a name for its type"
    )


@pytest.mark.asyncio
async def test_an_ordinary_exception_is_recorded_as_failed_and_not_re_raised():
    """A failing tool must not take the agent's turn down with a traceback."""
    registry = BackgroundRegistry()
    handle = registry.launch(
        _Raises(RuntimeError("boom")), caller_id=AGENT_ID, timeout=LONG_CEILING_MS
    )
    record = await _settled(registry, handle)

    assert record.state is TaskState.FAILED, (
        "an ordinary exception is a failure, never a timeout"
    )
    assert record.result == "Error: RuntimeError: boom", (
        "the failure carries the real type and message for the model"
    )
    assert isinstance(record.result, str), (
        "the exception object must never reach the model, only its rendering"
    )


# --- collect ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_of_a_handle_the_registry_never_issued_reads_running():
    """A spent handle and a wrong handle are different conditions."""
    registry = BackgroundRegistry()

    snapshot = await registry.collect("bg_404")

    assert snapshot.handle == "bg_404", "the handle asked about is echoed back"
    assert snapshot.state is TaskState.RUNNING, (
        "a handle this registry never issued must not read as finished; the "
        "model has to re-issue the call rather than wait for something that "
        "will never land"
    )
    assert snapshot.result == "unknown handle", (
        "the marker is what distinguishes 'never existed' from 'nothing to say'"
    )
    assert snapshot.caller_id is None, (
        "an unissued handle has no owning turn"
    )
    # `timeout` is deliberately not asserted here: zero is a plausible
    # placeholder, but it is a sentinel nothing else depends on, and pinning it
    # would fail on a change with no behavioural consequence.


@pytest.mark.asyncio
async def test_collect_of_a_handle_still_in_flight_reads_running_without_waiting():
    """Waiting here would reintroduce the blocking this registry removes."""
    registry = BackgroundRegistry()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_work() -> str:
        started.set()
        await release.wait()
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    await started.wait()

    snapshot = await registry.collect(handle)

    assert snapshot.state is TaskState.RUNNING, "collect reports, it never waits"
    assert snapshot.result == "", (
        "a running call has nothing to hand back"
    )
    assert snapshot.timeout == LONG_CEILING_MS, (
        "a running record still carries the ceiling, so the model can be told "
        "how long it has"
    )
    assert registry.pending_notifications() == [], (
        "an unfinished task has nothing to announce"
    )
    release.set()
    await _settled(registry, handle)


@pytest.mark.asyncio
async def test_a_second_collect_answers_collected_with_an_empty_result():
    """A result is spent once, so the model cannot be told the same thing twice."""
    registry, handle, _ = _settled_registry()
    await _settled(registry, handle)

    first = await registry.collect(handle)
    second = await registry.collect(handle)

    assert first.state is TaskState.DONE, "the first caller gets the outcome"
    assert first.result == WORK_RESULT, (
        "and hands back the rendered value the callee returned"
    )
    assert second.state is TaskState.COLLECTED, (
        "'ask again and you will get nothing' is the contract for a spent handle"
    )
    assert second.result == "", "the stored text is handed over exactly once"


@pytest.mark.asyncio
async def test_the_snapshot_from_the_first_collect_keeps_the_text_the_second_cannot():
    """The return is a copy, so a holder is not mutated by the next caller."""
    registry, handle, _ = _settled_registry()
    await _settled(registry, handle)

    first = await registry.collect(handle)
    await registry.collect(handle)

    assert (first.state, first.result) == (TaskState.DONE, WORK_RESULT), (
        "the snapshot a caller is holding must still read the state it was "
        f"given, got {first.state!r} / {first.result!r}"
    )


@pytest.mark.asyncio
async def test_two_concurrent_collects_do_not_both_observe_the_result():
    """Two callers racing one handle must not both be handed the same answer.

    ``collect`` awaits nothing, so the two calls cannot interleave: whichever
    runs first settles and spends the record and the other sees the spent form.
    That atomicity is the property, and it holds without a lock because the
    registry is only ever reached from the event loop thread.
    """
    registry, handle, _ = _settled_registry()
    await _settled(registry, handle)

    first, second = await asyncio.gather(
        registry.collect(handle), registry.collect(handle)
    )

    observed = [snap for snap in (first, second) if snap.result == WORK_RESULT]
    assert len(observed) == 1, (
        "exactly one of two simultaneous collects may observe the stored result, "
        f"got {[(snap.state, snap.result) for snap in (first, second)]!r}"
    )
    assert {first.state, second.state} == {TaskState.DONE, TaskState.COLLECTED}, (
        "one caller gets the outcome and the other the spent form"
    )


@pytest.mark.asyncio
async def test_collect_of_an_unknown_handle_reads_the_same_however_often_it_is_asked():
    """Asking about a wrong handle repeatedly must not change the answer."""
    registry = BackgroundRegistry()

    reads = [(await registry.collect("bg_404")) for _ in range(3)]

    assert all(read.state is TaskState.RUNNING for read in reads), (
        "the state must not drift between repeated asks"
    )
    assert all(read.result == "unknown handle" for read in reads), (
        "nor may the marker change"
    )


@pytest.mark.parametrize(
    "handle",
    ["", "bg_", "bg_-1", "nonsense", "BG_1", None],
    ids=["empty", "bare-prefix", "negative-number", "unrelated", "wrong-case", "none"],
)
@pytest.mark.asyncio
async def test_collect_of_a_handle_it_could_never_have_issued_reads_unknown(handle):
    """Whatever a model invents, the answer is the same marker and never an error.

    Handles are opaque strings that arrive from a model, so an implementation
    that parses one — to validate the prefix or the number — would raise on
    anything a model actually emits.  ``bg_1`` must not answer for ``BG_1``
    either: the registry keys on what it issued, not on what looks like it.
    """
    registry = BackgroundRegistry()

    snapshot = await registry.collect(handle)

    assert snapshot.handle == handle, "the handle asked about is echoed back verbatim"
    assert snapshot.state is TaskState.RUNNING, (
        "an unissued handle reads RUNNING so the model re-issues the call"
    )
    assert snapshot.result == "unknown handle", (
        "an unissued handle must read as 'nothing to report', never as an error"
    )
    assert registry.pending_notifications() == [], (
        "asking about a handle must not register it"
    )


@pytest.mark.asyncio
async def test_collecting_a_result_takes_it_off_the_notification_list():
    """A collected task has already told the model, so announcing it is noise."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    await _settled(registry, handle)
    assert _handles(registry.pending_notifications()) == [handle], (
        "and stays owed until something else changes it"
    )

    await registry.collect(handle)

    assert registry.pending_notifications() == [], (
        "the result was handed to the caller who asked; a notification would "
        "repeat an answer the model already has"
    )


# --- pending_notifications and mark_delivered -----------------------------


@pytest.mark.asyncio
async def test_pending_notifications_is_empty_while_the_task_is_running():
    """Only a settled task owes anything."""
    registry = BackgroundRegistry()

    async def slow_work() -> str:
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    await asyncio.sleep(0)

    assert registry.pending_notifications() == [], (
        "an in-flight task must not appear in the notification list"
    )
    registry.cancel_all()
    await _drain_loop()


@pytest.mark.asyncio
async def test_pending_notifications_is_ordered_by_launch():
    """Order is the launch order, which is the only deterministic reading."""
    registry = BackgroundRegistry()
    work = _Returns(WORK_RESULT)
    handles = [
        registry.launch(work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
        for _ in range(3)
    ]
    for handle in handles:
        await _settled(registry, handle)

    assert _handles(registry.pending_notifications()) == handles, (
        "pending_notifications walks the insertion-ordered record map, so two "
        "registries with the same launches must answer in the same order"
    )


@pytest.mark.asyncio
async def test_pending_notifications_hands_back_copies_not_the_stored_record():
    """A caller that edits what it was given must not edit the registry.

    Delivery reads the result out of these copies, so a shared object would let
    one consumer's formatting edit become another consumer's answer.
    """
    registry, handle, _ = _settled_registry()
    record = (await _settled(registry, handle))

    record.result = "tampered with by a caller"
    record.state = TaskState.RUNNING

    fresh = registry.pending_notifications()[0]
    assert fresh.result == WORK_RESULT, (
        "the registry's own record must be untouched by edits to the copy"
    )
    assert fresh.state is TaskState.DONE, "nor may its state be edited from outside"


@pytest.mark.asyncio
async def test_mark_delivered_clears_the_result_and_leaves_the_state_done():
    """A delivered task is still collectable; it just has nothing to announce."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    await _settled(registry, handle)

    registry.mark_delivered(handle)
    snapshot = await registry.collect(handle)

    assert snapshot.state is TaskState.DONE, (
        "delivery does not consume the outcome; the model can still collect it"
    )
    assert snapshot.result == "", (
        "the text has been read out, and keeping it would only pin memory"
    )
    assert registry.pending_notifications() == [], (
        "a delivered handle is filtered out of the notification list"
    )


@pytest.mark.asyncio
async def test_mark_delivered_is_idempotent_and_tolerates_an_unissued_handle():
    """A late or duplicated call must not raise inside a delivery path."""
    registry = BackgroundRegistry()
    handle = registry.launch(
        _Returns(WORK_RESULT), caller_id=AGENT_ID, timeout=LONG_CEILING_MS
    )
    await _settled(registry, handle)

    registry.mark_delivered(handle)
    registry.mark_delivered(handle)
    registry.mark_delivered("bg_never_issued")
    await registry.collect(handle)
    registry.mark_delivered(handle)

    assert registry.pending_notifications() == [], (
        "marking an unknown handle is a no-op, not a registry of its own"
    )
    assert (await registry.collect(handle)).state is TaskState.COLLECTED, (
        "delivering after the result was spent must not resurrect it"
    )


# --- cancel_all ------------------------------------------------------------


@pytest.mark.parametrize("caller_id", [AGENT_ID, None], ids=["owned", "unowned"])
@pytest.mark.asyncio
async def test_cancel_all_records_a_running_task_as_cancelled_before_it_returns(
    caller_id,
):
    """Synchronous by design: a caller that reads straight after must not see
    ``RUNNING`` for a task it just stopped, and must never see ``DONE``.

    Parametrised over the caller id because "never notified" and "still stopped"
    are different questions: a launch with no owning turn is skipped by delivery,
    but it is still work the registry is holding and must still cancel.
    """
    registry = BackgroundRegistry()
    started = asyncio.Event()

    async def slow_work() -> str:
        started.set()
        await asyncio.sleep(LONG_SLEEP_SECONDS)
        return WORK_RESULT

    handle = registry.launch(slow_work, caller_id=caller_id, timeout=LONG_CEILING_MS)
    await started.wait()

    registry.cancel_all()
    snapshot = await registry.collect(handle)

    assert snapshot.state is TaskState.CANCELLED, (
        "the state is written synchronously because the CancelledError is only "
        "scheduled by the time cancel_all returns"
    )
    assert snapshot.result == "cancelled before it finished", (
        "the model must be told the work was stopped, not left blank"
    )
    assert snapshot.state is not TaskState.DONE, (
        "a cancelled task is not a success, whatever it was about to return"
    )
    await _drain_loop()


@pytest.mark.asyncio
async def test_cancel_all_leaves_a_finished_task_untouched():
    """Cancelling a settled task would overwrite an outcome nobody was told."""
    registry, handle, _ = _settled_registry(work=_Raises(RuntimeError("boom")))
    await _settled(registry, handle)

    registry.cancel_all()
    snapshot = await registry.collect(handle)

    assert snapshot.state is TaskState.FAILED, (
        "a terminal task must be left exactly as it was settled"
    )
    assert snapshot.result == "Error: RuntimeError: boom", (
        "the recorded failure text survives the cancellation sweep"
    )


def test_cancel_all_on_an_empty_registry_changes_nothing():
    """Teardown paths call this unconditionally, including on a fresh registry.

    Synchronous by contract: it is called from teardown paths that cannot await,
    so making it a coroutine would break every one of them.
    """
    registry = BackgroundRegistry()

    returned = registry.cancel_all()

    assert returned is None, (
        "cancel_all must stay synchronous; a coroutine return would be "
        f"unawaited at every teardown call site (got {returned!r})"
    )
    assert registry.pending_notifications() == [], (
        "a fresh registry owes nothing"
    )


# --- the delivery gate -----------------------------------------------------


def test_the_running_state_literal_matches_the_agent_enum_it_mirrors():
    """The literal is duplicated and nothing checks it; this is that check."""
    from mitocode.core.agent.agent_state import AgentState

    assert RUNNING_STATE_VALUE == AgentState.RUNNING.value, (
        "AgentState.RUNNING has been renamed or revalued. Drift here can only "
        "park a notification rather than deliver one, but the stall is silent "
        "unless something reports it — see test_a_running_caller_is_parked_"
        "without_a_log_line"
    )


def test_the_deliverable_states_are_every_agent_state_except_running():
    """The module docstring says no such test exists; this is it."""
    from mitocode.core.agent.agent_state import AgentState

    assert set(DELIVERABLE_STATE_VALUES) == {
        state.value for state in AgentState
    } - {AgentState.RUNNING.value}, (
        "DELIVERABLE_STATE_VALUES must name every state this build recognises "
        "as safe; a value left off the list is a notification that never lands"
    )


def test_the_running_value_is_never_on_the_deliverable_list():
    """``park_reason`` re-checks this; the guard would be pointless if it did not."""
    assert RUNNING_STATE_VALUE not in DELIVERABLE_STATE_VALUES, (
        "delivering into a running caller makes send_message cancel that "
        "session and await it, destroying the turn that launched the "
        "background call"
    )


@pytest.mark.parametrize(
    "state", ["idle", "finished", "cancelled", "error"], ids=lambda s: s
)
def test_park_reason_delivers_into_every_recognised_non_running_state(state):
    """``cancelled`` and ``error`` deliver deliberately; see the module docstring."""
    assert park_reason(_FakeSession(state)) is None, (
        f"a caller in {state!r} is identifiable and cannot lose a turn, so the "
        "gate must not park it"
    )


def test_park_reason_reads_a_state_off_an_enum_like_member():
    """A real session carries an enum member, not a bare string."""
    assert park_reason(_FakeSession(SimpleNamespace(value="finished"))) is None, (
        "an enum member must resolve through its .value, like a real session"
    )


def test_park_reason_parks_a_running_caller_with_the_value_it_saw():
    """The ordinary case, and the reason the whole deferral exists."""
    assert park_reason(_FakeSession("running")) == (PARKED_RUNNING, "running"), (
        "the raw value read off the session is handed back for the drift log"
    )


@pytest.mark.parametrize(
    "value",
    ["orbiting", "Idle", " running", "idle "],
    ids=["unknown", "wrong-case", "leading-space", "trailing-space"],
)
def test_park_reason_parks_an_unrecognised_value_and_hands_it_back(value):
    """An unfamiliar value parks and is echoed back so the caller can log it.

    The near-misses are the point: the gate compares against a fixed set of
    literals and never normalises what it reads, so a value that differs from a
    deliverable one only in case or whitespace is unidentifiable rather than
    merely unusual.  Normalising it would both deliver into a state the enum does
    not have and log a value that does not match what the enum actually holds.
    """
    reason, seen = park_reason(_FakeSession(value))

    assert reason == PARKED_UNKNOWN_STATE, (
        f"the gate must not normalise {value!r} into a deliverable state"
    )
    assert seen == value, (
        "the value is echoed verbatim for the drift log; trimming or folding "
        f"case would make the log disagree with the enum (got {seen!r})"
    )


def test_park_reason_parks_a_missing_session_with_no_value_to_report():
    """A session mid-teardown is expected, so nothing is handed back to log."""
    assert park_reason(None) == (PARKED_NO_SESSION, ""), (
        "a missing session is its own park reason, with no value to report"
    )


@pytest.mark.parametrize(
    "session",
    [object(), _FakeSession(None), _FakeSession(7)],
    ids=["no-state-attribute", "state-is-none", "state-is-not-a-string"],
)
def test_park_reason_parks_any_state_it_cannot_identify(session):
    """An unidentifiable state must park rather than be guessed safe."""
    reason, value = park_reason(session)

    assert reason == PARKED_UNKNOWN_STATE, (
        "anything outside the allow-list parks; guessing that an unfamiliar "
        "value is safe is how a renamed enum kills somebody's turn"
    )
    assert value == "", "a state that could not be read is reported as empty"


# --- the gate as delivery sees it ------------------------------------------


@pytest.mark.asyncio
async def test_a_running_caller_is_parked_without_a_log_line():
    """A busy caller is the ordinary path, and logging it would bury drift."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("running")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        delivered = await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)

    assert delivered == [], "a running caller must not be delivered into"
    assert orchestrator.sent == [], "no message may reach a session mid-turn"
    assert not [line for line in lines if handle in line], (
        "parking a busy caller is expected on every completion; a log line for "
        f"it would drown out the one park that is an anomaly, got {lines!r}"
    )
    assert _handles(registry.pending_notifications()) == [handle], (
        "parked, not dropped: the next dispatch retries"
    )


@pytest.mark.asyncio
async def test_a_caller_that_was_running_receives_its_notification_once_it_goes_idle():
    """The session is read live on every dispatch, never captured at launch."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    session = _FakeSession("running")
    orchestrator = _FakeOrchestrator({AGENT_ID: session})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    assert await registry.dispatch_pending(ctx) == [], (
        "a mid-turn caller is never delivered into"
    )
    assert orchestrator.sent == [], (
        "so no message may reach the session at all"
    )

    session.state = "idle"

    assert await registry.dispatch_pending(ctx) == [handle], (
        "a caller that finished after the launch must still be delivered to on "
        "the next dispatch"
    )
    assert orchestrator.sent == [(AGENT_ID, EXPECTED_NOTIFICATION)], (
        "the waiting notification arrives once the caller is idle"
    )


@pytest.mark.asyncio
async def test_a_caller_that_starts_running_is_never_delivered_into():
    """The race the gate exists for: idle at launch, mid-turn at delivery."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _StartingCallerOrchestrator({AGENT_ID: _FakeSession("idle")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    delivered = await registry.dispatch_pending(ctx)

    assert delivered == [], (
        "delivering into a caller that has begun a turn makes send_message "
        "cancel it, discarding whatever it was in the middle of doing"
    )
    assert orchestrator.sent == [], (
        "and nothing is sent while the caller is still mid-turn"
    )
    assert _handles(registry.pending_notifications()) == [handle], (
        "the notification is parked rather than lost"
    )


# --- the drift log ---------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unrecognised_state_is_logged_naming_the_handle_and_the_value():
    """A park with no log line is indistinguishable from a busy agent."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("orbiting")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        delivered = await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)
    logged = "\n".join(lines)

    assert delivered == [], "an unidentifiable state must never be delivered into"
    assert handle in logged, "the log line must name the handle that was parked"
    assert "orbiting" in logged, "the log line must show the value it could not read"
    assert orchestrator.sent == [], (
        "a caller that begins a turn wins over the notification"
    )


@pytest.mark.asyncio
async def test_a_long_unrecognised_state_value_is_truncated_to_sixty_characters():
    """The value passes through ``repr`` and is bounded before it is formatted."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("z" * 200)})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)
    logged = "\n".join(lines)

    # The budget is asserted structurally, as the length of the longest run of
    # the value's own character: exactly the budget survives, and not one more.
    # Asserting a marker word instead would fail on a harmless rewording while
    # passing on a truncation at the wrong length.
    assert "z" * EXPECTED_TRUNCATION_CHARS in logged, (
        f"the first {EXPECTED_TRUNCATION_CHARS} characters must survive intact "
        f"so the reader can see what the value was; the line was {logged!r}"
    )
    assert "z" * (EXPECTED_TRUNCATION_CHARS + 1) not in logged, (
        "one character past the budget must not survive: the value comes from "
        "the agent package, which is not guaranteed to be the source"
    )


@pytest.mark.asyncio
async def test_an_unrecognised_state_containing_a_newline_stays_on_one_log_line():
    """A value with a newline in it must not be able to forge a log line."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    forged = "idle\nAgentState.RUNNING was renamed to 'thinking'"
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession(forged)})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)

    matches = [line for line in lines if "renamed" in line]
    assert len(matches) == 1, (
        f"the forged value must stay inside the one record, got {lines!r}"
    )
    assert matches[0].count("\n") == 1, (
        "loguru terminates each record with a single newline, so more than one "
        f"means the payload carried its own: {matches[0]!r}"
    )
    assert "idle\\nAgentState" in matches[0], (
        "the value must be shown through repr so the newline is escaped"
    )
    assert forged not in matches[0], "the raw value must never reach the log line"


@pytest.mark.asyncio
async def test_the_drift_log_repeats_on_every_dispatch_while_the_drift_persists():
    """De-duplicating the line would hide the one event a reader needs to see.

    The repetition is a documented, accepted trade: a drift that is reported
    once looks like a transient glitch, and the alternative — a seen-set — is
    what would make an ongoing stall invisible.
    """
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("orbiting")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        for _ in range(3):
            await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)
    drift = [line for line in lines if handle in line and "orbiting" in line]

    assert len(drift) == 3, (
        "one line per dispatch, so a persistent drift stays visible; got "
        f"{len(drift)} lines for 3 dispatches"
    )


@pytest.mark.asyncio
async def test_a_session_with_no_usable_state_is_logged_as_missing():
    """An empty read is reported as missing, not as an empty state name."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession(None)})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)
    logged = "\n".join(lines)

    assert handle in logged, (
        "the line must name the handle it parked"
    )
    assert "missing" in logged.lower(), (
        "a state that could not be read at all must be called out as missing; "
        "an empty value would read as a state literally named ''"
    )


# --- dispatch mechanics ----------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_delivers_the_notification_text_the_model_is_meant_to_see():
    """The text is the registry's whole outward-facing contract."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("idle")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    delivered = await registry.dispatch_pending(ctx)

    assert delivered == [handle], (
        "the idle caller is the one whose handle is reported"
    )
    assert orchestrator.sent == [(AGENT_ID, EXPECTED_NOTIFICATION)], (
        "the notification names the handle, the state and the result, in that "
        "order, so the model can act on it"
    )
    assert orchestrator.attempts == 1, "one delivery is one send"


@pytest.mark.asyncio
async def test_dispatch_returns_the_handles_it_delivered_in_launch_order():
    """Several finished calls, several idle callers, one ordered answer."""
    registry = BackgroundRegistry()
    orchestrator = _FakeOrchestrator(
        {AGENT_ID: _FakeSession("idle"), OTHER_AGENT_ID: _FakeSession("idle")}
    )
    ctx = _context(orchestrator)
    work = _Returns(WORK_RESULT)
    handles = [
        registry.launch(work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS),
        registry.launch(work, caller_id=OTHER_AGENT_ID, timeout=LONG_CEILING_MS),
    ]
    for handle in handles:
        await _settled(registry, handle)

    delivered = await registry.dispatch_pending(ctx)

    assert delivered == handles, "the answer must be in launch order"
    assert [agent_id for agent_id, _ in orchestrator.sent] == [
        AGENT_ID,
        OTHER_AGENT_ID,
    ], "each handle must reach its own caller"


@pytest.mark.asyncio
async def test_a_task_launched_with_no_caller_is_never_notified():
    """There is no session to message, and one will never acquire."""
    registry, handle, _ = _settled_registry(caller_id=None)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("idle")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    delivered = await registry.dispatch_pending(ctx)

    assert delivered == [], (
        "a failing send delivers nothing"
    )
    assert orchestrator.sent == [], (
        "and sends nothing at all"
    )
    assert orchestrator.attempts == 0, (
        "a caller-less handle must not even be looked up in the orchestrator"
    )
    assert _handles(registry.pending_notifications()) == [handle], (
        "the unowned handle stays owed rather than being quietly filed away"
    )


@pytest.mark.asyncio
async def test_dispatch_delivers_only_the_eligible_handles_in_a_mixed_pending_set():
    """The whole gate in one pass: four callers, one of them may be told.

    Every other park reason is represented — unowned, running, unidentifiable,
    and a session that has been removed — so a regression that widens delivery
    to any of them cannot hide behind the others still working.
    """
    registry = BackgroundRegistry()
    orchestrator = _FakeOrchestrator(
        {
            AGENT_ID: _FakeSession("idle"),
            BUSY_AGENT_ID: _FakeSession("running"),
            CONFUSED_AGENT_ID: _FakeSession("orbiting"),
        }
    )
    ctx = _context(orchestrator)
    work = _Returns(WORK_RESULT)
    deliverable = registry.launch(work, caller_id=AGENT_ID, timeout=LONG_CEILING_MS)
    unowned = registry.launch(work, caller_id=None, timeout=LONG_CEILING_MS)
    busy = registry.launch(work, caller_id=BUSY_AGENT_ID, timeout=LONG_CEILING_MS)
    confused = registry.launch(work, caller_id=CONFUSED_AGENT_ID, timeout=LONG_CEILING_MS)
    removed = registry.launch(work, caller_id=GONE_AGENT_ID, timeout=LONG_CEILING_MS)
    for handle in (deliverable, unowned, busy, confused, removed):
        await _settled(registry, handle)

    delivered = await registry.dispatch_pending(ctx)

    assert delivered == [deliverable], (
        f"only the idle caller's notification may go out, got {delivered!r}"
    )
    assert [agent_id for agent_id, _ in orchestrator.sent] == [AGENT_ID], (
        "only the one eligible caller is messaged"
    )
    assert set(_handles(registry.pending_notifications())) == {
        unowned,
        busy,
        confused,
        removed,
    }, (
        "every parked handle stays owed; parking must never be a silent drop"
    )


@pytest.mark.asyncio
async def test_a_context_with_no_orchestrator_delivers_nothing():
    """A context built before the orchestrator is attached must park, not raise."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    ctx = ToolContext()
    await _settled(registry, handle)

    delivered = await registry.dispatch_pending(ctx)

    assert delivered == [], (
        "a context with no orchestrator cannot deliver anything"
    )
    assert _handles(registry.pending_notifications()) == [handle], (
        "and must leave the handle owed for a later, wired dispatch"
    )


@pytest.mark.asyncio
async def test_a_missing_context_delivers_nothing():
    """The signature says non-optional; the body still guards the call.

    Pinning the defensive branch: a caller that drops its context mid-turn must
    get an empty list, not an ``AttributeError`` on an agent's turn.
    """
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    await _settled(registry, handle)

    delivered = await registry.dispatch_pending(None)

    assert delivered == [], (
        "no orchestrator means no delivery"
    )
    assert _handles(registry.pending_notifications()) == [handle], (
        "and the handle stays owed"
    )


@pytest.mark.asyncio
async def test_each_handle_is_delivered_exactly_once_however_many_dispatches_run():
    """The registry has no memory of having sent, other than the delivered set."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("idle")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    runs = [await registry.dispatch_pending(ctx) for _ in range(3)]

    assert runs == [[handle], [], []], (
        "only the first dispatch may report the handle; the rest find nothing "
        f"owed, which is a normal outcome and not an error (got {runs!r})"
    )
    assert orchestrator.sent == [(AGENT_ID, EXPECTED_NOTIFICATION)], (
        "the retry delivers the notification the failed send owed"
    )
    assert orchestrator.attempts == 1, "the caller must hear about the work exactly once"


@pytest.mark.asyncio
async def test_two_overlapping_dispatches_cannot_both_deliver_one_handle():
    """The claim is taken before the await, so a second dispatch skips the handle.

    This is the interleaving a sequential "run it three times" test cannot
    reach: the first dispatch is suspended inside ``send_message`` while the
    second one runs from start to finish.
    """
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    gate = asyncio.Event()
    entered = asyncio.Event()
    orchestrator = _BlockingFirstSendOrchestrator(
        {AGENT_ID: _FakeSession("idle")}, gate=gate, entered=entered
    )
    ctx = _context(orchestrator)
    await _settled(registry, handle)
    try:
        in_flight = asyncio.create_task(registry.dispatch_pending(ctx))
        await entered.wait()
        concurrent = await registry.dispatch_pending(ctx)

        assert concurrent == [], (
            "the handle is claimed before the await, so an overlapping dispatch "
            "must not deliver it a second time"
        )
        assert orchestrator.attempts == 1, "the second dispatch must not send at all"

        gate.set()
        assert await in_flight == [handle], (
            "the release lets the interrupted send complete and report its handle"
        )
        assert orchestrator.sent == [(AGENT_ID, EXPECTED_NOTIFICATION)], (
            "and the caller hears about the work exactly once"
        )
    finally:
        # Unconditional: without it, an assertion above that fires before the
        # release leaves a dispatch suspended on a gate nothing will ever open,
        # and the event loop hangs at teardown instead of reporting the failure.
        gate.set()
        if not in_flight.done():
            in_flight.cancel()
        await _drain_loop()


@pytest.mark.asyncio
async def test_a_dispatch_cancelled_mid_send_leaves_the_handle_pending_again():
    """The claim is un-claimed on every way the await can end badly.

    Without the ``CancelledError`` clause the handle would sit in the delivered
    set forever while its record still read as undelivered — a notification that
    is unreachable and yet apparently owed.
    """
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    gate = asyncio.Event()
    entered = asyncio.Event()
    orchestrator = _FakeOrchestrator(
        {AGENT_ID: _FakeSession("idle")}, gate=gate, entered=entered
    )
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    in_flight = asyncio.create_task(registry.dispatch_pending(ctx))
    await entered.wait()
    in_flight.cancel()
    with pytest.raises(asyncio.CancelledError):
        await in_flight

    assert orchestrator.sent == [], "a cancelled send must not have delivered"
    assert _handles(registry.pending_notifications()) == [handle], (
        "the claim must be released, or the notification is lost forever"
    )

    gate.set()

    assert await registry.dispatch_pending(ctx) == [handle], (
        "the interrupted delivery must be retried, not abandoned"
    )
    assert orchestrator.sent == [(AGENT_ID, EXPECTED_NOTIFICATION)], (
        "each handle must reach its own caller, once"
    )
    assert orchestrator.attempts == 2, "the send must be attempted again"


@pytest.mark.asyncio
async def test_a_failed_send_is_retried_and_delivers_on_a_later_dispatch():
    """A failed delivery must not become a failed dispatch call."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator(
        {AGENT_ID: _FakeSession("idle")}, failures=[RuntimeError("sink down")]
    )
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    first = await registry.dispatch_pending(ctx)

    assert first == [], "a raised send is not a delivered handle"
    assert _handles(registry.pending_notifications()) == [handle], (
        "the claim must be released so the notification is retried"
    )

    assert await registry.dispatch_pending(ctx) == [handle], (
        "the retried dispatch reports the handle it delivered"
    )
    assert orchestrator.sent == [(AGENT_ID, EXPECTED_NOTIFICATION)], (
        "and the caller finally hears about the work"
    )
    assert orchestrator.attempts == 2, (
        "an interrupted send must be attempted again, not assumed sent"
    )


@pytest.mark.asyncio
async def test_a_send_that_always_fails_is_attempted_five_times_then_retired():
    """A sink that never works must not be retried forever, once per dispatch."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator(
        {AGENT_ID: _FakeSession("idle")},
        failures=[RuntimeError("sink down")] * (EXPECTED_RETRY_CAP + 4),
    )
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    for _ in range(EXPECTED_RETRY_CAP + 4):
        assert await registry.dispatch_pending(ctx) == [], (
            "a failing send must never be reported as delivered"
        )

    assert orchestrator.attempts == EXPECTED_RETRY_CAP, (
        "delivery must stop at the cap, not keep retrying on every dispatch"
    )
    assert registry.pending_notifications() == [], (
        "the handle is retired from notification, not from the registry"
    )

    for _ in range(3):
        await registry.dispatch_pending(ctx)

    assert orchestrator.attempts == EXPECTED_RETRY_CAP, (
        "a retired handle must not be attempted again on later dispatches"
    )


@pytest.mark.asyncio
async def test_a_retired_handle_is_still_collectable():
    """Retirement gives up the announcement only; nothing asked for is destroyed."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator(
        {AGENT_ID: _FakeSession("idle")},
        failures=[RuntimeError("sink down")] * EXPECTED_RETRY_CAP,
    )
    ctx = _context(orchestrator)
    await _settled(registry, handle)
    for _ in range(EXPECTED_RETRY_CAP):
        await registry.dispatch_pending(ctx)

    snapshot = await registry.collect(handle)

    assert snapshot.state is TaskState.DONE, "the outcome survives the retirement"
    assert snapshot.result == WORK_RESULT, (
        "an agent still holding the handle must be able to collect what it asked for"
    )


@pytest.mark.asyncio
async def test_the_give_up_log_names_the_handle_and_the_still_collectable_result():
    """Giving up is silent otherwise, and would look like a lost result."""
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator(
        {AGENT_ID: _FakeSession("idle")},
        failures=[RuntimeError("sink down")] * EXPECTED_RETRY_CAP,
    )
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    lines = []
    sink = loguru_logger.add(lines.append, level="DEBUG")
    try:
        for _ in range(EXPECTED_RETRY_CAP):
            await registry.dispatch_pending(ctx)
    finally:
        loguru_logger.remove(sink)
    logged = "\n".join(lines)

    assert handle in logged, "the line must name the handle it gave up on"
    assert "collect" in logged.lower(), (
        "the line must say the result is still collectable, or an operator "
        "would read the retirement as a discarded result"
    )


# --- the missing-await hazard ---------------------------------------------


@pytest.mark.asyncio
async def test_calling_dispatch_pending_without_awaiting_delivers_nothing():
    """The docstring's warning, pinned so the hazard cannot be forgotten.

    Calling the coroutine without ``await`` builds a coroutine object, runs
    nothing, and hands back neither the handle list nor a report of failure.  The
    assertions are on what the call returns and on what it failed to do — not on
    a notification arriving later — so nothing here waits on the loop and the
    test cannot hang.  It would become obsolete rather than wrong if the method
    were ever made synchronous, which the first assertion says out loud.
    """
    registry, handle, _ = _settled_registry(caller_id=AGENT_ID)
    orchestrator = _FakeOrchestrator({AGENT_ID: _FakeSession("idle")})
    ctx = _context(orchestrator)
    await _settled(registry, handle)

    unawaited = registry.dispatch_pending(ctx)

    assert inspect.iscoroutine(unawaited), (
        "this test documents the missing-await hazard, which only exists while "
        "dispatch_pending is a coroutine function"
    )
    assert not isinstance(unawaited, list), (
        "an un-awaited call hands back a coroutine where the caller expects the "
        f"list of delivered handles, got {unawaited!r}"
    )
    assert orchestrator.sent == [], (
        "nothing runs until the coroutine is awaited, so the caller is left "
        "holding an object and an empty notification list"
    )
    unawaited.close()
    assert orchestrator.sent == [], "closing the coroutine delivers nothing either"
    assert _handles(registry.pending_notifications()) == [handle], (
        "the handle stays owed, which is exactly the invisible stall"
    )
