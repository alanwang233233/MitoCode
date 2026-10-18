"""Detached tool calls, and the notification that arrives when one finishes.

A long call — a test suite, a build, a search over a large tree — blocks the
agent that asked for it.  :class:`BackgroundRegistry` is the escape hatch:
:meth:`BackgroundRegistry.launch` hands the call to an :mod:`asyncio` task,
returns a short handle immediately, and leaves the agent free to do something
else.  The handle is later redeemed with :meth:`BackgroundRegistry.collect`,
or the agent is told the answer arrived on its own.

Four decisions in here are load-bearing, and none of them is obvious from the
method bodies.

**The ceiling is enforced inside the wrapper, not around the task.**  The
obvious spelling — ``asyncio.wait_for(coro, timeout)`` inside ``collect`` —
looks equivalent and is not.  Wrapping a *task* in ``wait_for`` cancels the
wrapper and leaves the task orphaned: still running, still holding its
subprocess, still able to write files, with nothing left to collect it.  So
:meth:`BackgroundRegistry.launch` creates the task exactly once and that task
runs :meth:`BackgroundRegistry._run`, which applies the ceiling to the inner
coroutine with ``wait_for`` *before* anything observes it.  One task, one
timeout, one place that can cancel it.  Because the deadline has already
passed by the time the record is settled, :meth:`~BackgroundRegistry.collect`
never has a task it must re-timeout — it reads a state, it does not wait.

**Cancellation is recorded and then re-raised, never converted into success.**
A task cancelled from outside — by :meth:`~BackgroundRegistry.cancel_all`, by
the interpreter shutting down, by a caller that changed its mind — must end up
:meth:`TaskState.CANCELLED`, not :meth:`TaskState.DONE`.  The mechanism is
deliberately asymmetric: the wrapper catches :class:`asyncio.CancelledError`
only to write the state, and then lets it continue out of the coroutine.  If it
were swallowed the task would return normally with a ``None`` result, and
:meth:`BackgroundRegistry._settle` would record an empty success — an
"it worked" for a piece of work that never ran.  This is the same rule
:mod:`~mitocode.core.tool.builtin.errors` already enforces at the tool
boundary, and for the same reason: on Python 3.8+ ``CancelledError`` is a
:class:`BaseException`, so it is *not* caught by a plain
``except Exception``.  The ``except CancelledError`` clause below exists to add
the bookkeeping, not to absorb the signal, and the ``raise`` on the next line
is the part that matters.

**A result is spent once.**  :meth:`~BackgroundRegistry.collect` returns the
outcome and clears the stored text, so a second call answers
:meth:`TaskState.COLLECTED` instead of repeating an answer the model already
has.  A handle the registry never issued is a *different* condition and reads
differently: :meth:`TaskState.RUNNING` carrying the marker
:data:`UNKNOWN_HANDLE`.  "Spent" and "never existed" call for different
responses from the model — one means go and do something else, the other means
the handle is wrong and the call must be made again — so they are not
collapsed into a single empty answer.

**Delivery waits for the caller to be idle.**  This is the subtle one.
:meth:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator.send_message`
*cancels* a session that is currently running and awaits the cancelled task
before appending the new prompt.  A completion callback that delivered
immediately would therefore interrupt the very turn that launched the
background call, discarding whatever that agent was in the middle of doing —
and it would do so on every single completion, not occasionally.  So
:meth:`BackgroundRegistry.dispatch_pending` skips any task whose caller is
still running and leaves it parked; the next dispatch, from whoever notices the
session went idle, delivers it.  Each handle is claimed before the await and
un-claimed if the send fails, so a dispatch interrupted mid-flight cannot
deliver the same notification twice.

The "is it running" test reads ``session.state`` by its ``.value`` string
rather than comparing against :class:`~mitocode.core.agent.agent_state.AgentState`.
That is a deliberate constraint, not a shortcut.  The dependency between the two
packages is one-way — ``agent_session`` imports ``tool_manager``, so the tool
package must never import the agent package at runtime — and ``TYPE_CHECKING``
does not help here because the value is needed at runtime for the comparison.
``"running"`` is the string the enum member carries; :func:`_state_value`
accepts either the enum or a bare string so a test double needs no import of
the agent package at all.  Do not "fix" this by importing the enum.

For the same reason :class:`ToolContext` is imported here under
``TYPE_CHECKING`` only.  ``context.py`` imports this module at module level
(to build ``field(default_factory=BackgroundRegistry)``), so a runtime import
back would be a cycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
)

from loguru import logger as loguru_logger

from mitocode.core.tool.builtin.errors import (
    MitoCodeBuiltinToolError,
    safe_exception_message,
)

if TYPE_CHECKING:
    # Type annotation only.  `context.py` imports this module at module level,
    # so importing it back at runtime would be a cycle; see the module
    # docstring.  Nothing in this file calls into the context.
    from mitocode.core.tool.builtin.context import ToolContext

logger = loguru_logger.bind(name=__name__)

#: Shape of the callable :meth:`BackgroundRegistry.launch` is given.  A
#: *factory*, not a coroutine: a coroutine object is created at call time and
#: would hold its frame — and therefore its arguments — even if the launch is
#: never scheduled.  A factory defers all of that to the moment the task runs.
CoroFactory = Callable[[], Awaitable[Any]]

#: ``result`` handed back for a handle this registry never issued.
UNKNOWN_HANDLE = "unknown handle"

#: Value of ``AgentState.RUNNING``, compared as a string so this module never
#: imports the agent package.  See the module docstring.
RUNNING_STATE_VALUE = "running"

#: Prefix of a generated handle; the counter starts at 1 so the first handle a
#: registry issues is ``bg_1`` rather than a zero that reads like a
#: placeholder.
HANDLE_PREFIX = "bg_"

#: Ceiling applied when a caller passes ``None`` or a non-positive timeout.
#: Five minutes: long enough for a test suite, short enough that a wedged
#: subprocess does not pin a session open for the rest of the day.
DEFAULT_TIMEOUT_MS = 300_000

#: Recorded when a task is cancelled from outside.  Distinct from a failure so
#: the model is not told something went wrong with the work when in fact the
#: work was deliberately stopped.
CANCELLED_TEXT = "cancelled before it finished"

#: Template for the timeout outcome.  Names the handle and the ceiling, because
#: "timed out" without them tells the model nothing it can act on.
TIMEOUT_TEXT = "timed out after {timeout} ms"

#: Template for the notification body handed to ``send_message``.
NOTIFICATION_TEXT = "Background task {handle} finished ({state}):\n{result}"


class TaskState(Enum):
    """Lifecycle of one detached call.

    ``RUNNING`` is the only non-terminal member; every other one means the
    outcome is known and ``BackgroundTask.result`` is meaningful (or, for
    ``COLLECTED``, deliberately gone).
    """

    #: The task is live.  ``result`` is empty, except for the unknown-handle
    #: case where it carries :data:`UNKNOWN_HANDLE`.
    RUNNING = "running"

    #: The coroutine returned; ``result`` is its rendered return value.
    DONE = "done"

    #: The coroutine raised an ordinary ``Exception``; ``result`` is the
    #: rendered ``"<Type>: <message>"`` pair, never the exception object.
    FAILED = "failed"

    #: The ceiling passed and the inner coroutine was cancelled by
    #: ``asyncio.wait_for``.  The cancellation is real, not an abandonment.
    TIMEOUT = "timeout"

    #: The task was cancelled from outside — ``cancel_all``, shutdown, a
    #: caller changing its mind — and the wrapper re-raised the signal rather
    #: than absorbing it.  Never reported as success.
    CANCELLED = "cancelled"

    #: The result was handed to somebody (via ``collect``) and has been
    #: dropped.  Reaching this state means "ask again and you will get
    #: nothing", which is not the same as an unknown handle.
    COLLECTED = "collected"


#: States a task can no longer leave.  ``pending_notifications`` walks this
#: rather than testing ``state is not RUNNING`` so that adding a member of
#: :class:`TaskState` is a decision that has to be made twice rather than
#: silently included by omission.
TERMINAL_STATES: FrozenSet[TaskState] = frozenset(
    {
        TaskState.DONE,
        TaskState.FAILED,
        TaskState.TIMEOUT,
        TaskState.CANCELLED,
    }
)


@dataclass
class BackgroundTask:
    """What is known about one detached call.

    Attributes:
        handle: The ``bg_<n>`` string the model holds.  It is the only thing
            that survives the process, so it is also the key everything else is
            filed under.
        state: Where the call has got to.  ``RUNNING`` until the wrapper
            settles it.
        result: The rendered outcome.  Empty while running, empty again once
            collected — the text is handed over exactly once.
        caller_id: Id of the agent whose turn launched this, or ``None`` for a
            launch with no owning turn.  A ``None`` caller is never notified:
            there is no session to send a message to.
        timeout: The ceiling that was requested, in milliseconds.  Carried on
            the record so a timeout can be explained after the fact.
    """

    handle: str
    state: TaskState
    result: str
    caller_id: Optional[int]
    timeout: int


class BackgroundRegistry:
    """Tracks detached tool calls and the notifications they still owe.

    One instance per :class:`~mitocode.core.tool.builtin.context.ToolContext`,
    constructed by the context's ``default_factory`` so two contexts cannot
    share a registry and leak each other's handles into a notification.

    The class is not thread-safe, and does not need to be: every entry point is
    reached from inside an agent turn, which is running on the event loop
    thread.  The mutable state here is guarded by that fact rather than by a
    lock.
    """

    def __init__(self) -> None:
        """Create an empty registry whose first handle is ``bg_1``."""
        # handle -> record.  Ordered by launch, which is what makes
        # `pending_notifications` deterministic.
        self._records: Dict[str, BackgroundTask] = {}
        # handle -> the single task created for it.  Kept apart from the
        # record because the record is handed to callers as a copy and must
        # not carry a live task with it.
        self._tasks: Dict[str, "asyncio.Task[None]"] = {}
        # Handles whose notification has been claimed.  Separate from the
        # record's state: a delivered task is still DONE and still collectable,
        # it just has nothing left to announce.
        self._delivered: Set[str] = set()
        self._counter = 0

    def launch(
        self,
        coro_factory: CoroFactory,
        *,
        caller_id: Optional[int],
        timeout: int,
    ) -> str:
        """Start *coro_factory* as a detached task and return its handle.

        Returns as soon as the task is scheduled.  Nothing here awaits the
        call, so a ten-second operation costs the caller the same as a
        ten-millisecond one — that is the entire point of the registry.

        Args:
            coro_factory: A zero-argument callable returning the awaitable to
                run.  A factory rather than a coroutine object so that nothing
                is built, and no frame held, if the launch never runs.
            caller_id: Id of the agent whose turn is launching this, or
                ``None`` when there is no owning turn.  ``None`` means the
                task can be collected but never notified.
            timeout: Ceiling in milliseconds.  A non-positive or ``None``
                value falls back to :data:`DEFAULT_TIMEOUT_MS`.

        Returns:
            The fresh handle, ``bg_1`` for the first launch on this registry.

        Raises:
            MitoCodeBuiltinToolError: When called with no running event loop.
                A built-in tool body is always inside one, so this means the
                wiring is wrong rather than the request — which is exactly
                what that exception type is for.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise MitoCodeBuiltinToolError(
                "BackgroundRegistry.launch() was called outside a running "
                "event loop; a built-in tool body is always inside one."
            ) from exc

        self._counter += 1
        handle = f"{HANDLE_PREFIX}{self._counter}"
        record = BackgroundTask(
            handle=handle,
            state=TaskState.RUNNING,
            result="",
            caller_id=caller_id,
            timeout=timeout if timeout else DEFAULT_TIMEOUT_MS,
        )
        self._records[handle] = record
        # The single task.  Never re-wrapped: see the module docstring on why
        # a second wait_for around a task orphans it.
        task = loop.create_task(
            self._run(record, coro_factory), name=f"mitocode-background-{handle}"
        )
        self._tasks[handle] = task
        # Safety net, not the main path: `_run` settles the record itself, and
        # it does so *before* the task is marked done.  This callback only
        # matters for a task cancelled before its first line ran, where the
        # coroutine body never got the chance to record anything.
        task.add_done_callback(lambda finished, key=handle: self._on_done(key, finished))
        return handle

    async def collect(self, handle: str) -> BackgroundTask:
        """Redeem *handle* and return what is known about it.

        Does not wait.  A call still in flight comes back ``RUNNING``, which
        is the same answer a caller gets for a handle that has not finished
        yet, and the model is expected to ask again — waiting here would
        reintroduce exactly the blocking this registry exists to remove, on
        the one code path a model uses to check on its work.

        The return is a copy, so a caller that holds it across another
        ``collect`` still sees the state it was given rather than the state
        the second call left behind.

        Args:
            handle: The ``bg_<n>`` string returned by :meth:`launch`.

        Returns:
            A snapshot.  ``DONE``/``FAILED``/``TIMEOUT``/``CANCELLED`` with
            the result text the first time; ``COLLECTED`` with an empty result
            once that text has been taken; ``RUNNING`` with
            :data:`UNKNOWN_HANDLE` when the handle was never issued.
        """
        record = self._records.get(handle)
        if record is None:
            # Never issued.  RUNNING plus the marker, so this cannot be
            # confused with a spent handle: the model has to re-issue the
            # call rather than wait for something that will never land.
            return BackgroundTask(
                handle=handle,
                state=TaskState.RUNNING,
                result=UNKNOWN_HANDLE,
                caller_id=None,
                timeout=0,
            )
        if record.state is TaskState.COLLECTED:
            return replace(record)
        task = self._tasks.get(handle)
        if task is None or not task.done():
            return replace(record)
        # Done, but the done-callback may not have run yet: callbacks are
        # scheduled with call_soon, not executed inline.  Settling here makes
        # the answer independent of callback timing.
        self._settle(handle, task)
        if record.state is TaskState.RUNNING:
            return replace(record)
        outcome = replace(record)
        # Spent.  Clearing the text is what makes a second call answer
        # COLLECTED rather than repeating an answer the model already has.
        record.state = TaskState.COLLECTED
        record.result = ""
        return outcome

    def pending_notifications(self) -> List[BackgroundTask]:
        """Return finished tasks whose notification has not been delivered.

        Returns:
            Copies of every record in a terminal state that has not been
            claimed by :meth:`mark_delivered` and whose result has not been
            spent through :meth:`collect` — a collected task has already told
            the model, so announcing it again would be noise.  Ordered by
            launch.
        """
        pending: List[BackgroundTask] = []
        for record in self._records.values():
            if record.state not in TERMINAL_STATES:
                continue
            if record.handle in self._delivered:
                continue
            pending.append(replace(record))
        return pending

    def mark_delivered(self, handle: str) -> None:
        """Record that *handle*'s notification has been sent.

        Idempotent, and a no-op for a handle this registry never issued — a
        late or duplicated call must not raise in a delivery path.

        Args:
            handle: The handle whose notification was delivered.
        """
        self._delivered.add(handle)
        record = self._records.get(handle)
        if record is not None:
            # The text has been read out; keeping it would only pin memory for
            # as long as the registry lives.
            record.result = ""

    async def dispatch_pending(self, ctx: "ToolContext") -> List[str]:
        """Deliver every owed notification whose caller is not running.

        **This is a coroutine, and it must be awaited.**  Delivery *awaits*
        ``orchestrator.send_message``, so calling this without ``await`` builds
        a coroutine object, runs nothing, and returns ``None`` instead of the
        handle list — the notification is silently never delivered and every
        parked task stays parked.  The failure is silent in the worst possible
        way: no exception, no log line, and the agent simply never hears about
        its finished work.  A caller that does ``registry.dispatch_pending(ctx)``
        without ``await`` has written a no-op that looks like a call.

        The design brief for this task spells the signature
        ``def dispatch_pending(self, ctx) -> list[str]``, without an ``async``
        marker, while its own prose says this method awaits
        ``send_message``, and its own test asserts on a list that only an
        awaited coroutine can populate.  Those three cannot all hold; this
        ``async`` is the one that can.  A synchronous alternative would have to
        fire the send with ``create_task``, which cannot report whether delivery
        succeeded and drops the notification outright if the loop is closing —
        strictly worse than the hazard described above.

        Skips, without marking delivered, anything that must not be sent yet:

        * a task launched with ``caller_id=None`` — there is no session to
          message, and it will never acquire one;
        * a context with no orchestrator attached;
        * a caller whose session has since been removed;
        * **a caller that is currently running.**  This is the case the whole
          method exists for.  ``send_message`` cancels a running session and
          awaits the cancelled task before appending its prompt, so delivering
          into one would destroy the turn that launched the background call in
          the first place.  Parking is cheap and the next dispatch retries.

        A handle is claimed *before* the await and un-claimed if the send
        raises, so two dispatches overlapping on one handle cannot both
        deliver it, and a failed send is retried rather than lost.

        Args:
            ctx: The context supplying the orchestrator to message through.
                Passed in rather than read from the module-level current
                context so the caller decides which context's orchestrator is
                used, and so this method is testable without global state.

        Returns:
            The handles actually delivered, in launch order — and only
            reachable if this coroutine was awaited; see the note at the top
            of this docstring.  Empty when everything was parked or already
            sent, which is a normal outcome and not an error.
        """
        delivered: List[str] = []
        for record in self.pending_notifications():
            caller_id = record.caller_id
            if caller_id is None or ctx is None or ctx.orchestrator is None:
                continue
            session = ctx.orchestrator.get_session(caller_id)
            if session is None:
                # The agent is gone.  Left pending rather than dropped: the
                # registry cannot tell "removed for now" from "removed for
                # good", and a stale notification is cheaper than a lost one.
                continue
            if _state_value(session.state) == RUNNING_STATE_VALUE:
                # Parked.  See the method docstring.
                continue
            text = NOTIFICATION_TEXT.format(
                handle=record.handle, state=record.state.value, result=record.result
            )
            self._delivered.add(record.handle)
            try:
                await ctx.orchestrator.send_message(caller_id, text)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # A failed delivery must not become a failed dispatch call:
                # this runs on an agent's turn, and an exception here would
                # take that turn down over a notification.
                self._delivered.discard(record.handle)
                logger.opt(exception=exc).error(
                    "Delivery of background notification for handle '{}' "
                    "failed; it stays pending and will be retried.",
                    record.handle,
                )
                continue
            self.mark_delivered(record.handle)
            delivered.append(record.handle)
        return delivered

    def cancel_all(self) -> None:
        """Cancel every task still running and record them as cancelled.

        Synchronous by design — it is called from teardown paths that cannot
        await — so the ``CancelledError`` is *scheduled* rather than delivered
        by the time it returns.  That is why the state is written here rather
        than left to the wrapper: a caller that reads a record immediately
        after must not see ``RUNNING`` for a task it just stopped, and must
        never see ``DONE``.

        Terminal tasks are left alone.  Cancelling them would be a no-op at
        best, and at worst would overwrite a failure the model has not been
        told about yet.
        """
        for handle, record in list(self._records.items()):
            if record.state is not TaskState.RUNNING:
                continue
            record.state = TaskState.CANCELLED
            record.result = CANCELLED_TEXT
            task = self._tasks.get(handle)
            if task is not None and not task.done():
                task.cancel()

    async def _run(self, record: BackgroundTask, coro_factory: CoroFactory) -> None:
        """Run *coro_factory* under its ceiling and settle *record*.

        Settling happens here, inside the coroutine, and not only in the
        done-callback: a callback cannot observe a cancelled task's outcome
        without re-raising the cancellation itself, and re-raising it from a
        callback is how a cancelled task gets reported as a success.

        Args:
            record: The record to settle.  Mutated in place; the caller's
                stored entry is the same object, which is what lets ``collect``
                read the outcome without going through the task.
            coro_factory: The factory handed to :meth:`launch`.
        """
        try:
            value = await asyncio.wait_for(
                coro_factory(), _seconds(record.timeout)
            )
        except asyncio.TimeoutError:
            # wait_for has already cancelled the inner coroutine and awaited
            # its cancellation, so "abandoned" is not a state this reaches.
            record.state = TaskState.TIMEOUT
            record.result = TIMEOUT_TEXT.format(timeout=record.timeout)
        except asyncio.CancelledError:
            # Recorded, then re-raised on purpose.  Returning normally here
            # would leave the task looking like it completed successfully with
            # a None result — the exact bug class `errors.py` documents for
            # the tool boundary, and the reason that module keeps
            # CancelledError out of its `except Exception`.
            record.state = TaskState.CANCELLED
            record.result = CANCELLED_TEXT
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Computed once and reused for both sinks, as `errors.py` does:
            # interpolating `exc` into either string is the thing that makes
            # the error path able to raise.
            message = safe_exception_message(exc)
            record.state = TaskState.FAILED
            record.result = f"Error: {type(exc).__name__}: {message}"
            logger.opt(exception=exc).error(
                "Background task '{}' raised; it is recorded as failed and "
                "never re-raised into the agent: {}: {}",
                record.handle,
                type(exc).__name__,
                message,
            )
        else:
            record.state = TaskState.DONE
            record.result = _render(value)

    def _on_done(self, handle: str, task: "asyncio.Task[None]") -> None:
        """Settle *handle* if its task ended without recording an outcome.

        Args:
            handle: The handle whose task just finished.
            task: The finished task, used only to read its disposition.
        """
        self._settle(handle, task)

    def _settle(self, handle: str, task: "asyncio.Task[None]") -> None:
        """Fill in a record whose wrapper never ran, and leave the rest alone.

        Reached only when the wrapper was cancelled before its first line
        executed, so the record is still ``RUNNING`` while its task is already
        finished.  Called from both the done-callback and ``collect`` because
        the callback is scheduled with ``call_soon`` and may not have run by
        the time a caller asks.

        Args:
            handle: The handle to reconcile.
            task: That handle's finished task.
        """
        record = self._records.get(handle)
        if record is None or record.state is not TaskState.RUNNING:
            return
        if task.cancelled():
            record.state = TaskState.CANCELLED
            record.result = CANCELLED_TEXT
            return
        error = task.exception()
        if error is None:
            # Unreachable while `_run` is the only thing creating tasks: it
            # always records.  Treated as success rather than left RUNNING so
            # a future caller cannot observe a permanently un-settled handle,
            # but the empty result is the honest reading of "the wrapper
            # returned without settling".
            record.state = TaskState.DONE
            record.result = ""
            return
        record.state = TaskState.FAILED
        record.result = f"Error: {type(error).__name__}: {safe_exception_message(error)}"


def _seconds(timeout_ms: int) -> float:
    """Convert a millisecond ceiling into the seconds ``wait_for`` wants.

    A floor at zero rather than a rejection.  ``asyncio.wait_for`` treats a
    zero ceiling as "expire immediately and cancel", which is a sane reading
    if one ever reaches here, and the alternative — letting a negative
    ``TimeoutError`` argument through — is not.  In practice
    :meth:`BackgroundRegistry.launch` has already replaced a falsy ``timeout``
    with :data:`DEFAULT_TIMEOUT_MS`, because a model that wrote ``timeout=0``
    almost certainly meant "no ceiling" rather than "fail instantly", and a
    zero budget would fail every call.  This floor is the defence behind that
    policy, not the policy itself.

    Args:
        timeout_ms: The ceiling as the model stated it.

    Returns:
        The ceiling in seconds, never negative.  Falls back to
        :data:`DEFAULT_TIMEOUT_MS` when the value is not a number at all.
    """
    try:
        return max(float(timeout_ms), 0.0) / 1000.0
    except (TypeError, ValueError):
        # A model can emit anything; a broken ceiling must not become a
        # crashed tool body.
        return DEFAULT_TIMEOUT_MS / 1000.0


def _render(value: Any) -> str:
    """Return ``str(value)``, or a placeholder when that raises.

    Delegates to :func:`~mitocode.core.tool.builtin.errors.safe_exception_message`
    for the failure case so this package has exactly one guarded ``str``
    rather than two that can drift.

    Args:
        value: Whatever the detached coroutine returned.  Built-in tools return
            ``str``, so this is a no-op in practice; the guard is there for a
            tool that returns something with a hostile ``__str__``.

    Returns:
        The rendered value, or a placeholder naming the type that raised.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return safe_exception_message(exc)


def _state_value(state: Any) -> str:
    """Return the ``"idle"``/``"running"``-style value behind *state*.

    Accepts an enum member, a bare string, or anything else, so a caller — or
    a test double — can supply either shape without this module having to know
    which.  Deliberately duck-typed: see the module docstring on why the
    agent package is never imported to answer this.

    Args:
        state: A session's ``state`` attribute.

    Returns:
        The enum's ``.value`` when there is one, the object itself when it is
        already a string, and an empty string otherwise — which never equals
        :data:`RUNNING_STATE_VALUE`, so an unrecognised state is treated as
        "not running" and the notification goes out.
    """
    value = getattr(state, "value", state)
    return value if isinstance(value, str) else ""
