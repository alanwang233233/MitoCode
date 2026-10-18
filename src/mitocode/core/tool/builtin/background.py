"""Detached tool calls, and the notification that arrives when one finishes.

A long call — a test suite, a build, a search over a large tree — blocks the
agent that asked for it.  :class:`BackgroundRegistry` is the escape hatch:
:meth:`BackgroundRegistry.launch` hands the call to an :mod:`asyncio` task,
returns a short handle immediately, and leaves the agent free to do something
else.  The handle is later redeemed with :meth:`BackgroundRegistry.collect`,
or the agent is told the answer arrived on its own.

Four decisions in here are load-bearing, and none of them is obvious from the
method bodies.

**The ceiling is a sibling task, not a wrapper around the work.**  The obvious
spelling — ``asyncio.wait_for(coro, timeout)`` around the detached call —
cannot be the discriminator here for two independent reasons.  On Python 3.11+
``asyncio.TimeoutError`` *is* the builtin ``TimeoutError``, so a socket or
subprocess timeout raised by the tool itself is indistinguishable from this
registry's deadline and would be recorded as a timeout it never hit.  And
``wait_for`` hands back whatever the inner coroutine produced when that
coroutine suppresses its own ``CancelledError`` — 3.10's implementation returns
``fut.result()`` after ``_cancel_and_wait``, and 3.11+'s ``Timeout.__aexit__``
only raises when the exception is ``CancelledError`` — so a task that blew its
ceiling could be reported as a success.

So :meth:`BackgroundRegistry._run` creates one task for the work and one for the
deadline and races them.  ``TIMEOUT`` is decided by *which future completed*,
never by the type of an exception, and it wins over anything the inner produced:
the ceiling is the registry's promise, not the callee's to extend.  One task for
the work, created once and never re-wrapped, so ``collect`` never sees a task it
has to re-timeout — it reads a state, it does not wait.  What this does **not**
promise is that the work has *stopped*: a coroutine that suppresses its own
cancellation, or that awaits inside ``asyncio.shield``, outlives the deadline,
and the registry records ``TIMEOUT`` rather than waiting on a cancellation it
cannot force.

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

**Delivery waits for the caller to be idle, and the rule that decides that
lives in :mod:`~mitocode.core.tool.builtin.delivery_gate`.**  This is the subtle
one.  :meth:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator.send_message`
*cancels* a session that is currently running and awaits the cancelled task
before appending the new prompt.  A completion callback that delivered
immediately would therefore interrupt the very turn that launched the
background call, discarding whatever that agent was in the middle of doing —
and it would do so on every single completion, not occasionally.  So
:meth:`BackgroundRegistry.dispatch_pending` asks that module whether each caller
may be delivered into and leaves the rest parked; the next dispatch, from
whoever notices the session went idle, delivers them.  Each handle is claimed
before the await and un-claimed if the send fails, so a dispatch interrupted
mid-flight cannot deliver the same notification twice.

The gate itself — which states may be delivered into, why the answer is
recognised-and-not-running rather than simply not-running, why an unrecognised
value parks *and is logged*, and what breaks if the duplicated state literal
drifts from :class:`~mitocode.core.agent.agent_state.AgentState` — is documented
in full in that module rather than here.  The short form is that delivery
happens only on a value this build positively recognises as safe, so drift in the
enum can stop a notification but can never start one into a live turn.

The session and its state are read live, inside the dispatch loop, on every
iteration — never captured at launch.  A caller that finished an hour ago must
still be delivered to, so there is nothing to invalidate.

For the same reason :class:`ToolContext` is imported here under
``TYPE_CHECKING`` only.  ``context.py`` imports this module at module level
(to build ``field(default_factory=BackgroundRegistry)``), so a runtime import
back would be a cycle.  :mod:`~mitocode.core.tool.builtin.delivery_gate` is
imported at module level instead, and imports nothing from here: that direction
is one-way for the same reason.
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

from mitocode.core.tool.builtin.delivery_gate import (
    PARKED_UNKNOWN_STATE,
    log_unrecognised_state,
    park_reason,
)
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

#: How many times a single handle's notification may fail to send before it is
#: retired from notification entirely.  Bounded because an orchestrator that
#: answers ``get_session`` but always raises from ``send_message`` would
#: otherwise produce an unbounded retry loop and an unbounded stream of log
#: lines.  Five is deliberately generous: transient causes (a session mid
#: teardown, a loop briefly busy) clear within a couple of dispatches, so the
#: cap only fires against a sink that is genuinely broken.
MAX_DELIVERY_ATTEMPTS = 5

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

    #: The ceiling fired and the inner work was asked to stop.  Guaranteed:
    #: the deadline future completed, and any value the inner produced anyway
    #: was discarded.  **Not** guaranteed: that the inner work actually
    #: stopped.  A coroutine that suppresses ``CancelledError`` or that awaits
    #: inside ``asyncio.shield`` keeps running past this point, and this state
    #: is recorded anyway — a late failure is logged, never promoted to a
    #: result.  Recorded instead of a fabricated ``DONE``, which is the whole
    #: point: exceeding the ceiling is a fact about the call, whatever the
    #: callee does with its cancellation.
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
        # Handles given up on after MAX_DELIVERY_ATTEMPTS failed sends.  Kept
        # apart from `_delivered` so `mark_delivered` can be told "this was
        # announced" from "this was retired" -- the difference decides whether
        # the stored result is cleared or preserved for a later `collect`.
        self._abandoned: Set[str] = set()
        # handle -> how many delivery attempts have failed.  Survives a
        # successful delivery so a handle that flaps does not reset its budget.
        self._attempts: Dict[str, int] = {}
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
            claimed by :meth:`mark_delivered` or retired by
            :meth:`_abandon`, and whose result has not been spent through
            :meth:`collect` — a collected task has already told the model, so
            announcing it again would be noise.  Ordered by launch.
        """
        pending: List[BackgroundTask] = []
        for record in self._records.values():
            if record.state not in TERMINAL_STATES:
                continue
            if record.handle in self._delivered or record.handle in self._abandoned:
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

        Skips, without marking delivered, anything not positively identifiable as
        safe to deliver into:

        * a task launched with ``caller_id=None`` — there is no session to
          message, and it will never acquire one;
        * a context with no orchestrator attached;
        * **a caller that is currently running.**  This is the case the whole
          method exists for.  ``send_message`` cancels a running session and
          awaits the cancelled task before appending its prompt, so delivering
          into one would destroy the turn that launched the background call in
          the first place.  Parking is cheap and the next dispatch retries.
        * a caller whose session has been removed;
        * **a caller in any state this build does not recognise** — an unknown
          string, a missing or non-string ``state``, a malformed session
          object.  These park *and are logged*
          (see :func:`~mitocode.core.tool.builtin.delivery_gate.log_unrecognised_state`),
          because they should never occur: a silent park is indistinguishable
          from a busy caller, so an ``AgentState`` rename would otherwise turn
          every notification into an invisible stall.

        What is *not* parked: a caller in ``cancelled`` or ``error``.
        ``send_message`` only cancels a session that is running, so neither can
        lose a turn, and
        :data:`~mitocode.core.tool.builtin.delivery_gate.DELIVERABLE_STATE_VALUES`
        names every state this build recognises as safe.  The gate is
        "positively safe to deliver", not "not running" — an identifiable, safe
        state delivers even when the session will never run again, though
        whether a terminal session *should* receive a new prompt is a judgement
        this module cannot make; see the gate module's docstring.

        Every one of those reads is **live**: the session and its state are
        fetched from the orchestrator inside the loop, on every call, and never
        captured when the task was launched.  A caller that finished an hour ago
        is therefore still delivered to on the next dispatch.

        A handle is claimed *before* the await and un-claimed on **every** way
        the await can end badly — a raising ``send_message`` and a cancelled
        ``dispatch_pending`` alike — so two dispatches overlapping on one
        handle cannot both deliver it, and a failed or interrupted send is
        retried rather than lost.  The ``CancelledError`` clause is not optional
        bookkeeping: without it, a dispatch cancelled mid-``await`` by session
        teardown or a user interrupt would leave the handle sitting in
        :attr:`_delivered`, which :meth:`pending_notifications` filters on, and
        the notification would be unreachable forever while the record still
        reads as undelivered.

        Retries are bounded by :data:`MAX_DELIVERY_ATTEMPTS`.  A
        ``send_message`` that always raises — an orchestrator torn down but
        still answering ``get_session`` — would otherwise be retried on every
        dispatch forever, one log line each time, for a handle that is never
        going to succeed.  After the cap the handle is retired from
        *notification* only: its record and result stay in :attr:`_records`, so
        ``collect`` still answers it and nothing the agent asked for is
        destroyed.  The cap is set high enough that it only ever fires against
        a genuinely broken sink, not against a transient one.

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
            # Read live, every iteration: the session and its state are fetched
            # here and not cached, so a caller that finished since the task
            # launched is seen as finished.  Nothing to invalidate.
            session = ctx.orchestrator.get_session(caller_id)
            parked = park_reason(session)
            if parked is not None:
                if parked[0] == PARKED_UNKNOWN_STATE:
                    # The only park that is an anomaly rather than the
                    # ordinary "not yet" case.  Logged so enum drift is
                    # visible; the other two are expected and stay silent.
                    log_unrecognised_state(record.handle, parked[1])
                # Parked, not dropped: the registry cannot tell "removed for
                # now" from "removed for good", and a stale notification is
                # cheaper than a lost one.
                continue
            text = NOTIFICATION_TEXT.format(
                handle=record.handle, state=record.state.value, result=record.result
            )
            self._delivered.add(record.handle)
            try:
                await ctx.orchestrator.send_message(caller_id, text)
            except asyncio.CancelledError:
                # `dispatch_pending` itself was cancelled while awaiting the
                # send — teardown, a user interrupt.  CancelledError is a
                # BaseException, so the `except Exception` below never sees it;
                # without this clause the claim taken above would survive it and
                # the handle would never be pending again.  Re-raised, never
                # absorbed: the cancellation is the caller's to act on, and
                # swallowing it would make an interrupted turn look completed.
                self._delivered.discard(record.handle)
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # A failed delivery must not become a failed dispatch call:
                # this runs on an agent's turn, and an exception here would
                # take that turn down over a notification.
                self._delivered.discard(record.handle)
                if self._count_attempt(record.handle) >= MAX_DELIVERY_ATTEMPTS:
                    self._abandon(record.handle)
                    continue
                logger.opt(exception=exc).error(
                    "Delivery of background notification for handle '{}' "
                    "failed; it stays pending and will be retried.",
                    record.handle,
                )
                continue
            self.mark_delivered(record.handle)
            delivered.append(record.handle)
        return delivered

    def _count_attempt(self, handle: str) -> int:
        """Record one delivery failure against *handle* and return the total.

        Args:
            handle: The handle whose delivery just failed.

        Returns:
            How many times delivery has now failed for this handle, counting
            the failure that just happened.
        """
        attempts = self._attempts.get(handle, 0) + 1
        self._attempts[handle] = attempts
        return attempts

    def _abandon(self, handle: str) -> None:
        """Stop notifying about *handle* after too many failed sends.

        Retires the handle from notification only.  The record itself stays
        in :attr:`_records` with its state and result intact, so an agent that
        still holds the handle can ``collect`` the outcome; what is given up is
        the automatic announcement, which is the part that cannot succeed.

        Args:
            handle: The handle to retire.
        """
        self._abandoned.add(handle)
        self._delivered.add(handle)
        logger.error(
            "Giving up on delivering the background notification for handle "
            "'{}' after {} failed attempts. The result is not discarded: "
            "collect('{}') still returns it.",
            handle,
            MAX_DELIVERY_ATTEMPTS,
            handle,
        )

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

        **The ceiling is a sibling task, not a wrapper around the work.**
        ``asyncio.wait_for`` cannot be the discriminator here, on two counts.
        First, on Python 3.11+ ``asyncio.TimeoutError`` *is* the builtin
        ``TimeoutError``, so a socket or subprocess timeout raised by the tool
        itself — common in exactly the work this registry exists to run — is
        indistinguishable from this registry's own deadline and would be
        recorded as :attr:`TaskState.TIMEOUT` carrying a fabricated "timed out
        after N ms".  The two environments then disagree about the same code,
        because on 3.10 the same exception arrives as a plain ``TimeoutError``
        and lands in the ordinary failure branch.

        Second, and worse, ``wait_for`` returns whatever the inner coroutine
        produced when that coroutine suppresses its own ``CancelledError``:
        3.10's implementation calls ``fut.result()`` after
        ``_cancel_and_wait``, and 3.11+'s ``Timeout.__aexit__`` only raises when
        the exception is ``CancelledError``.  A callee that catches its
        cancellation and returns normally therefore makes a task that blew its
        ceiling look like a success — the one outcome this module must never
        produce.

        So the deadline is expressed as its own future racing the work.
        :attr:`TaskState.TIMEOUT` is chosen when *that* future completed, and
        it wins over anything the inner produced, because the ceiling is the
        registry's promise and not the callee's to extend.  A tool's own
        ``TimeoutError`` is just an exception on the inner future and is
        reported as :attr:`TaskState.FAILED` on every supported interpreter.

        What this does **not** guarantee: that the inner work has stopped.  A
        coroutine that suppresses ``CancelledError``, or that awaits inside
        ``asyncio.shield``, keeps running after the deadline fires, and the
        registry records ``TIMEOUT`` anyway rather than waiting for a
        cancellation it cannot force.  :meth:`_drain` consumes a late exception
        from such a survivor so asyncio does not print an orphan warning about
        it.

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
        inner = asyncio.ensure_future(coro_factory())
        expired = False

        async def _expire() -> None:
            """Arm the ceiling: record that it fired, then stop the work."""
            nonlocal expired
            await asyncio.sleep(_seconds(record.timeout))
            expired = True
            # Requested, not guaranteed.  A callee that suppresses its own
            # cancellation keeps going; TIMEOUT is recorded regardless.
            inner.cancel()

        deadline = asyncio.ensure_future(_expire())

        def _drain(survived: "asyncio.Future[Any]") -> None:
            """Consume a late outcome from work that outlived its ceiling.

            ``shield``, or a bare ``except asyncio.CancelledError`` in the tool,
            lets the inner future outlive this coroutine.  If it then raises,
            nobody is left to retrieve the exception and asyncio answers that
            with a "Task exception was never retrieved" line on stderr that
            nothing here asked for.  Retrieving it here also gets it into the
            log, where an operator can actually see it.

            Args:
                survived: The finished inner future.
            """
            if survived.cancelled():
                return
            error = survived.exception()
            if error is None:
                return
            if record.state is TaskState.RUNNING:
                # `_run` has not settled yet; it reads the same exception.
                return
            logger.opt(exception=error).error(
                "Background task '{}' exceeded its {} ms ceiling and the work "
                "it could not stop then raised anyway: {}: {}",
                record.handle,
                record.timeout,
                type(error).__name__,
                safe_exception_message(error),
            )

        inner.add_done_callback(_drain)
        try:
            await asyncio.wait({inner, deadline}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # The wrapper itself was cancelled — `cancel_all`, shutdown, the
            # interpreter going away.  Recorded, the inner work asked to stop,
            # then re-raised on purpose.  Returning normally here would leave
            # the task looking like it completed successfully with a None
            # result — the exact bug class `errors.py` documents for the tool
            # boundary, and the reason that module keeps CancelledError out of
            # its `except Exception`.
            record.state = TaskState.CANCELLED
            record.result = CANCELLED_TEXT
            inner.cancel()
            raise
        finally:
            if not deadline.done():
                # The work won the race; disarm the ceiling so it cannot fire
                # afterwards and cancel an already-finished task.
                deadline.cancel()

        if expired:
            # The ceiling fired, so TIMEOUT wins over anything the inner
            # produced — including a value it returned after suppressing the
            # cancellation it was just handed.
            record.state = TaskState.TIMEOUT
            record.result = TIMEOUT_TEXT.format(timeout=record.timeout)
            inner.cancel()
            return

        if inner.cancelled():
            record.state = TaskState.CANCELLED
            record.result = CANCELLED_TEXT
            return

        error = inner.exception()
        if error is not None:
            # Read off the inner future rather than caught from an `except`,
            # which is what keeps a tool's own TimeoutError on the FAILED
            # side of the ledger on every supported interpreter.
            message = safe_exception_message(error)
            record.state = TaskState.FAILED
            record.result = f"Error: {type(error).__name__}: {message}"
            logger.opt(exception=error).error(
                "Background task '{}' raised; it is recorded as failed and "
                "never re-raised into the agent: {}: {}",
                record.handle,
                type(error).__name__,
                message,
            )
            return

        record.state = TaskState.DONE
        record.result = _render(inner.result())

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
