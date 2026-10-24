"""The rule that decides whether a background notification may be delivered.

A detached call finishes, and the registry knows who asked for it.  Before it
can tell that agent, something has to answer one question about that agent's
session: **is it safe to write a prompt into it right now?**  That question is
answered here, in
:func:`~mitocode.core.tool.builtin.delivery_gate.park_reason`, and nowhere else.

The rule is small; the reasons for it are not, and the reasons are the whole
reason this module is separate from the registry.  A reader who wants to change
*what counts as safe* needs the argument for why it is written this way, and
that argument has nothing to do with timeouts, handles or task states — the
subjects of :mod:`~mitocode.core.tool.builtin.background`.  Splitting them means
neither file has to explain the other's concerns.

**The gate is "positively safe to deliver", not "not running".**  Those are
different questions, and only the first can be answered correctly for a value
this module has never heard of.  :data:`DELIVERABLE_STATE_VALUES` is an
allow-list: delivery happens when this build *recognises* the value and it is not
:data:`RUNNING_STATE_VALUE`, and never merely because a value failed to match.
Everything else parks.

**The asymmetry is why it is an allow-list and not a deny-list.**  The two
mistakes are not symmetric.  Delivering into a running caller makes
:meth:`~mitocode.core.agent.agent_orchestrator.AgentOrchestrator.send_message`
cancel that session and await the cancelled task before appending its prompt,
so the turn which launched the background call is destroyed mid-flight and its
work cannot be recovered.  Parking a caller whose state this build cannot
identify delays a notification that the agent can still collect for itself on
its next turn.  A missed notification is recoverable; a killed turn is not.

**``cancelled`` and ``error`` deliver, and that is a judgement this module
cannot make for the caller.**  :meth:`AgentOrchestrator.send_message` only
cancels a session that is running, so neither state can lose a turn and both
are safe to write into.  What is not settled here is whether a session that
ended in ``cancelled`` or ``error`` *should* receive a new user prompt at all:
the call is safe, but it appends to a session in a terminal state, and whether
that is desirable is a product decision.  Parking them instead would be a
permanent silent non-delivery for a case this module can read perfectly well,
which is the worse of the two mistakes; delivering them is the reversible one.
A reviewer who disagrees should change :data:`DELIVERABLE_STATE_VALUES` rather
than reach around it.

**The comparison reads ``session.state`` by its ``.value`` string rather than
against :class:`~mitocode.core.agent.agent_state.AgentState`.**  That is a
deliberate constraint, not a shortcut.  The dependency between the two packages
is one-way — ``agent_session`` imports ``tool_manager``, so the tool package must
never import the agent package at runtime — and ``TYPE_CHECKING`` cannot help,
because the value is needed at runtime to compare against.  :func:`_state_value`
accepts either an enum member or a bare string, so a test double needs no import
of the agent package at all.  Do not "fix" this by importing the enum.

**Drift stops delivery; it never starts it.**  :data:`RUNNING_STATE_VALUE` is a
duplicated literal that nothing in this repository checks, so it can drift from
the enum it mirrors.  Because the gate is an allow-list, that drift can only
ever park a notification, never deliver one — a caller in a renamed running
state is simply not on the list.  The cost is a stall, and
:func:`log_unrecognised_state` makes the stall visible so it is not mistaken for
a busy agent.  That log repeats on every dispatch for as long as the drift
persists; the repetition is a known, accepted trade, since the alternative —
throttling — would hide the one event a reader needs to see.

This module imports nothing from
:mod:`~mitocode.core.tool.builtin.background`, at runtime or otherwise.  The
registry imports *this*; the dependency is one-way, and a cycle here would put
back exactly the tangle the split exists to remove.
"""

from __future__ import annotations

from typing import Any, FrozenSet, Optional, Tuple

from loguru import logger as loguru_logger

logger = loguru_logger.bind(name=__name__)

#: Value of :attr:`mitocode.core.agent.agent_state.AgentState.RUNNING`, spelled
#: out here so this module never has to import the agent package to reason about
#: it.  **It is a duplicated literal that nothing in this repository checks.**
#: The package dependency is one-way — ``agent_session`` imports
#: ``tool_manager``, so the tool package may not import the agent package at
#: runtime — and ``TYPE_CHECKING`` cannot help, because the value is needed at
#: runtime.
#:
#: It is no longer the value compared against; :data:`DELIVERABLE_STATE_VALUES`
#: is.  It stays because it is the anchor a reader lands on when changing
#: ``AgentState``, and :func:`park_reason` re-checks it as a guard against a
#: future edit that adds the running value to the deliverable set.
#:
#: **If it drifts, the failure is a stalled notification, not a killed turn.**
#: Rename the member or change its value and the deliverable set keeps matching
#: the old strings, so an idle caller still receives its notification; a
#: *running* caller, whose new value is not on the set, has its notification
#: parked instead of delivered — and that park is logged rather than silent.
#: Nothing is destroyed: the result stays collectable.  Anything that changes
#: ``AgentState`` must update :data:`DELIVERABLE_STATE_VALUES`; a test is the
#: right home for that check and none exists yet.
RUNNING_STATE_VALUE = "running"

#: Every state value this build knows it can safely deliver into: each
#: :attr:`~mitocode.core.agent.agent_state.AgentState` value other than the
#: running one.  ``idle`` and ``finished`` are obvious; ``cancelled`` and
#: ``error`` are here because they are *identifiable and safe* —
#: ``send_message`` only cancels a session that is running, so neither can lose
#: a turn.  Parking them would mean a caller that ended in either state never
#: hears about the background task it launched, which is a permanent silent
#: non-delivery for a case this module can read perfectly well.
#:
#: This is an allow-list on purpose, and that is what keeps the gate
#: fail-closed: delivery happens on a value this build recognises, never on a
#: value merely *not equal to* :data:`RUNNING_STATE_VALUE`.  Drift therefore
#: stops delivery instead of starting it — the dangerous direction is the one
#: the rule is built to block.
DELIVERABLE_STATE_VALUES: FrozenSet[str] = frozenset(
    {"idle", "finished", "cancelled", "error"}
)

#: Park reason: the caller is mid-turn.  Expected on every completion of a task
#: the caller is still waiting on, so it is never logged — it is the whole
#: ordinary path, not an anomaly.
PARKED_RUNNING = "the caller is running"

#: Park reason: the caller's state could not be identified at all — an
#: unfamiliar string, a ``state`` attribute that is missing or not a string, or
#: no session object.  **This should never happen in normal operation**, which
#: is exactly why it is logged: a bare ``AgentState`` rename turns every
#: notification into a silent stall, and silence is what hides the drift.
PARKED_UNKNOWN_STATE = "the caller state could not be identified"

#: Park reason: ``get_session`` returned nothing for the caller.  Expected
#: while a session is being torn down, and cheap to re-check on the next
#: dispatch, so it is not logged.
PARKED_NO_SESSION = "the caller has no session"

#: Longest fragment of an unrecognised state value echoed into the drift log.
#: The value originates in our own enum rather than in a model response or a
#: socket, so it is not untrusted input — but :func:`_state_value` will hand
#: back *any* string a malformed session object carries, and a log line is the
#: last place that should be able to grow without bound.  Truncated, and shown
#: through :func:`repr` so a value containing quotes or newlines cannot forge
#: extra log lines.
_MAX_STATE_VALUE_CHARS = 60


def park_reason(session: Any) -> Optional[Tuple[str, str]]:
    """Return why *session* may not be delivered into, or ``None`` if it may.

    **The gate is "positively safe to deliver", not "not running".**  Those
    are different questions and the first is the one that can be answered
    correctly for a value this module has never heard of.  A state is
    deliverable when this build *recognises* it and it is not
    :data:`RUNNING_STATE_VALUE`; a value that cannot be identified at all is
    parked, because guessing that an unfamiliar value is safe is exactly how a
    renamed enum kills somebody's turn.

    The three park reasons are separated because only one of them is an
    anomaly.  A running caller is the ordinary case this deferral exists for;
    a missing session is a session mid-teardown.  An unidentifiable state
    should never occur, which is what makes it worth logging — see
    :data:`PARKED_UNKNOWN_STATE`.

    Args:
        session: Whatever ``orchestrator.get_session`` returned — a real
            session, a test double, or ``None``.

    Returns:
        ``None`` when the notification may be delivered, otherwise a
        ``(reason, state_value)`` pair.  *state_value* is the raw value read
        off the session, empty when there was none to read; it is returned
        rather than logged here so this function stays pure and the caller
        decides what is worth a log line.
    """
    if session is None:
        return (PARKED_NO_SESSION, "")
    value = _state_value(getattr(session, "state", None))
    if value == RUNNING_STATE_VALUE:
        return (PARKED_RUNNING, value)
    if value in DELIVERABLE_STATE_VALUES:
        return None
    # Guard, not redundancy: if a future edit adds the running value to the
    # deliverable set, every caller below that line starts having its turn
    # killed, and the set alone would not object.  Cheap enough to check on
    # every delivery, and it names the constant it is protecting.
    return (PARKED_UNKNOWN_STATE, value)


def log_unrecognised_state(handle: str, value: str) -> None:
    """Report a caller state this build cannot identify.

    The one park reason that should never happen, and the one that is invisible
    if ignored: an ``AgentState`` rename makes every notification park forever,
    and a park with no log line is indistinguishable from a caller that is
    simply still busy.  The line names the offending value and the handle so
    the drift can be traced to a specific edit.

    This repeats on every dispatch for as long as the drift persists.  That is
    a known, accepted trade: the alternative is throttling or de-duplicating,
    and both trade away the visibility of the one event a reader needs to see.

    Args:
        handle: The background handle whose notification was parked.
        value: The raw state value read off the session.  May be empty, or any
            string a malformed session object carries.
    """
    shown = value if value else "<missing, not a string, or not an enum>"
    # Bounded before it is formatted, not after: `_state_value` passes through
    # any string a session carries, and this value is expected to come from our
    # own enum but is not guaranteed to.
    if len(shown) > _MAX_STATE_VALUE_CHARS:
        shown = shown[:_MAX_STATE_VALUE_CHARS] + "... (truncated)"
    logger.error(
        "Parking the notification for background handle '{}': its caller's "
        "state is {}. Recognised and deliverable states are [{}]. This is "
        "drift, not a busy agent -- AgentState in "
        "mitocode.core.agent.agent_state and DELIVERABLE_STATE_VALUES in "
        "mitocode.core.tool.builtin.delivery_gate have diverged, or the session "
        "object is malformed. The notification is parked rather than lost: it "
        "is delivered on the first dispatch after the value is recognised, and "
        "the result stays collectable.",
        handle,
        repr(shown),
        ", ".join(sorted(DELIVERABLE_STATE_VALUES)),
    )


def _state_value(state: Any) -> str:
    """Return the ``"idle"``/``"running"``-style value behind *state*.

    Accepts an enum member, a bare string, or anything else, so a caller — or
    a test double — can supply either shape without this module having to know
    which.  Deliberately duck-typed: see the module docstring on why the
    agent package is never imported to answer this.

    Args:
        state: A session's ``state`` attribute, or ``None``.

    Returns:
        The enum's ``.value`` when there is one, the object itself when it is
        already a string, and an empty string otherwise.  The empty string is
        in no set here, so a missing or non-string state is unidentifiable and
        parks rather than delivering — see :func:`park_reason`.
    """
    value = getattr(state, "value", state)
    return value if isinstance(value, str) else ""
