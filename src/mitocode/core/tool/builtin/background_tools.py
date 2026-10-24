"""The ``Background`` and ``BackgroundResult`` tools.

A long call — a test suite, a build, a large search — blocks the agent that
asked for it for as long as it runs.  ``Background`` is the escape hatch: it
takes one line of the tool-call mini-DSL, ``Bash("python -m pytest test/")``,
hands the parsed call to
:class:`~mitocode.core.tool.builtin.background.BackgroundRegistry` as a
detached task, and returns a ``bg_<n>`` handle without waiting.
``BackgroundResult`` redeems the handle later, and the registry's delivery
path (see :mod:`~mitocode.core.tool.builtin.background`) announces the
outcome on its own once the calling agent goes idle.

Three decisions in here are load-bearing, and none of them is obvious from
the method bodies.

**``Background`` never blocks, and its ``timeout`` is a ceiling, not a
wait.**  There is no code path that awaits the detached call.  The tool
returns the handle the instant the registry schedules the task, which is the
whole reason ``BackgroundResult`` exists: a model that needs the answer asks
for it with a second call instead of holding its turn open.  ``timeout``
therefore means "never let the work run longer than this" — it bounds the
detached task, not this call.  The value is normalised here, in the tool,
before it reaches the registry (see :func:`_resolve_timeout` and
:data:`MAX_TIMEOUT_MS`): a ceiling of ``inf`` — what ``1e400`` parses to —
would be a promise never to interrupt, which is the exact hang the ceiling
exists to prevent, so a non-finite value is refused outright and a merely
absurd finite one is clamped to one hour.

**The registry is the allowlist.**  ``parsed.tool_name`` comes from
model-authored text and is resolved with exactly one operation:
``ctx.tool_manager.get_spec(name)`` — a dict lookup.  ``parse_call``
happily yields ``tool_name == "__import__"`` for ``__import__("os")``; that
is inert because ``__import__`` is not a registered tool and nothing in this
module ever resolves a name any other way — no ``getattr``, no ``eval``, no
``globals()``.  A second, name-based allowlist here would create a second
source of truth that could drift from the registry, so there is none: a name
is dispatchable exactly when the registry has it.

**Every failure is prose, and nothing is dispatched when validation
fails.**  A malformed call, an unknown tool, a missing argument, a bad
timeout, a missing tool manager — all are returned as ``"Error: ..."``
lines naming what went wrong, before
:meth:`~mitocode.core.tool.builtin.background.BackgroundRegistry.launch` is
ever reached, so a refused call has no side effects at all.  Anticipated
failures are returned rather than raised on purpose: the boundary renders
every escaped exception as ``"Error: <TypeName>: <message>"``, and a Python
class name is not a sentence the model can act on.  The boundary keeps that
format for the genuinely unexpected, which is what it is there for.

The result of the detached call is coerced with ``str()`` inside the factory
handed to the registry: the registry stores a string, and
:class:`~mitocode.core.agent.agent_runtime.AgentRuntime` renders tool output
with ``str(result)``, so the coercion happens where the value is born rather
than wherever it is read later.  A result is collectable exactly once — the
registry drops the text after the first redeem, and :func:`background_result`
renders "already collected" differently from "unknown handle", because the
two call for different responses from the model: the first means the work ran
and the text is gone, the second means the handle is wrong and the call must
be made again.
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Optional, Tuple

from mitocode.core.tool.builtin.background import (
    DEFAULT_TIMEOUT_MS,
    HANDLE_PREFIX,
    UNKNOWN_HANDLE,
    BackgroundRegistry,
    BackgroundTask,
    TaskState,
)
from mitocode.core.tool.builtin.call_dsl import bind_arguments, parse_call
from mitocode.core.tool.builtin.context import (
    clamp,
    current,
    current_agent_context,
)
from mitocode.core.tool.builtin.errors import (
    MitoCodeBuiltinToolError,
    tool_boundary,
)

#: Largest ceiling :func:`background` will hand to the registry, in
#: milliseconds.  One hour: longer than any single tool call this harness
#: expects to host, short enough that a wedged call cannot pin an agent
#: session open past the working day.  The bound lives here, in the tool,
#: rather than in the registry, because this is the first place a
#: model-supplied number is believed: the registry's own fallback covers only
#: a *falsy* timeout, and a value that arrives at ``launch`` unbounded is a
#: promise this module already had one chance to bound and did not take.
MAX_TIMEOUT_MS = 3_600_000

#: Longest fragment of an untrusted name echoed back in an error message.
#: ``parsed.tool_name`` is model-authored and unbounded, and a 50 000
#: character identifier has no business reaching the transcript whole.
_MAX_ECHO = 60

#: What :func:`background` answers when no tool manager is attached.
#:
#: A shared constant rather than an inline sentence, matching the practice of
#: ``agent_tools``: the wording must not be able to drift between call sites,
#: and the three things the model needs — what is missing, that nothing was
#: dispatched, and that retrying will not help — are exactly what it names.
NO_TOOL_MANAGER = (
    "Error: no tool manager is attached to the tool context, so Background "
    "has no registry to resolve the requested call against and nothing was "
    "dispatched. This is a wiring problem rather than a bad request: attach "
    "one during startup so ToolContext.tool_manager is set. Retrying will "
    "not help until it is attached."
)

#: Recognises one handle inside a longer piece of model-supplied text.
#: :data:`~mitocode.core.tool.builtin.background.HANDLE_PREFIX` is escaped so
#: a future prefix change cannot quietly turn this into a different pattern.
_HANDLE_PATTERN = re.compile(rf"{re.escape(HANDLE_PREFIX)}\d+")


async def _collect(registry: BackgroundRegistry, handle: str) -> BackgroundTask:
    """Redeem *handle*, falling back to the handle embedded in reply text.

    A model frequently hands ``BackgroundResult`` back the whole sentence
    ``Background`` replied with rather than the bare ``bg_<n>`` token: both
    reach the tool as one string, and the transcript gives the model no way
    to tell them apart.  When the raw argument misses, the first
    ``bg_<n>`` token in it is retried — it is the handle that reply was
    about.  The fallback only changes the *key* of the registry's dict
    lookup, which stays exact; nothing is dispatched and no other behaviour
    moves.

    Args:
        registry: The registry to redeem from.
        handle: The raw model-supplied argument, possibly a full reply
            sentence rather than a bare handle.

    Returns:
        The record for the raw argument when it resolves, otherwise the
        record for the first embedded handle when there is one, otherwise
        the raw argument's unknown-handle record.
    """
    record = await registry.collect(handle)
    if not (record.state is TaskState.RUNNING and record.result == UNKNOWN_HANDLE):
        return record
    text = handle if isinstance(handle, str) else str(handle)
    match = _HANDLE_PATTERN.search(text)
    if match is None or match.group() == text:
        # "bg_42" handed back whole: the retry would be the identical
        # lookup, and the raw record is already the answer to render.
        return record
    return await registry.collect(match.group())


def _clip(text: Any) -> str:
    """Render untrusted *text* for inclusion in an error message, bounded.

    A local sibling of ``call_dsl._clip_name`` rather than a reuse of it:
    that helper is private to its module, and this one is reached with a
    model-authored tool name that may be any string at all.  Total, like the
    helpers it is modelled on: it never raises, so it stays safe inside the
    error paths that call it.

    Args:
        text: The offending value.  Anything without the string protocol
            renders as its type name instead of being measured.

    Returns:
        *text* unchanged when it is a string within :data:`_MAX_ECHO`,
        otherwise its first ``_MAX_ECHO`` characters plus a count of the
        remainder, or a ``<TypeName>`` marker for a non-string.
    """
    if not isinstance(text, str):
        return f"<{type(text).__name__}>"
    if len(text) <= _MAX_ECHO:
        return text
    return f"{text[:_MAX_ECHO]}... (+{len(text) - _MAX_ECHO} more characters)"


def _resolve_timeout(timeout: Any) -> Tuple[int, Optional[str]]:
    """Normalise the model-supplied ceiling before it reaches the registry.

    **The rule, and where the bound lives.**  The bound is
    :data:`MAX_TIMEOUT_MS` — one hour — and it lives in this module because
    this is the first place a model-supplied number is believed: the
    registry's fallback catches only a *falsy* timeout, and by the time a
    value reaches ``launch`` unbounded the chance to bound it is gone.
    Concretely:

    * ``None`` becomes :data:`~mitocode.core.tool.builtin.background.DEFAULT_TIMEOUT_MS`
      (five minutes), the same default the registry applies to a falsy
      timeout.
    * A non-number, or a boolean — ``True`` is an ``int`` in Python, and a
      one-millisecond ceiling is never what a model sending ``true`` meant —
      is refused.
    * A non-finite float is refused outright.  ``1e400`` parses to ``inf``,
      and an infinite ceiling is a promise never to interrupt: the exact hang
      the ceiling exists to prevent.
    * A finite value at or below zero becomes the default, matching the
      registry's documented treatment of a non-positive timeout.
    * A finite value above :data:`MAX_TIMEOUT_MS` is *clamped* to it, with a
      note in the launch reply saying so, because the request is still
      meaningful — only its magnitude is not.

    Args:
        timeout: The ceiling as the model stated it, in milliseconds.
            Typically an ``int``; eventually anything.

    Returns:
        ``(timeout_ms, note)``: the effective ceiling in milliseconds, and a
        prose sentence to append to the launch reply when the requested value
        had to be clamped (``None`` otherwise).

    Raises:
        MitoCodeBuiltinToolError: When the value is not a usable number at
            all — a non-number, a boolean, or a non-finite float.  The
            caller converts this into ``"Error: ..."`` prose rather than
            letting it escape.
    """
    if timeout is None:
        return DEFAULT_TIMEOUT_MS, None
    if isinstance(timeout, bool):
        raise MitoCodeBuiltinToolError(
            "timeout must be a number of milliseconds; got the boolean " f"{timeout}."
        )
    if isinstance(timeout, int):
        # Every int is finite by construction, so no finiteness check here;
        # an absurd magnitude is handled by the clamp below.
        value = timeout
    elif isinstance(timeout, float):
        # `math.isfinite` on an int converts to float and overflows for very
        # large integers, which is why the finiteness check is deliberately
        # float-only and reached through this branch alone.
        if not math.isfinite(timeout):
            raise MitoCodeBuiltinToolError(
                f"timeout must be a finite number of milliseconds; got "
                f"{timeout}. An infinite ceiling would never fire, which is "
                "the hang the timeout exists to prevent. State a concrete "
                "ceiling such as timeout=300000."
            )
        value = int(timeout)
    else:
        raise MitoCodeBuiltinToolError(
            "timeout must be a number of milliseconds; got "
            f"{type(timeout).__name__}."
        )

    if value <= 0:
        # "No ceiling" is what a zero or negative value plausibly means, and
        # the default is what the registry applies to a falsy one.
        return DEFAULT_TIMEOUT_MS, None
    if value > MAX_TIMEOUT_MS:
        return MAX_TIMEOUT_MS, (
            f"The requested ceiling of {value} ms was clamped to "
            f"{MAX_TIMEOUT_MS} ms (one hour), the largest ceiling Background "
            "grants."
        )
    return value, None


@tool_boundary
async def background(call: str, timeout: int) -> str:
    """Detach ``ToolName("arg")`` as a background task; return its handle.

    This is the ``Background`` tool.  It **never blocks**: there is no wait
    anywhere in the body.  ``timeout`` is the ceiling the detached call is
    held to — the registry stops the work when it fires — and not a wait
    this tool performs, which is the whole reason :func:`background_result`
    exists.

    The call text is parsed by
    :func:`~mitocode.core.tool.builtin.call_dsl.parse_call`, resolved against
    the tool registry with one exact lookup, and bound by
    :func:`~mitocode.core.tool.builtin.call_dsl.bind_arguments` before
    anything is dispatched, so a refused call has no side effects at all.

    Args:
        call: One tool call in the mini-DSL, e.g. ``Bash("ls -la")``.  The
            name must match a registered tool exactly.
        timeout: Ceiling for the detached call, in milliseconds.  Normalised
            by :func:`_resolve_timeout` before it reaches the registry.

    Returns:
        A line starting with the fresh ``bg_<n>`` handle, naming the detached
        tool and the effective ceiling.  When the call has no owning agent
        turn, the reply says automatic notification is unavailable and points
        at :func:`background_result`.  Every failure — a missing tool
        manager, a malformed call, an unknown tool, a missing argument, a
        refused timeout — is an ``"Error: ..."`` string naming what went
        wrong, and dispatches nothing.
    """
    caller_id, ctx = current_agent_context()
    manager = ctx.tool_manager
    if manager is None:
        return NO_TOOL_MANAGER

    try:
        parsed = parse_call(call)
        spec = manager.get_spec(parsed.tool_name)
        if spec is None:
            return (
                f"Error: unknown tool '{_clip(parsed.tool_name)}': no tool "
                "is registered under that exact name, so the call was "
                "refused and nothing was dispatched. Tool names are resolved "
                "by exact match; check the registered name and re-issue the "
                "call."
            )
        bound = bind_arguments(spec, parsed)
        timeout_ms, note = _resolve_timeout(timeout)
    except MitoCodeBuiltinToolError as exc:
        return f"Error: {exc}"

    async def _detached() -> str:
        """Run the resolved tool and render its outcome as text.

        Returns:
            The tool's return value coerced with ``str()``: the registry
            stores a string, and the runtime renders tool output with
            ``str(result)``, so the coercion happens here where the value is
            born rather than wherever it is read later.
        """
        outcome = spec.func(**bound)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return str(outcome)

    handle = ctx.background.launch(_detached, caller_id=caller_id, timeout=timeout_ms)

    if caller_id is None:
        tail = (
            "No agent turn is associated with this call, so automatic "
            "notification is unavailable; use "
            f'BackgroundResult("{handle}") to check on it and collect the '
            "result."
        )
    else:
        tail = (
            "You will be notified here automatically when it finishes, and "
            f'BackgroundResult("{handle}") returns the result sooner if you '
            "ask."
        )
    reply = (
        f"{handle}: detached '{_clip(parsed.tool_name)}' with a "
        f"{timeout_ms} ms ceiling; the call was not waited on. {tail}"
    )
    if note is not None:
        reply = f"{reply} {note}"
    return reply


@tool_boundary
async def background_result(handle: str) -> str:
    """Redeem one background handle and render what is known about it.

    This is the ``BackgroundResult`` tool.  Like ``Background`` it does not
    wait: a task still in flight is reported as running and the model asks
    again, because waiting here would reintroduce the blocking the registry
    exists to remove, on the one path a model uses to check on its work.

    Args:
        handle: The ``bg_<n>`` string :func:`background` returned — or the
            whole sentence that carried it, which :func:`_collect` resolves
            to the same handle by exact lookup when the raw text misses.

    Returns:
        One distinct rendering per lifecycle state: running; done, with the
        output; failed, with the registry's rendered error; timed out;
        cancelled; already collected — the handle was spent; and unknown —
        the handle was never issued.  "Spent" and "never existed" are
        deliberately different answers, because they call for different
        responses from the model: the first means the work ran and the text
        is gone, the second means the handle is wrong and the call must be
        re-issued.
    """
    ctx = current()
    record = await _collect(ctx.background, handle)
    state = record.state

    if state is TaskState.RUNNING:
        if record.result == UNKNOWN_HANDLE:
            body = (
                f'Error: unknown handle "{_clip(str(record.handle))}": no '
                "background task with that handle was issued, so there is "
                "nothing to collect and waiting will not change that. "
                f"Handles look like {HANDLE_PREFIX}3 and are returned by "
                "Background; re-issue the original call with Background if "
                "the handle was lost."
            )
        else:
            body = (
                f"{record.handle} is still running. BackgroundResult does "
                "not wait; ask again later."
            )
    elif state is TaskState.DONE:
        body = (
            f"{record.handle} finished successfully:\n"
            f"{clamp(record.result, ctx.max_output_chars)}"
        )
    elif state is TaskState.FAILED:
        body = (
            f"{record.handle} failed:\n" f"{clamp(record.result, ctx.max_output_chars)}"
        )
    elif state is TaskState.TIMEOUT:
        body = (
            f"{record.handle} {record.result} and produced no result. "
            "Re-issue the call with a larger timeout if it needs more time."
        )
    elif state is TaskState.CANCELLED:
        body = (
            f"{record.handle} was {record.result}; its result will not "
            "arrive. Re-issue the call if the work is still wanted."
        )
    elif state is TaskState.COLLECTED:
        body = (
            f"{record.handle} was already collected: its result was handed "
            "over once and then dropped, so asking again cannot return it. "
            "This is not an unknown handle — the task ran and was read. "
            "Re-issue the original call with Background if you need the "
            "output again."
        )
    else:
        # A TaskState member this build predates.  Rendered rather than
        # raised so the boundary's type-name format stays reserved for
        # genuinely unexpected failures.
        body = (
            f"Error: background task {record.handle} reports the "
            f"unrecognised state '{state.value}', so there is nothing "
            "sensible to report."
        )
    return body
