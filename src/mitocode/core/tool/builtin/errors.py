"""The single error boundary that every built-in tool returns through.

Eleven tools will each fail in their own way — a file that vanished between
the path check and the open, a regex the model mis-typed, a subprocess that
died on a signal.  ``AgentRuntime`` awaits a tool, does ``str(result)``, and
feeds the result straight back to the model: it has no ``try``/``except``
around the call, so an exception escaping a tool is not a helpful transcript
entry but an unhandled crash that loses the turn.  One decorator placed on
each tool turns every such escape into prose the model can react to, and
because it is one decorator the rendering cannot drift between tools.

Two design decisions are load-bearing and neither is obvious from the code:

**Cancellation is not an error.**  ``AgentRuntime`` stops an agent by
cancelling the task the tool is running in, and ``asyncio`` delivers that as
``CancelledError``.  Rendering it as ``"Error: CancelledError: ..."`` would
let a cancelled turn run to completion and then report a result, which is
exactly the bug the boundary exists to prevent.  It is safe to write a plain
``except Exception`` because, since Python 3.8, ``asyncio.CancelledError``
inherits from ``BaseException`` and not from ``Exception`` — verified on this
repository's interpreter (CPython 3.10.20) and unchanged through 3.14.  The
same property keeps ``KeyboardInterrupt`` and ``SystemExit`` out.  It is not
belt-and-braces: ``test_builtin_context.py`` asserts the hierarchy directly,
so a future interpreter change fails a test instead of silently breaking
cancellation.

**Nothing is swallowed silently.**  Every caught exception is logged through
loguru with its traceback *before* it is rendered, so an operator reading the
log can reconstruct what happened even though the model only ever saw one
line.  The traceback is attached with ``opt(exception=exc)`` rather than
formatted into the message, and that choice is load-bearing: loguru formats a
message's ``{}`` arguments **eagerly, inside the ``.error()`` call**, so
``logger.exception("...: {}", exc)`` calls ``str(exc)`` at precisely the point
where a raising ``__str__`` would defeat the boundary — and takes the log line
down with it.  Handing loguru the exception object instead leaves rendering to
loguru's own formatter, which is defensive about a failing ``__str__``: it
prints ``<exception str() failed>`` and still emits every frame.

**The boundary is total with respect to ``Exception``.**  Not just the tool's
body is protected but the error path itself.  Some exceptions raise from
``__str__`` — a proxy object, or one whose ``__str__`` reads an attribute that
is already ``None`` — and an f-string over such an exception raises *while the
decorator is already unwinding*, turning a caught error into an escaping one:
the exact failure this module exists to prevent, plus a lost log line.  So the
message is obtained through a guarded helper, and when it cannot be rendered
the type prefix — the part a reader, human or model, can actually branch on —
still reaches the model intact.  The guarantee stops precisely at
``BaseException``: if a broken ``__str__`` raises ``KeyboardInterrupt``,
``SystemExit`` or ``CancelledError``, that halt signal still propagates, because
swallowing one is the failure mode the paragraph above exists to prevent.

The rendered form is ``"Error: <Type>: <message>"`` — the exception *type* is
kept because "Error: something went wrong" tells the model nothing it can
branch on, while ``"Error: FileNotFoundError: ..."`` tells it whether retrying
could plausibly help.

:class:`MitoCodeBuiltinToolError` is the one exception in this package that
signals a *programming* fault rather than a bad argument — "no orchestrator
attached", "unknown tool name in the mini-DSL" — states that the wiring is
incomplete.  It is deliberately **not** exempted from the boundary: a tool
that raises it should still answer the model with prose, because leaving the
turn to crash over a wiring mistake helps nobody.  It exists as a type so the
cause is identifiable in a log.
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable

from loguru import logger as loguru_logger

from mitocode.core.exception.mitocode_exception import MitoCodeBaseError

logger = loguru_logger.bind(name=__name__)

#: Callable shape of a built-in tool before the boundary is applied.
ToolFunction = Callable[..., Awaitable[Any]]


class MitoCodeBuiltinToolError(MitoCodeBaseError):
    """Raised when a built-in tool cannot proceed for a wiring-level reason.

    Distinguishes "the model asked for something impossible" (an ordinary
    ``"Error: ..."`` return string) from "this code path was never wired up"
    — no orchestrator attached for an agent tool, a tool name the mini-DSL
    parser produced that is not in the registry, a background handle that was
    never issued.  The second kind is a bug in MitoCode rather than a
    mistake in the request, so it deserves its own type in a log even though
    the boundary renders it to the model exactly like everything else.

    Inherits :class:`~mitocode.core.exception.mitocode_exception.MitoCodeBaseError`,
    so the existing top-level error handling already catches it.
    """


#: Rendered in place of an exception message that cannot be produced.  Names the
#: type ``__str__`` raised rather than saying only "unavailable", because the
#: inner failure is usually the more diagnostic half of the story: an
#: ``AttributeError: 'NoneType' object has no attribute 'command'`` explains a
#: broken proxy exception far better than the proxy's own type does.
UNPRINTABLE_MESSAGE = "<message unavailable: __str__ raised {}>"


def safe_exception_message(exc: BaseException) -> str:
    """Return ``str(exc)``, or a placeholder when rendering the exception fails.

    Some exception types raise from ``__str__`` itself.  Calling it would then
    raise *inside the boundary's own error path* — while the decorator is
    already unwinding — which defeats the one guarantee this module makes and
    takes the log line with it, because loguru formats message arguments
    eagerly at the call site.  Every place that needs the message goes through
    here instead of interpolating ``exc`` directly.

    A legitimate empty message is not treated as a failure: ``str(KeyError())``
    is ``""``, and the caller still renders the trailing ``": "``, because
    dropping it would make an empty message indistinguishable from a malformed
    line.

    Args:
        exc: The exception whose message is wanted.

    Returns:
        ``str(exc)`` when it can be rendered, otherwise
        :data:`UNPRINTABLE_MESSAGE` naming the type ``__str__`` raised.

    Raises:
        BaseException: Propagated on purpose, and only this one.  If ``__str__``
            raises ``KeyboardInterrupt``, ``SystemExit`` or
            ``asyncio.CancelledError`` then a halt signal was raised while this
            helper was running, and masking it because a broken ``__str__``
            fired mid-log is the same class of bug as swallowing a cancellation.
            Every ordinary ``Exception`` — the realistic case, a broken proxy's
            ``AttributeError`` — is contained.
    """
    try:
        return str(exc)
    except Exception as inner:  # pylint: disable=broad-exception-caught
        # Broad on purpose and scoped to one str() call: the failure being
        # defended against is precisely that this raises something
        # unpredictable, so naming the type is the only way to keep going.
        return UNPRINTABLE_MESSAGE.format(type(inner).__name__)


def tool_boundary(func: ToolFunction) -> ToolFunction:
    """Wrap a tool coroutine so no exception escapes as a traceback.

    Expected failures already return ``"Error: ..."`` strings from the tool
    body; this catches everything else and renders ``"Error: <Type>: <message>"``,
    logging the trace to loguru first so nothing is silently lost.

    The guarantee covers the error path as well as the tool body.  Rendering
    goes through :func:`safe_exception_message` and the log line is attached
    with ``opt(exception=exc)`` rather than by interpolating the exception into
    the message, so an exception whose ``__str__`` raises still reaches the
    model as prose *and* still reaches loguru with its traceback.  The one thing
    deliberately still escaping is a ``BaseException`` raised by ``__str__``
    itself; see :func:`safe_exception_message`.

    Only ``Exception`` is caught.  ``asyncio.CancelledError`` derives from
    ``BaseException`` on Python 3.8 and later — verified on this repository's
    CPython 3.10.20 — so a bare ``except Exception`` already lets cancellation
    through untouched.  That is deliberate and load-bearing: ``AgentRuntime``
    stops a turn by cancelling the tool's task, and swallowing that here
    would make a cancelled agent uncancellable.  Do not widen this to
    ``BaseException`` and do not add a ``CancelledError`` re-raise.

    The return type is ``Any`` rather than ``str``: the decorator does not
    convert a tool's result, it only replaces an escaped exception.  Every
    built-in tool in this package returns ``str``, but saying so here would
    force every future caller through a cast for no gain.

    Args:
        func: The tool coroutine to guard.  Any ``async def`` whose result is
            intended to reach the model as-is.

    Returns:
        A coroutine function with the same call signature and the same
        ``__name__``/``__doc__``/signature as *func* — ``functools.wraps``
        matters because ``ToolManager.register`` derives a tool's JSON Schema
        from its annotations, and it reads them off the decorated object.
    """

    @functools.wraps(func)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        """Await *func* and convert any escaped exception into an error string.

        Args:
            *args: Positional arguments forwarded verbatim to *func*.
            **kwargs: Keyword arguments forwarded verbatim to *func*.

        Returns:
            Whatever *func* returns, or ``"Error: <Type>: <message>"`` if it
            raised.  The message is whatever
            :func:`safe_exception_message` could obtain; the type prefix is
            present either way.
        """
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Broad on purpose, and it is the whole point of this decorator:
            # eleven tools must not be able to drift on which exception types
            # they report, and "every Exception becomes prose" is the
            # contract.  Narrowing it would put the model back at the mercy
            # of whatever a filesystem or a subprocess raised today.
            #
            # Not caught, because they are BaseException subclasses and this
            # clause names Exception: asyncio.CancelledError, KeyboardInterrupt
            # and SystemExit.  The first is how a turn is stopped.
            exc_name = type(exc).__name__
            # Computed once and reused for both sinks.  Interpolating `exc`
            # into either string is what made this path unsafe: an exception
            # whose __str__ raises would then raise here, mid-unwind, and the
            # new exception would escape the decorator.
            message = safe_exception_message(exc)
            logger.opt(exception=exc).error(
                "Built-in tool '{}' raised; the model will see one line: {}: {}",
                func.__name__,
                exc_name,
                message,
            )
            return f"Error: {exc_name}: {message}"

    return _wrapper
