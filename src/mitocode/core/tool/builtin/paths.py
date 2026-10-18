"""The one place built-in *file* paths are resolved, and the approval gate for them.

Every file-touching tool in this package — ``Read``, ``Write``, ``Edit``,
``Glob``, ``Grep`` — begins by handing its path arguments to
:func:`resolve` and branching on the answer.  There is no second containment
check anywhere else in the package, which is the entire point: a second check
is a second thing to get wrong, and a tool that forgot it would have no
boundary at all rather than a slightly different one.  ``Bash`` has its own
chokepoint for its own reason — it is the other call site that may have to
wait on a human — but a *path* reaches an approver only through here.

The function is ``async`` for a reason that has nothing to do with the
filesystem.  Path resolution is synchronous and always will be; what is
asynchronous is the *verdict*, because a future approver is a dialog that has
to hand control to the TUI's event loop and wait for a keystroke.  Making the
signature synchronous now would mean changing every call site later, at exactly
the moment somebody is adding that dialog.

Four decisions in the body are load-bearing, and none of them is obvious from
the code alone.

**The approver is consulted only on escape.**  An in-root path is answered
without awaiting anything.  The failure mode this avoids is not exotic: a
permission prompt per keystroke, or a dialog whose "yes" is muscle memory, and
an agent whose tool calls take ten times as long because every ordinary read
of an ordinary file stops to ask about an ordinary file.  A permissive default
approver is safe to ship *because* of this rule — with it, ``AlwaysAllow`` is
never reached for the traffic that makes up almost all of the work, so
installing a real dialog later cannot degrade the common path.

**The containment test is :meth:`Workspace.is_within`, not
:meth:`Workspace.resolve_within`.**  Both answer "is this inside the root?",
but only ``is_within`` also answers "is this resolvable at all?", and the
second question is not a refinement.  ``Path.resolve()`` changed behaviour on
a symlink loop in Python 3.13: on 3.10-3.12 it raises ``RuntimeError``, which
``resolve_within`` does not catch, so a loop propagates out of the gate; on
3.13+ it does not raise at all — it returns the path with the looping
component left in place, still spelled under the root, and containment alone
answers True.  A caller that trusts that verdict goes on to ``open()`` a path
that can only ever fail with ``ELOOP``.  ``is_within`` answers False for a loop
on every supported version (by catching ``RuntimeError`` before 3.13 and by
requiring a fully-resolved path from 3.13 on), and it never raises for a path
it dislikes.  ``resolve_within`` is still used, but only *after* ``is_within``
has already said yes, purely as the thing that produces the resolved path —
its exception means nothing here, because it is only reached on a path already
known to be in bounds.

**A denial is a string, not an exception.**  ``AgentRuntime`` feeds a tool's
return value straight back to the model, so a refusal that arrives as prose is
something the model can read and route around — it will try a different path,
ask for permission, or give up in a way it can explain.  The same refusal as a
traceback would be noise in a transcript.  The refusal names the action, the
concrete path *and* the reason, because "declined" on its own leaves the model
unable to tell an out-of-workspace refusal from a missing file or a malformed
argument — and those are the two other failures a file tool reports in exactly
the same shape.

**No workspace means no boundary.**  ``ToolContext.workspace is None`` is a
legitimate state — a context built in a test, in a smoke script, or before a
workspace has been initialised — and in that state there is no root to escape
*from*.  Every path is therefore trivially in-root and the approver is never
consulted.  This is documented behaviour rather than an oversight or a hole:
the boundary is the workspace root, and with no workspace there is no boundary
to enforce, which is why the field is optional in the first place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from mitocode.core.tool.builtin.approval import ApprovalDecision, ApprovalRequest
from mitocode.core.tool.builtin.context import ToolContext


def _resolve_best_effort(candidate: Path) -> Path:
    """Return *candidate* with ``..`` and symlinks collapsed, or *candidate*.

    :meth:`pathlib.Path.resolve` is the normaliser everywhere in this package,
    and it is fallible in three different ways: it raises ``RuntimeError`` on a
    symlink loop before Python 3.13, ``OSError`` when a component cannot be
    traversed, and ``ValueError`` for a path carrying an embedded NUL byte.
    None of those may be allowed to escape the chokepoint as an exception,
    because an exception here is a traceback in a transcript rather than a
    sentence the model can read — and because the caller may be about to ask an
    approver about the path, which needs a *description* of the path in order
    to ask, not an abort.

    Falling back to the un-normalised candidate is safe precisely because the
    caller has already decided what to do with it: the containment gate has
    rejected it, and the only paths that reach here un-normalised are ones that
    cannot be normalised.  Whatever the verdict, the string handed to the
    approver names the path the model actually asked for, and the returned
    path — on an allow — is the same path that will fail at ``open()`` with a
    far more specific error than anything this module could invent.

    Args:
        candidate: An **absolute** path, already joined to the root by the
            caller.  Not validated here; joining happens in :func:`resolve` so
            that there is exactly one place that decides what a relative path
            means.

    Returns:
        The fully resolved path, or *candidate* unchanged when resolution is
        impossible.  Never raises for a path that was constructed at all.
    """
    try:
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return candidate


async def resolve(
    ctx: ToolContext, target: Union[str, Path], action: str
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve *target* and return it only if it may be operated on.

    The single approval chokepoint for file operations.  Returns a
    ``(path, error)`` pair in which exactly one member is ``None`` — never an
    exception for a refusal, because the caller feeds the pair straight back to
    the model and prose is something a model can act on while a traceback is
    not.

    An in-root path never reaches the approver.  Only a path the workspace
    rejects becomes an :class:`~mitocode.core.tool.builtin.approval.ApprovalRequest`
    and is awaited, which is what keeps an interactive approver viable later:
    the dialog appears for escapes, not for ordinary work.

    Args:
        ctx: The tool context supplying the root, the workspace that bounds
            it, and the approver consulted on escape.  Read for its workspace
            and approver only; the caller is expected to have obtained it from
            :func:`~mitocode.core.tool.builtin.context.current`.
        target: The path as the model wrote it.  An **absolute** path is taken
            as given and is never re-rooted — a model that asked for
            ``/etc/hostname`` must get a verdict about ``/etc/hostname`` and
            not about some file inside the workspace that happens to share its
            name.  A **relative** path is joined to ``ctx.root()``.
        action: The verb being attempted — ``"read"``, ``"write"``, ``"edit"``,
            ``"glob"`` or ``"grep"``.  Passed through to the approver and
            quoted in the refusal so that both the human and the model can tell
            which operation was refused.  Deliberately **not** validated: the
            action is only displayed and echoed, so an unrecognised verb is
            already inert, and a check here would be a second source of truth
            for which verbs exist alongside the eleven tool definitions.

    Returns:
        ``(path, None)`` when the target may be operated on.  *path* is fully
        resolved, so a caller may hand it straight to ``open()``, ``glob()`` or
        ``os.walk()`` without re-deriving it.

        ``(None, error)`` when the target may not be.  *error* is a refusal
        string beginning with ``"Error: "`` — the shape
        :mod:`~mitocode.core.tool.builtin.errors` renders every failure in,
        so the model sees one vocabulary regardless of which tool asked.  On a
        denial it names the action, the resolved path and the reason, for
        example ``"Error: read on /etc/hostname was declined: it resolves
        outside the workspace root /proj."``  The root is repeated on purpose:
        it is the one piece of the sentence that tells the model which
        directory it was supposed to be working in.

    Raises:
        TypeError: If *target* is not something :class:`pathlib.Path` accepts.
            That is a fault in the calling tool rather than something the model
            asked for, so it is left to propagate: a tool is wrapped in
            :func:`~mitocode.core.tool.builtin.errors.tool_boundary`, which
            renders it to the model as ``"Error: TypeError: ..."``, and which
            this function cannot invoke on its own.
        MitoCodeWorkspacePathError: If the workspace accepts the target on the
            containment check and then rejects it when asked for the resolved
            path.  Only a filesystem change *between* the two calls can cause
            that — they run without an intervening ``await`` — and the escape
            is deliberately not re-submitted to the approver, because doing so
            on a race would mean asking about a path whose answer depends on
            when the question was asked.
    """
    root = ctx.root()
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = root / candidate

    workspace = ctx.workspace
    if workspace is None:
        # No workspace, no boundary, nothing to escape from.  See the module
        # docstring: this is the documented no-workspace behaviour, and it is
        # why the approver below is not reached.
        return _resolve_best_effort(candidate), None

    if workspace.is_within(candidate):
        # The fast path, and the one that carries the ordinary traffic.  No
        # await, no approver, no allocation beyond this tuple.  `is_within` has
        # already answered "inside the root and fully resolvable", so the
        # `resolve_within` below is being asked only for the path itself; its
        # exception carries no information that has not already been used.
        return workspace.resolve_within(candidate), None

    # `is_within` is False: the target escaped the root, or sits inside it yet
    # cannot be used (a symlink loop, an untraversable component).  The single
    # predicate covers both, and collapsing them here is the point: a loop is
    # inside the root, so nothing outside it can be reached through it either
    # way, and an approved loop path fails later at `open()` with a precise
    # `OSError` instead of a refusal that would misreport an in-root path as
    # an escape.
    path = _resolve_best_effort(candidate)
    detail = str(path)
    decision = await ctx.approver.request(
        ApprovalRequest(
            action=action,
            detail=detail,
            reason=f"resolves outside the workspace root {root}",
        )
    )
    if decision is ApprovalDecision.ALLOW:
        return path, None
    # Named on all three axes the model needs: the action, so it knows what was
    # refused, the path, so it knows what to try instead, and the reason, so it
    # knows that "declined" means "outside the workspace" and not "the file is
    # gone" or "the arguments were wrong" — the two other failures a file tool
    # reports in the same shape.
    return None, (
        f"Error: {action} on {detail} was declined: "
        f"it resolves outside the workspace root {root}."
    )
