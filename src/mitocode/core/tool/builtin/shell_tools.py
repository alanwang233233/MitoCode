"""The one built-in tool that runs a shell command: ``Bash``.

Every other workspace-confined tool in this package reaches the world through
a *path*, and :func:`~mitocode.core.tool.builtin.paths.resolve` is the one
chokepoint that decides whether a path may be touched.  A shell command is
not a path: it is a whole program's worth of reach in one string, and it can
create, move or destroy things nowhere near any path it names.  So ``Bash``
runs its own chokepoint -- a textual escape scan feeding the same approver
the file tools use -- and this module is the only place that chokepoint
lives.

**The escape scan is a guardrail against accidents, not a security
boundary.**  It tokenises the command and flags the tokens that *look like*
they reach outside the workspace: a leading ``~``, an absolute path that
resolves outside the root, a ``..`` climb.  It is bypassable by anyone who
tries -- command substitution, variables, quoted whitespace, a flag with an
attached path such as ``-I/usr/include``, a ``..`` buried mid-path, a
redirection operator glued to its target such as ``2>/tmp/x`` -- and it
is not a sandbox, and nothing in this module should ever be described as
one.  Its purpose is to catch the model reaching outside the workspace *by
mistake*, which is the overwhelmingly common case, and to route that case to
the approver -- where a deliberate escape was always going to end up anyway.

**The approver's verdict is authoritative, and the approver hears only about
escapes.**  A command whose every token sits inside the root never consults
the approver at all -- the same rule
:mod:`~mitocode.core.tool.builtin.paths` follows, for the same reason: an
interactive approver that prompted on every ordinary command would train its
user to answer "yes" without reading, and the common path would pay for the
rare one.  When a token *is* flagged, the request goes to the approver
exactly once and its answer is final: on ``ALLOW`` the command runs with no
second check behind it, exactly as an approved escape does for the file
tools -- re-checking containment here would override the verdict and make a
granting approver inert for this tool.  No workspace
(``ToolContext.workspace is None``) means no boundary and no approver, the
documented behaviour of
:class:`~mitocode.core.tool.builtin.context.ToolContext`.

**A timeout -- and a cancellation -- kill the process GROUP, not the
shell.**  The subprocess is launched with ``start_new_session=True``, which
makes it the leader of its own process group, and on expiry -- or when the
tool's own task is cancelled, which is how ``AgentRuntime`` interrupts a
running command -- the whole group is signalled with
``os.killpg(os.getpgid(...), SIGKILL)``.  A bare ``proc.kill()`` kills the
shell and *only* the shell: a ``sleep 30`` it spawned survives the call,
keeps running detached from the tool that started it, and leaks.  The group
kill is the difference between a tool that ends -- by timeout or by
cancellation -- and a tool that leaves strays behind.

**The output is the child's own words, plus the exit code when it failed.**
stdout and stderr are merged so nothing the command tried to say is lost,
bytes are decoded with ``errors="replace"`` so a binary ``cat`` cannot
raise, and the result is clamped to ``ToolContext.max_output_chars``.  A
non-zero exit is appended as an ``[exit code: N]`` marker rather than
prefixed with ``Error: `` -- the tool call itself succeeded, and the
command's failure is *data for the model*, where ``Error: `` is reserved for
the tool's own failures: a refused escape, a timeout, a process that could
not be spawned at all.  That last one, like every unexpected exception, is
rendered by :func:`~mitocode.core.tool.builtin.errors.tool_boundary`.

Like the other tool modules, this one deliberately does **not** use
``from __future__ import annotations``: ``ToolManager.register`` derives a
tool's JSON Schema from the raw annotations of the registered callable, and
stringified ones would break it.
"""

import asyncio
import math
import os
import re
import signal
from pathlib import Path
from typing import Any, Optional, Tuple

from mitocode.core.tool.builtin.approval import ApprovalDecision, ApprovalRequest
from mitocode.core.tool.builtin.context import clamp, current
from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError, tool_boundary

#: Separator run that ends one shell token and starts the next: whitespace
#: (which includes newlines), ``;``, ``&`` and ``|``.  Splitting on the
#: individual characters subsumes the two-character operators -- ``&&`` is a
#: run of ``&``, ``||`` a run of ``|`` -- and also splits on a single ``&``
#: (a backgrounded job).  Redirections are where the split is only half
#: right.  ``2>&1`` is broken into ``2>`` and ``1``, and no misjudgment
#: follows -- neither fragment is path-shaped, so neither is judged as if it
#: were a path.  The reverse case is a blind spot: a redirection operator
#: *glued* to its target survives the split as one token -- ``echo hi
#: >/etc/passwd`` tokenises to ``>/etc/passwd`` and ``sort 2>/tmp/x`` to
#: ``2>/tmp/x`` -- and a leading ``>`` or digit defeats the absolute-path
#: rule in :func:`scan_for_escape`, so the target is never flagged.  That is
#: not because splitting makes redirections harmless; it is one more shape
#: the scan does not see, same class as a flag with an attached path, and it
#: is listed in the bypass documentation rather than claimed away.
_TOKEN_SEPARATORS = re.compile(r"[;&|\s]+")

#: Seconds to wait for a killed process group to release its output pipes
#: before giving up on collecting its final output.  Generous, because the
#: group has just been SIGKILLed and pipes close the moment every holder
#: dies -- but bounded, so a grandchild that escaped into a session of its
#: own and inherited the pipe cannot hold the timeout path open forever.
#: The kill itself has already happened either way; this bounds only the
#: collection of the dead command's last words.
_REAP_WAIT_SECONDS = 5.0


def _kill_group(pid: int) -> None:
    """SIGKILL the process group *pid* leads, best effort.

    The shell is launched with ``start_new_session=True``, so it is the
    leader of its own group and its pgid is its own pid; signalling the group
    reaches every child and grandchild it spawned, which a bare
    ``proc.kill()`` would not.

    Args:
        pid: The pid of the shell whose group should die.

    Notes:
        Every ``OSError`` is swallowed on purpose: a group that is already
        gone (``ProcessLookupError``) needs no killing, one that cannot be
        signalled from here (``PermissionError``) cannot be helped, and the
        caller's "timed out" message must reach the model even when the
        cleanup fails.  This is cleanup, not control flow.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        pass


def scan_for_escape(command: str, root: Path) -> Optional[str]:
    """Return the first token of *command* that looks like it leaves *root*.

    **This is a guardrail against accidental escapes, not a security
    boundary, and it is not a sandbox.**  It is bypassable by anyone who
    tries -- command substitution, variables, quoted whitespace, a ``..``
    buried mid-path, a flag with an attached path, a redirection operator
    glued to its target (``echo hi >/etc/passwd`` produces the token
    ``>/etc/passwd``, whose leading ``>`` defeats the absolute-path rule) --
    and it exists to catch
    the model reaching outside the workspace *by mistake* and hand that case
    to the approver, never to stop a determined escapee.

    The command is tokenised on runs of whitespace (including newlines) and
    shell separators (``;``, ``&&``, ``||``, ``|``).  The first token matching
    any of these rules is returned:

    * a token starting with ``~`` -- a home-relative path leaves the root by
      definition, and there is no ``~`` that means this workspace;
    * the token ``..`` itself, or one starting with ``../``;
    * an absolute path -- a token starting with ``/`` -- that, after
      resolving, does not land under *root*.  Both sides are resolved, so a
      root reached through a symlink (on macOS ``/var`` versus
      ``/private/var``) still compares equal.  A token that is not
      path-shaped is skipped: one containing ``=`` before any ``/`` is an
      option or an environment assignment (``--file=/tmp/x``,
      ``FOO=/tmp/x``), not a path, and is never flagged as one.  A token
      starting with ``/`` that cannot be resolved at all counts as an
      escape, so the questionable case reaches the approver rather than
      being waved through.

    Args:
        command: The command line as the model wrote it.  Not parsed with a
            shell grammar -- the scan reads shapes, not syntax, and quotes
            are ordinary characters to it.
        root: The workspace root.  Resolved here, so a caller may pass it
            unresolved (``ToolContext.root()`` already is resolved).

    Returns:
        The offending token, or ``None`` when nothing was flagged.  ``None``
        means "this scan saw no accidental escape" and nothing more: it is a
        statement about the scan, never a claim that the command cannot
        reach outside the root.
    """
    try:
        resolved_root = root.resolve()
    except (OSError, RuntimeError, ValueError):
        # Same fallibility as paths._resolve_best_effort: resolve() raises on
        # a symlink loop (before 3.13), an untraversable component, or an
        # embedded NUL.  Comparing against the un-normalised root is then the
        # honest option; it can only make the scan *more* likely to flag,
        # never less.
        resolved_root = root
    for token in _TOKEN_SEPARATORS.split(command):
        if not token:
            # Adjacent separators produce empty tokens; they name nothing.
            continue
        if token.startswith("~"):
            return token
        if token == ".." or token.startswith("../"):
            return token
        equals = token.find("=")
        slash = token.find("/")
        if equals != -1 and (slash == -1 or equals < slash):
            # `--file=/tmp/x` or `FOO=/tmp/x`: an option or an environment
            # assignment, not a path.  The path rules do not apply to it.
            continue
        if token.startswith("/"):
            try:
                candidate = Path(token).resolve()
            except (OSError, RuntimeError, ValueError):
                # Unresolvable absolute path: flag it and let the approver
                # decide, rather than silently passing the odd case through.
                return token
            if not candidate.is_relative_to(resolved_root):
                return token
    return None


#: Largest ceiling :func:`bash` will hand to the subprocess, in
#: milliseconds.  One hour: longer than any single command this harness
#: expects to host, short enough that a wedged call cannot pin an agent
#: session open past the working day.
MAX_TIMEOUT_MS = 3_600_000


def _resolve_timeout(timeout: Any) -> Tuple[int, Optional[str], Any]:
    """Normalise the model-supplied ceiling before it reaches the subprocess.

    **The rule, and where the bound lives.**  The bound is
    :data:`MAX_TIMEOUT_MS` -- one hour -- and it lives in this module because
    this is the first place a model-supplied number is believed: the
    subprocess layer has no notion of a ceiling, and by the time a
    value reaches ``wait_for`` unbounded the chance to bound it is gone.
    Concretely:

    * ``None`` becomes the default (120 000 ms), matching the signature
      default.
    * A non-number, or a boolean -- ``True`` is an ``int`` in Python, and a
      one-millisecond ceiling is never what a model sending ``true`` meant --
      is refused.
    * A non-finite float is refused outright.  ``1e400`` parses to ``inf``,
      and an infinite ceiling is a promise never to interrupt: the exact hang
      the ceiling exists to prevent.
    * A finite value at or below zero falls back to the default
      (120 000 ms, two minutes), matching the signature default.  This
      allows the model to omit the timeout for quick commands.
    * A finite value above :data:`MAX_TIMEOUT_MS` is *clamped* to it, with a
      note in the launch reply saying so, because the request is still
      meaningful -- only its magnitude is not.

    Args:
        timeout: The ceiling as the model stated it, in milliseconds.
            Typically an ``int``; eventually anything.

    Returns:
        ``(timeout_ms, note, timeout_for_display)``: the effective ceiling in
        milliseconds, a prose sentence to append to the launch reply when the
        requested value had to be clamped (``None`` otherwise), and the
        original timeout value for display in error messages.

    Raises:
        MitoCodeBuiltinToolError: When the value is not a usable number at
            all - a non-number, a boolean, or a non-finite float.  The
            caller converts this into ``"Error: ..."`` prose rather than
            letting it escape.
    """
    original_timeout = timeout
    if timeout is None:
        return 120_000, None, original_timeout
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
                "timeout must be a finite number of milliseconds; got "
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
        # Zero or negative means "no explicit ceiling" -- fall back to the
        # default so the model doesn't have to specify a timeout for quick
        # commands.  This matches the signature default and existing behavior.
        return 120_000, None, original_timeout
    if value > MAX_TIMEOUT_MS:
        return MAX_TIMEOUT_MS, (
            f"The requested ceiling of {value} ms was clamped to "
            f"{MAX_TIMEOUT_MS} ms (one hour), the largest ceiling Bash "
            "grants."
        ), original_timeout
    return value, None, original_timeout


@tool_boundary
async def bash(command: str, timeout: int = 120_000) -> str:
    """Run *command* in the workspace root and return its output.

    The command runs under the system shell (``/bin/sh``) with its working
    directory set to the workspace root, its stdin attached to ``DEVNULL`` --
    a command that waits for input gets EOF instead of holding the call open
    until the timeout, and a TUI terminal is never shared with a
    subprocess -- and its stdout and stderr merged, in order, into one
    stream.

    Before anything runs, :func:`scan_for_escape` looks for tokens reaching
    outside the workspace.  The scan is a guardrail against accidents, not a
    security boundary and not a sandbox; when it flags a token, the command
    goes to the context approver exactly once and the verdict is
    authoritative: ``ALLOW`` runs the command with no second check behind
    it, ``DENY`` returns a refusal.  An in-root command never consults the
    approver.  With no workspace on the context there is no root to escape
    from, so the scan and the approver are both skipped -- the documented
    no-boundary behaviour of ``ToolContext``, the same
    :func:`~mitocode.core.tool.builtin.paths.resolve` follows.

    On timeout the subprocess is killed as a whole process group
    (``start_new_session=True`` plus ``os.killpg`` with ``SIGKILL``), so
    children it spawned -- a backgrounded ``sleep``, a server it started --
    die with it instead of outliving the call.

    Cancellation kills the group too.  Interrupting a running command is
    done by cancelling the task the tool runs in -- that is how
    ``AgentRuntime`` stops a turn -- so the ``CancelledError`` handler
    signals the whole process group before re-raising, exactly as the
    timeout path does.  The cancellation then propagates untouched
    (``tool_boundary`` never catches ``BaseException``), which means there
    is no result to attach output to: whatever the command printed before
    it was killed is dropped on this path, and the reader task that was
    collecting it is cancelled rather than awaited -- bounded, best-effort
    cleanup, not a second result channel.  Like the reaper bound below,
    this can leave the subprocess transport to be closed at garbage
    collection, which surfaces as bounded ``Event loop is closed`` noise
    from ``__del__`` at teardown; the kill itself always happens first.

    Args:
        command: The shell command line to run.  Interpreted by the shell,
            so pipes, redirection and quoting all work -- and so a command
            is exactly as powerful as a shell, which is why the escape scan
            and the approver sit in front of it.
        timeout: The subprocess timeout, in **milliseconds**.  The command
            is killed as a process group when it exceeds this.  A value of
            zero or less expires immediately, killing a command that has
            only just started.

    Returns:
        The command merged stdout and stderr, decoded with
        ``errors="replace"`` so non-UTF-8 output cannot raise, with
        ``"[exit code: N]"`` appended when the exit code is non-zero, and
        clamped to ``ToolContext.max_output_chars``.  A non-zero exit code
        is reported as data and deliberately does **not** begin with
        ``"Error: "`` -- the tool call succeeded; the command failed.

        A refused escape or a timeout is a failure of the tool and begins
        with ``"Error: "``: the refusal names the command, the workspace
        root and the token that was flagged; the timeout message says
        ``"timed out"`` and names the command, and normally includes
        whatever the command managed to print before it was killed --
        normally, because collecting that output is bounded by a wait
        (``_REAP_WAIT_SECONDS``) and when the bound expires -- a
        grandchild that escaped the killed group into a session of its own
        still holding the output pipe -- the printed output is dropped and
        the message carries no output section at all.  The kill itself has
        always happened by then either way.  Never raises for anything
        unexpected -- that is rendered by ``tool_boundary`` -- and never
        swallows a cancellation: the one deliberate escape is
        ``CancelledError``, re-raised after the group kill (above).
    """
    ctx = current()
    root = ctx.root()

    # Normalise and clamp the timeout before anything else; this prevents
    # infinite or absurd ceilings from reaching wait_for and hanging the turn.
    try:
        timeout_ms, timeout_note, timeout_for_display = _resolve_timeout(timeout)
    except MitoCodeBuiltinToolError as exc:
        return f"Error: {exc}"

    if ctx.workspace is not None:
        flagged = scan_for_escape(command, root)
        if flagged is not None:
            decision = await ctx.approver.request(
                ApprovalRequest(
                    action="bash",
                    detail=command,
                    reason=(
                        f"the command reaches outside the workspace root "
                        f"{root} (flagged token: {flagged})"
                    ),
                )
            )
            if decision is not ApprovalDecision.ALLOW:
                # The approver's DENY is the refusal: one sentence naming the
                # command, the root it was supposed to stay inside, and the
                # token that tripped the scan -- the three things the model
                # needs to route around it.
                return (
                    f"Error: bash on {command} was declined: it reaches "
                    f"outside the workspace root {root} "
                    f"(flagged token: {flagged})."
                )

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(root),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    # The reader is a task created exactly once, and the first timeout is
    # aimed at a shield around it: on expiry the shield is cancelled while
    # the reader keeps running, the group is killed, and the SAME reader is
    # awaited once more to collect whatever the command printed before it
    # died.  Re-entering communicate() after a cancelled first attempt is
    # not something the asyncio contract promises, so it is never done.
    communicate = asyncio.create_task(proc.communicate())
    try:
        stdout_bytes, _ = await asyncio.wait_for(
            asyncio.shield(communicate), timeout_ms / 1000
        )
    except asyncio.CancelledError:
        # The tool's own task was cancelled -- how AgentRuntime interrupts a
        # running command -- and wait_for cancelled the shield wrapper, not
        # the reader behind it.  Without this handler the cancellation would
        # propagate with nothing signalling the group: the shell and every
        # child it spawned would keep running detached from the tool, the
        # exact stray-leaking failure the timeout path exists to prevent.
        # Kill first, consume the orphaned reader, then re-raise; swallowing
        # the cancellation instead would make a cancelled turn uncancellable,
        # the same bug class errors.py exists to prevent.
        _kill_group(proc.pid)
        # The reader survived the shield's cancellation and is still running;
        # cancel it so no task lingers pending on a loop that may be shutting
        # down.  Its partial output is deliberately dropped -- a cancellation
        # carries no result to attach it to -- and whatever ends the reader
        # (its own cancellation, an unexpected error, even a second
        # cancellation of this task arriving mid-await) must not replace the
        # cancellation about to be re-raised: cleanup never outranks the
        # signal.
        communicate.cancel()
        try:
            await communicate
        except BaseException:  # pylint: disable=broad-exception-caught
            pass
        raise
    except asyncio.TimeoutError:
        _kill_group(proc.pid)
        partial = b""
        try:
            partial, _ = await asyncio.wait_for(communicate, _REAP_WAIT_SECONDS)
        except asyncio.TimeoutError:
            # wait_for has cancelled the reader: something outside the group
            # still holds the pipes.  The kill has already happened, the
            # asyncio child watcher reaps the process on its own, and waiting
            # behind a stranger would hang the tool a second time.
            pass
        text = partial.decode("utf-8", errors="replace")
        if text.strip():
            return (
                f"Error: bash timed out after {timeout_for_display} ms: {command}. "
                "The process group was killed; its output up to then:\n\n"
                + clamp(text, ctx.max_output_chars)
            )
        return (
            f"Error: bash timed out after {timeout_for_display} ms: {command}. "
            "The process group was killed."
        )

    text = stdout_bytes.decode("utf-8", errors="replace")
    returncode = proc.returncode
    if returncode is not None and returncode != 0:
        # Appended before clamping, and clamp keeps both ends, so the marker
        # survives even a full output budget -- a model must never have to
        # guess whether a command it cannot see the end of failed.  The
        # child's own trailing newline is dropped so the marker sits one
        # blank line under the last line of output, not two.
        text = text.rstrip("\n") + f"\n\n[exit code: {returncode}]"
    return clamp(text, ctx.max_output_chars)
