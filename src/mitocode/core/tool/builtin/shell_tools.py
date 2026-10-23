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
attached path such as ``-I/usr/include``, a ``..`` buried mid-path -- and it
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

**A timeout kills the process GROUP, not the shell.**  The subprocess is
launched with ``start_new_session=True``, which makes it the leader of its
own process group, and on expiry the whole group is signalled with
``os.killpg(os.getpgid(...), SIGKILL)``.  A bare ``proc.kill()`` kills the
shell and *only* the shell: a ``sleep 30`` it spawned survives the call,
keeps running detached from the tool that started it, and leaks.  The group
kill is the difference between a tool that times out and a tool that leaves
strays behind.

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
import os
import re
import signal
from pathlib import Path
from typing import Optional

from mitocode.core.tool.builtin.approval import ApprovalDecision, ApprovalRequest
from mitocode.core.tool.builtin.context import clamp, current
from mitocode.core.tool.builtin.errors import tool_boundary

#: Separator run that ends one shell token and starts the next: whitespace
#: (which includes newlines), ``;``, ``&`` and ``|``.  Splitting on the
#: individual characters subsumes the two-character operators -- ``&&`` is a
#: run of ``&``, ``||`` a run of ``|`` -- and also splits on a single ``&``
#: (a backgrounded job) and on redirections such as ``2>&1``, which is
#: harmless: the fragments on either side are not paths, and are not judged
#: as if they were.
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
    buried mid-path, a flag with an attached path -- and it exists to catch
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


@tool_boundary
async def bash(command: str, timeout: int = 120_000) -> str:
    """Run *command* in the workspace root and return its output.

    The command runs under the system shell (``/bin/sh``) with its working
    directory set to the workspace root, its stdin attached to ``DEVNULL`` --
    a command that waits for input gets EOF instead of holding the call open
    until the timeout, and a TUI's terminal is never shared with a
    subprocess -- and its stdout and stderr merged, in order, into one
    stream.

    Before anything runs, :func:`scan_for_escape` looks for tokens reaching
    outside the workspace.  The scan is a guardrail against accidents, not a
    security boundary and not a sandbox; when it flags a token, the command
    goes to the context's approver exactly once and the verdict is
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
        The command's merged stdout and stderr, decoded with
        ``errors="replace"`` so non-UTF-8 output cannot raise, with
        ``"[exit code: N]"`` appended when the exit code is non-zero, and
        clamped to ``ToolContext.max_output_chars``.  A non-zero exit code
        is reported as data and deliberately does **not** begin with
        ``"Error: "`` -- the tool call succeeded; the command failed.

        A refused escape or a timeout is a failure of the tool and begins
        with ``"Error: "``: the refusal names the command, the workspace
        root and the token that was flagged; the timeout message says
        ``"timed out"`` and names the command, and includes whatever the
        command managed to print before it was killed.  Never raises:
        anything unexpected is rendered by ``tool_boundary``.
    """
    ctx = current()
    root = ctx.root()

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
            asyncio.shield(communicate), timeout / 1000
        )
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
                f"Error: bash timed out after {timeout} ms: {command}. "
                "The process group was killed; its output up to then:\n\n"
                + clamp(text, ctx.max_output_chars)
            )
        return (
            f"Error: bash timed out after {timeout} ms: {command}. "
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
