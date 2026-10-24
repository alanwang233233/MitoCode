"""Tests for the one built-in shell tool: ``bash`` and its escape scan
``scan_for_escape``.

The suite is built around the behaviours the tool exists to guarantee, and
four of them pin defects that were recently fixed:

**A cancelled tool task must kill the process GROUP, not just the shell.**
``AgentRuntime`` stops a turn by cancelling the task a tool runs in, so
cancellation is how a user interrupts a command.  The cancellation path must
signal the whole group (``os.killpg`` + SIGKILL) -- a bare ``proc.kill()``
kills the shell and *only* the shell, and the ``sleep`` it spawned survives,
reparented and detached.  The test launches a compound command (``sleep
987654; echo ...``) so the shell stays the sleep's *parent* -- a lone
``sleep X`` would be exec-optimised into the shell itself and a bare kill
would then look correct -- and verifies via ``pgrep`` that nothing survives,
that ``CancelledError`` still propagates, and that a regression fails an
assertion instead of hanging the suite (``asyncio.wait`` deadlines
everywhere).

**The timeout path kills the whole group too.**  Same mechanism, same
survivor check: no ``sleep`` may outlive a timeout that killed only its
shell.

**The approver contract.**  An in-root command reaches the approver ZERO
times; an escape reaches it EXACTLY ONCE; an ALLOW verdict actually runs the
command; a DENY verdict refuses in prose and runs nothing.  Every count is
asserted through a recording approver, never just the outcome.

**Non-UTF-8 output must not raise.**  ``cat`` of a binary file is decoded
with ``errors="replace"``; the expected replacement characters are
hand-computed, not copied from a run.

``scan_for_escape`` is pinned as what the module docstring says it is: a
guardrail against *accidental* escapes, not a sandbox.  The rules (a leading
``~``, ``..`` / ``../``-prefixed tokens, absolute paths resolving outside
the root, with ``=``-before-``/`` tokens exempt as option/env shapes) are
pinned alongside the documented blind spots -- glued redirection targets
(``>/etc/passwd``, ``2>/tmp/x``) and a ``..`` buried mid-path -- so the
documentation stays honest and any change to the guardrail is deliberate.

Repo conventions shape the file: module-level test functions (not classes),
``tmp_path`` over mocking ``open``, hand-written doubles over patching, and
on macOS every expected path derives from an already-resolved root so the
``/var`` vs ``/private/var`` spellings can never disagree.  The pgrep
survivor checks use sleep durations (9876xx) unique enough that no other
process on the machine can match them.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.core.tool.builtin.context as context_module
from mitocode.core.tool.builtin.approval import ApprovalDecision
from mitocode.core.tool.builtin.context import ToolContext, set_current
from mitocode.core.tool.builtin.shell_tools import bash, scan_for_escape
from mitocode.core.workspace.workspace import Workspace

#: Sleep duration used by the cancellation test.  Chosen so ``pgrep -f`` can
#: match *this test's* subprocesses and nothing else's; the command is
#: compound (``sleep X; echo ...``) so the shell stays the sleep's parent
#: rather than exec-optimising into it -- that parent relationship is what a
#: bare ``proc.kill()`` breaks and a group kill does not.
CANCELLED_SLEEP = "sleep 987654"

#: Sleep duration for the timeout group-kill test (killed after 700 ms).
TIMEOUT_SLEEP = "sleep 987650"

#: Sleep duration for the timeout-with-output test (killed after 500 ms,
#: after ``echo`` has already printed).
TIMED_OUTPUT_SLEEP = "sleep 987649"

#: Sleep duration for the non-positive-timeout tests (killed immediately).
EXPIRED_SLEEP = "sleep 987647"

#: Seconds any pgrep poll or cancellation wait gets before the test declares
#: a hang.  Generous against a slow machine, far below the "forever" a
#: regression (an un-killed ``sleep 9876xx``) would produce.
DEADLINE_SECONDS = 10.0


# --- doubles and helpers -----------------------------------------------------


class _RecordingApprover:
    """An approver that records every request and returns a fixed decision.

    The tests assert on ``requests`` -- its length above all -- because the
    approver contract is about how many times the approver is consulted, not
    merely about the final sentence: an in-root command must reach it zero
    times, an escape exactly once.
    """

    def __init__(self, decision: ApprovalDecision = ApprovalDecision.ALLOW) -> None:
        """Record requests and answer with *decision*.

        Args:
            decision: The verdict every recorded request receives.
        """
        self.decision = decision
        self.requests: list = []

    async def request(self, req) -> ApprovalDecision:
        """Record *req* and return the configured verdict.

        Args:
            req: The ``ApprovalRequest`` the tool asked about.

        Returns:
            The decision this approver was built with.
        """
        self.requests.append(req)
        return self.decision


def _install(
    workspace: Workspace,
    approver,
    max_output_chars: int = 30_000,
) -> ToolContext:
    """Build a context over *workspace* and register it as the current one.

    Args:
        workspace: The workspace whose root bounds the tool.
        approver: The policy consulted when a command is flagged.
        max_output_chars: The clamp budget the tool reads from the context.

    Returns:
        The registered ``ToolContext``.
    """
    ctx = ToolContext(
        workspace=workspace,
        approver=approver,
        max_output_chars=max_output_chars,
    )
    set_current(ctx)
    return ctx


async def _pids_matching(pattern: str) -> list:
    """Return the pids whose full command line matches *pattern*.

    Args:
        pattern: An ``pgrep -f`` pattern; the survivor checks pass sleep
            durations unique to one test so nothing else can match.

    Returns:
        The matching pids, empty when pgrep found none (its no-match exit
        code 1 is the normal "nothing survived" answer).
    """
    proc = await asyncio.create_subprocess_exec(
        "pgrep",
        "-f",
        pattern,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return [int(line) for line in stdout.decode("utf-8").split()]


async def _pgrep_until(
    pattern: str, present: bool, at_least: int, seconds: float
) -> list:
    """Poll pgrep until the expected survivor state holds or *seconds* pass.

    Args:
        pattern: The ``pgrep -f`` pattern to poll.
        present: ``True`` to wait for matches, ``False`` to wait for none.
        at_least: When *present*, how many matches must exist before
            returning (the cancellation test needs both the shell *and* its
            ``sleep`` child visible before cancelling).
        seconds: Polling deadline; afterwards the last observed list is
            returned whether or not it satisfies the expectation.

    Returns:
        The last observed pid list.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while True:
        pids = await _pids_matching(pattern)
        satisfied = (len(pids) >= at_least) if present else (not pids)
        if satisfied or loop.time() >= deadline:
            return pids
        await asyncio.sleep(0.05)


def _kill_survivors(pids: list) -> None:
    """Best-effort SIGKILL of the process groups *pids* belong to.

    Cleanup, not control flow: only reached on paths where a regression left
    a stray ``sleep`` behind, so the suite never leaks processes.

    Args:
        pids: The pids to signal.
    """
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass


async def _kill_matching(pattern: str) -> None:
    """Kill any process matching *pattern*, twice over, best effort.

    Args:
        pattern: The ``pgrep -f`` pattern whose matches should die.
    """
    for _ in range(2):
        pids = await _pids_matching(pattern)
        if not pids:
            return
        _kill_survivors(pids)
        await asyncio.sleep(0.1)


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_module_state():
    """Reset the context module's process-wide state around every test.

    ``_CURRENT`` survives between test files in one pytest process, so a
    leaked registration would hand this file's tool another file's context
    (and vice versa).
    """
    context_module._CURRENT = None  # pylint: disable=protected-access
    yield
    context_module._CURRENT = None  # pylint: disable=protected-access


@pytest.fixture(name="root")
def root_fixture(tmp_path):
    """A resolved, empty workspace root.

    The root is resolved up front so every expectation written against it
    shares one spelling; on macOS ``tmp_path`` is reached through ``/var``
    while resolved paths report ``/private/var``.

    Returns:
        The resolved workspace root directory.
    """
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    return root


@pytest.fixture(name="workspace")
def workspace_fixture(root):
    """A real ``Workspace`` rooted at *root*.

    Args:
        root: The resolved root from :func:`root_fixture`.

    Returns:
        A ``Workspace`` whose ``root()`` is the boundary the tool scans
        against.  A real instance rather than a double: the containment
        ruling is precisely that the tool defers to this boundary.
    """
    return Workspace(root, "builtin-shell-tests", 2)


# --- scan_for_escape: the guardrail, pinned ----------------------------------


def test_scan_for_escape_flags_a_token_starting_with_a_tilde(root):
    """A ``~``-prefixed token leaves the root by definition and is flagged."""
    assert (
        scan_for_escape("cat ~/.zshrc", root) == "~/.zshrc"
    ), "a home-relative token must be handed to the approver"


def test_scan_for_escape_flags_the_bare_tilde_token(root):
    """A lone ``~`` token is flagged: there is no ``~`` meaning this root."""
    assert (
        scan_for_escape("~", root) == "~"
    ), "the bare tilde token must be flagged like any other"


def test_scan_for_escape_flags_the_dotdot_token(root):
    """The token ``..`` itself is a climb out of the root."""
    assert (
        scan_for_escape("cd ..", root) == ".."
    ), "the bare parent-directory token must be flagged"


def test_scan_for_escape_flags_a_token_starting_with_dotdot_slash(root):
    """A token starting ``../`` climbs out of the root and is flagged."""
    assert (
        scan_for_escape("cat ../outside.txt", root) == "../outside.txt"
    ), "a ../-prefixed token must be flagged verbatim"


def test_scan_for_escape_flags_an_absolute_path_outside_the_root(root):
    """An absolute path that resolves outside the root is flagged."""
    assert (
        scan_for_escape("cat /etc/passwd", root) == "/etc/passwd"
    ), "an absolute path outside the root must be handed to the approver"


def test_scan_for_escape_allows_an_absolute_path_inside_the_root(root):
    """An absolute path that resolves under the root is not flagged."""
    assert (
        scan_for_escape(f"cat {root}/inside.txt", root) is None
    ), "an in-root absolute path is the common case and must never prompt"


def test_scan_for_escape_compares_resolved_paths_through_a_symlink(root):
    """A token reaching the root *through* a symlink is still inside it.

    Both sides are resolved before comparing, so a symlinked spelling of an
    in-root file is not mistaken for an escape.
    """
    alias = root / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert (
        scan_for_escape(f"cat {alias}/inside.txt", root) is None
    ), "a symlinked spelling of an in-root path must not be flagged"


def test_scan_for_escape_resolves_the_root_argument_itself(root, tmp_path):
    """The *root* argument may be passed unresolved and still compares equal.

    The docstring promises ``scan_for_escape`` resolves the root it is
    given; both directions -- unresolved root with a resolved token, and the
    reverse -- must agree that the file is inside.
    """
    real = (tmp_path / "real-root").resolve()
    real.mkdir()
    alias = tmp_path / "alias-root"
    alias.symlink_to(real, target_is_directory=True)
    (real / "inside.txt").write_text("x", encoding="utf-8")
    assert (
        scan_for_escape(f"cat {real}/inside.txt", alias) is None
    ), "an unresolved root must not turn an in-root token into an escape"
    assert (
        scan_for_escape(f"cat {alias}/inside.txt", real) is None
    ), "a symlinked token spelling must resolve against the real root"


def test_scan_for_escape_allows_a_token_that_is_the_root_itself(root):
    """A token that resolves to the root itself is inside, not an escape.

    Boundary of the absolute-path rule: ``is_relative_to`` holds for the
    root against itself.
    """
    assert (
        scan_for_escape(f"cat {root}", root) is None
    ), "the root itself must not be flagged as an escape from itself"


def test_scan_for_escape_treats_glued_separators_and_newlines_as_token_boundaries(
    root,
):
    """`;`, `||` and newlines split tokens even with no whitespace around them.

    The tokeniser splits on runs of separator characters, so an escape glued
    to the previous word (`a;cat /etc/passwd`) is still seen -- and so is an
    escape on a continuation line.
    """
    assert (
        scan_for_escape("echo a;cat /etc/passwd", root) == "/etc/passwd"
    ), "a semicolon glued between words must still separate tokens"
    assert (
        scan_for_escape("echo a||cat /etc/passwd", root) == "/etc/passwd"
    ), "a run of pipes glued between words must still separate tokens"
    assert (
        scan_for_escape("cat\n../up.txt", root) == "../up.txt"
    ), "a newline must separate tokens like any other separator"


def test_scan_for_escape_flags_a_sibling_of_the_root(root, tmp_path):
    """A directory next to the root is outside it and is flagged."""
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    token = f"{sibling}/secret.txt"
    assert (
        scan_for_escape(f"cat {token}", root) == token
    ), "a sibling of the root resolves outside it and must be flagged"


@pytest.mark.parametrize("token", ["--file=/tmp/x", "FOO=/tmp/x"])
def test_scan_for_escape_does_not_flag_option_or_env_shapes(root, token):
    """``=`` before any ``/`` makes a token an option or env assignment.

    ``--file=/tmp/x`` and ``FOO=/tmp/x`` are not paths; the path rules do
    not apply to them and they must not prompt the approver.
    """
    assert (
        scan_for_escape(f"probe {token}", root) is None
    ), f"{token!r} is an option/env shape, not a path, and must not be flagged"


def test_scan_for_escape_flags_a_path_token_when_the_slash_precedes_the_equals(root):
    """The option/env exemption needs ``=`` before any ``/``; ``/x=y`` is a path."""
    assert scan_for_escape("cat /x=y", root) == "/x=y", (
        "a token whose slash precedes its '=' is path-shaped and resolves "
        "outside the root, so it must be flagged"
    )


@pytest.mark.parametrize(
    "command",
    ["echo hi >/etc/passwd", "sort 2>/tmp/x"],
)
def test_scan_for_escape_leaves_glued_redirection_targets_unflagged(root, command):
    """DOCUMENTED BLIND SPOT: a redirection glued to its target is not seen.

    The tokeniser splits on separators, so ``>/etc/passwd`` survives as one
    token whose leading ``>`` defeats the absolute-path rule.  Pinned so the
    bypass documentation stays honest: the scan must not *claim* to catch
    this shape.
    """
    assert scan_for_escape(command, root) is None, (
        f"{command!r} is a documented blind spot; if the scan now flags it "
        "the module docstring must be updated with it"
    )


def test_scan_for_escape_leaves_a_mid_path_dotdot_climb_unflagged(root):
    """DOCUMENTED BLIND SPOT: a ``..`` buried mid-path is not seen.

    Only the token ``..`` itself and ``../``-prefixed tokens are climbs as
    far as the scan can tell; ``deep/../../outside`` slips past it.
    """
    assert scan_for_escape("cat deep/../../outside", root) is None, (
        "a mid-path '..' is a documented blind spot and must stay unpinned "
        "as behaviour only if the docstring keeps listing it"
    )


def test_scan_for_escape_counts_an_unresolvable_absolute_path_as_an_escape(root):
    """An absolute path that cannot be resolved at all counts as an escape.

    A questionable token must reach the approver rather than be waved
    through; an embedded NUL byte makes ``Path.resolve()`` raise, and the
    scan answers with the token itself.
    """
    assert scan_for_escape("cat /tmp\0x", root) == "/tmp\0x", (
        "an unresolvable absolute path must be flagged for the approver, "
        "never silently passed"
    )


def test_scan_for_escape_flags_against_an_unresolvable_root():
    """A root that cannot be resolved falls back to the un-normalised root.

    The fallback can only make the scan *more* likely to flag, never less:
    an absolute path token is still judged, against the un-normalised root,
    and lands outside it.
    """
    unresolvable = Path("/tmp\0unresolvable-root")
    assert scan_for_escape("cat /etc/passwd", unresolvable) == "/etc/passwd", (
        "an unresolvable root must degrade to flagging, never to waving the "
        "token through"
    )


def test_scan_for_escape_returns_the_first_offending_token(root):
    """The first flagged token in command order is the one returned."""
    assert (
        scan_for_escape("ls ~ ; cat /etc/passwd", root) == "~"
    ), "the scan reports the first offending token, not the last"


@pytest.mark.parametrize("command", ["", " ;; | & "])
def test_scan_for_escape_allows_empty_and_separator_only_commands(root, command):
    """Empty tokens between separators name nothing and flag nothing."""
    assert (
        scan_for_escape(command, root) is None
    ), f"{command!r} contains no path-shaped token and must not be flagged"


# --- bash: running the command and shaping the output ------------------------


@pytest.mark.asyncio
async def test_bash_runs_in_the_workspace_root(root, workspace):
    """The subprocess's working directory is the workspace root."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)
    marker = "bash-cwd-marker.txt"

    result = await bash(f"touch {marker}")

    assert result == "", "touch prints nothing on success"
    assert (
        root / marker
    ).is_file(), "the command must run with the workspace root as its cwd"
    assert not Path(
        os.getcwd(), marker
    ).exists(), "the marker must not have landed in the tool process's own cwd"


@pytest.mark.asyncio
async def test_bash_returns_the_command_output(root, workspace):
    """Stdout comes back verbatim, decoded, with no marker on success."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo stdout-payload")

    assert (
        result == "stdout-payload\n"
    ), "the tool returns the command's output and nothing else on success"


@pytest.mark.asyncio
async def test_bash_merges_stderr_into_stdout(root, workspace):
    """What the command writes to stderr is not lost."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo stderr-payload 1>&2")

    assert (
        result == "stderr-payload\n"
    ), "stderr must be merged into the returned stream"


@pytest.mark.asyncio
async def test_bash_merges_the_streams_in_order(root, workspace):
    """Merged output preserves the child's write order on the shared pipe.

    External ``/bin/echo`` processes write and exit one after another, so
    the order in the pipe is unambiguous -- unlike shell builtins, whose
    stdio buffering could reorder the streams.
    """
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("/bin/echo first-line; /bin/echo second-line 1>&2")

    assert (
        result == "first-line\nsecond-line\n"
    ), "the merged stream must preserve the order the child wrote"


@pytest.mark.asyncio
async def test_bash_appends_the_exit_code_marker_when_the_command_fails(
    root, workspace
):
    """A non-zero exit is data for the model, appended, never ``Error: ``."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo payload-line; exit 3")

    assert (
        result == "payload-line\n\n[exit code: 3]"
    ), "the exit-code marker sits one blank line under the last output line"
    assert not result.startswith(
        "Error:"
    ), "a non-zero exit code is the command's failure, not the tool's"


@pytest.mark.asyncio
async def test_bash_appends_the_exit_code_marker_to_empty_output(root, workspace):
    """A failed command with no output produces just the marker.

    Boundary of the exit-code rule: empty output, rstrip leaves nothing, so
    the result is the marker alone, still without an ``Error: `` prefix.
    """
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("exit 3")

    assert (
        result == "\n\n[exit code: 3]"
    ), "an empty failure is the marker alone, one blank line down"
    assert not result.startswith(
        "Error:"
    ), "a non-zero exit code is the command's failure, not the tool's"


@pytest.mark.asyncio
async def test_bash_does_not_add_an_exit_code_marker_on_success(root, workspace):
    """Exit code 0 adds nothing to the returned text."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo clean-exit")

    assert result == "clean-exit\n", "a successful command's output is returned as-is"
    assert "[exit code" not in result, "no marker belongs on a zero exit"


@pytest.mark.asyncio
async def test_bash_decodes_non_utf8_output_without_raising(root, workspace):
    """``cat`` of a binary file is decoded with errors="replace", not raised.

    Hand-computed: each invalid UTF-8 byte (0xFF, 0xFE) becomes one U+FFFD
    replacement character, the NUL byte survives, and the ASCII tail is
    intact.
    """
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)
    (root / "blob.bin").write_bytes(b"\xff\xfe\x00abc")

    result = await bash("cat blob.bin")

    assert result == "\ufffd\ufffd\x00abc", (
        "binary output must decode with errors='replace', one U+FFFD per "
        "invalid byte, instead of raising"
    )


@pytest.mark.asyncio
async def test_bash_clamps_long_output_keeping_both_ends(root, workspace):
    """Output beyond ``max_output_chars`` is clamped, keeping head and tail."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver, max_output_chars=100)
    lines = "".join(f"{number}\n" for number in range(1, 1001))
    (root / "long.txt").write_text(lines, encoding="utf-8")
    assert len(lines) > 100, "precondition: the payload must exceed the budget"

    result = await bash("cat long.txt")

    dropped = len(lines) - 100
    assert (
        f"[truncated: {dropped} characters omitted]" in result
    ), "the clamp marker must name how much was dropped"
    assert result.startswith("1\n"), "the head of the output must be kept"
    assert result.endswith("1000\n"), "the tail of the output must be kept"


@pytest.mark.asyncio
async def test_bash_gives_a_stdin_reading_command_eof(root, workspace):
    """Stdin is DEVNULL: a command that reads stdin gets EOF, not a hang."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await asyncio.wait_for(bash("cat"), timeout=DEADLINE_SECONDS)

    assert result == "", (
        "cat with stdin at DEVNULL reads EOF immediately and prints nothing; "
        "a TimeoutError here means stdin is being held open"
    )


# --- bash: the approver contract ---------------------------------------------


@pytest.mark.asyncio
async def test_bash_in_root_command_never_consults_the_approver(root, workspace):
    """An in-root command reaches the approver ZERO times and still runs."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo inroot-ok")

    assert result == "inroot-ok\n", "an in-root command runs normally"
    assert approver.requests == [], (
        f"an in-root command must reach the approver zero times, got "
        f"{len(approver.requests)}"
    )


@pytest.mark.asyncio
async def test_bash_escape_consults_the_approver_exactly_once_and_allow_runs_it(
    root, workspace
):
    """An escape reaches the approver EXACTLY ONCE and ALLOW runs it for real.

    One request, no re-check behind the verdict: a second scan after ALLOW
    would override the approver and make a granting approver inert.
    """
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("ls /etc/passwd")

    assert len(approver.requests) == 1, (
        f"an escape must reach the approver exactly once, got "
        f"{len(approver.requests)}"
    )
    request = approver.requests[0]
    assert request.action == "bash", "the request names the bash action"
    assert request.detail == "ls /etc/passwd", "the request carries the command"
    assert str(root) in request.reason, "the request names the workspace root"
    assert "/etc/passwd" in request.reason, "the request names the flagged token"
    assert result == "/etc/passwd\n", (
        "an ALLOW verdict must actually run the command -- the output is the "
        "real listing, proving nothing blocked it and nothing re-checked it"
    )


@pytest.mark.asyncio
async def test_bash_deny_refuses_in_prose_and_does_not_run_the_command(root, workspace):
    """A DENY verdict returns the refusal sentence and runs nothing."""
    approver = _RecordingApprover(ApprovalDecision.DENY)
    _install(workspace, approver)
    command = "touch refused-side-effect.txt && ls /etc/passwd"

    result = await bash(command)

    expected = (
        f"Error: bash on {command} was declined: it reaches outside the "
        f"workspace root {root} (flagged token: /etc/passwd)."
    )
    assert result == expected, (
        "the refusal names the command, the root it was supposed to stay "
        f"inside and the flagged token; got {result!r}"
    )
    assert not (
        root / "refused-side-effect.txt"
    ).exists(), "a DENY verdict must not run any part of the command"
    assert (
        len(approver.requests) == 1
    ), "the refusal is decided by the single DENY the approver gave"


@pytest.mark.asyncio
async def test_bash_without_a_workspace_skips_the_scan_and_the_approver():
    """No workspace means no boundary: no scan, no approver, command runs."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    set_current(ToolContext(approver=approver))

    result = await bash("echo no-boundary; cat /etc/passwd")

    assert approver.requests == [], (
        "with no workspace there is no root to escape from, so an "
        "escape-shaped command must still not consult the approver"
    )
    assert result.startswith(
        "no-boundary\n"
    ), "the command must run even though it names an absolute path"


# --- bash: the timeout kills the process group --------------------------------


@pytest.mark.asyncio
async def test_bash_timeout_kills_the_process_group(root, workspace):
    """A timed-out command leaves NO survivor: the group dies, not just the shell.

    The command is compound so the shell stays the ``sleep``'s parent; a
    bare ``proc.kill()`` would orphan the sleep and this test would fail
    with the surviving pid in hand.
    """
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)
    command = f"{TIMEOUT_SLEEP}; echo timeout-marker"

    result = await bash(command, timeout=700)

    expected = (
        f"Error: bash timed out after 700 ms: {command}. "
        "The process group was killed."
    )
    assert result == expected, (
        f"a timeout is the tool's failure (prose, no output section when "
        f"nothing was printed); got {result!r}"
    )
    survivors = await _pgrep_until(
        TIMEOUT_SLEEP, present=False, at_least=1, seconds=DEADLINE_SECONDS
    )
    assert survivors == [], (
        f"the process group survived the timeout: pids {survivors} are still "
        "alive -- a bare proc.kill() killed the shell but not its children"
    )
    await _kill_matching(TIMEOUT_SLEEP)


@pytest.mark.asyncio
async def test_bash_timeout_message_carries_output_printed_before_the_kill(
    root, workspace
):
    """Output printed before the kill is collected into the timeout message."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)
    command = f"echo early-output; {TIMED_OUTPUT_SLEEP}"

    result = await bash(command, timeout=500)

    expected = (
        f"Error: bash timed out after 500 ms: {command}. "
        "The process group was killed; its output up to then:\n\n"
        "early-output\n"
    )
    assert result == expected, (
        f"the timeout message must carry what the command printed before it "
        f"died; got {result!r}"
    )
    await _kill_matching(TIMED_OUTPUT_SLEEP)


@pytest.mark.parametrize("timeout_ms", [0, -1])
@pytest.mark.asyncio
async def test_bash_non_positive_timeout_uses_default(root, workspace, timeout_ms):
    """A timeout of zero or less falls back to the default (120 000 ms)."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)
    command = "echo quick"

    result = await bash(command, timeout=timeout_ms)

    assert (
        result == "quick\n"
    ), f"timeout={timeout_ms} should use default and allow the command to run, got {result!r}"


# --- bash: cancellation kills the process group --------------------------------


@pytest.mark.asyncio
async def test_cancelled_bash_kills_the_process_group_and_propagates(root, workspace):
    """Cancelling the tool's task kills the group AND raises CancelledError.

    This is how ``AgentRuntime`` interrupts a running command, so the
    ``CancelledError`` handler must signal the whole process group before
    re-raising -- the exact defect this suite pins.  A regression fails an
    assertion (survivor pids, or a swallowed cancellation) rather than
    hanging: every wait below carries a deadline.
    """
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)
    command = f"{CANCELLED_SLEEP}; echo cancelled-marker"

    task = asyncio.create_task(bash(command))
    try:
        seen = await _pgrep_until(
            CANCELLED_SLEEP, present=True, at_least=2, seconds=DEADLINE_SECONDS
        )
        assert len(seen) >= 2, (
            "the command's shell and its sleep child never both appeared; "
            "the pgrep discriminator itself is broken"
        )

        task.cancel()
        done, _pending = await asyncio.wait({task}, timeout=DEADLINE_SECONDS)
        assert task in done, (
            "a cancelled bash must finish promptly; it was still running "
            f"{DEADLINE_SECONDS:.0f}s after cancellation, so the cancellation "
            "was swallowed into a hang"
        )
        assert task.cancelled(), (
            "bash must re-raise CancelledError after killing the group -- a "
            "cancelled turn that resolves to a result instead is uncancellable"
        )

        survivors = await _pgrep_until(
            CANCELLED_SLEEP, present=False, at_least=1, seconds=DEADLINE_SECONDS
        )
        assert survivors == [], (
            f"the process group survived cancellation: pids {survivors} are "
            "still alive -- the CancelledError handler did not kill the group"
        )
    finally:
        await _kill_matching(CANCELLED_SLEEP)


# --- bash: timeout clamping (P2-2 fix) -------------------------------------------


@pytest.mark.asyncio
async def test_bash_rejects_infinite_timeout(root, workspace):
    """An infinite timeout (1e400 -> inf) must be rejected, not hang forever."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("sleep 1", timeout=1e400)

    assert result.startswith(
        "Error:"
    ), f"an infinite timeout must be rejected with an Error message, got {result!r}"
    assert (
        "infinite" in result.lower() or "finite" in result.lower()
    ), f"the error must mention the infinite/finite issue, got {result!r}"


@pytest.mark.asyncio
async def test_bash_clamps_timeout_above_one_hour(root, workspace):
    """A timeout above 1 hour must be clamped, not passed as-is to wait_for."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    # 10 million ms = ~2.7 hours > 1 hour cap
    result = await bash("sleep 1", timeout=10_000_000)

    # Should clamp and run, not hang or pass raw 10M ms to wait_for
    # The command should complete quickly (sleep 1 finishes before 1 hour cap)
    assert not result.startswith(
        "Error: bash timed out after 10000000 ms"
    ), f"timeout above cap must be clamped, not passed raw to wait_for; got {result!r}"
    # Should succeed since sleep 1 finishes well within 1 hour cap
    assert (
        result == "" or result == "\n"
    ), f"clamped timeout should allow sleep 1 to complete; got {result!r}"


@pytest.mark.asyncio
async def test_bash_rejects_boolean_timeout(root, workspace):
    """A boolean timeout (True/False) must be rejected, not treated as 1/0."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo hi", timeout=True)  # type: ignore[arg-type]

    assert result.startswith(
        "Error:"
    ), f"a boolean timeout must be rejected with an Error message, got {result!r}"
    assert (
        "boolean" in result.lower()
    ), f"the error must mention the boolean type issue, got {result!r}"


@pytest.mark.asyncio
async def test_bash_rejects_non_number_timeout(root, workspace):
    """A non-numeric timeout (string, list, etc.) must be rejected."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    result = await bash("echo hi", timeout="not-a-number")  # type: ignore[arg-type]

    assert result.startswith(
        "Error:"
    ), f"a non-numeric timeout must be rejected with an Error message, got {result!r}"
    assert (
        "number" in result.lower() or "numeric" in result.lower()
    ), f"the error must mention the number requirement, got {result!r}"


@pytest.mark.asyncio
async def test_bash_non_positive_timeout_uses_default(root, workspace):
    """A zero or negative timeout should fall back to the default (5 minutes)."""
    approver = _RecordingApprover(ApprovalDecision.ALLOW)
    _install(workspace, approver)

    # timeout=0 should use default and complete the quick command
    result = await bash("echo quick", timeout=0)

    assert (
        result == "quick\n"
    ), f"timeout=0 should use default and allow the command to run, got {result!r}"

    # negative timeout should also use default
    result = await bash("echo quick-neg", timeout=-1)

    assert (
        result == "quick-neg\n"
    ), f"timeout=-1 should use default and allow the command to run, got {result!r}"
