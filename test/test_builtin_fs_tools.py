"""Tests for the five built-in file tools: ``read``, ``write``, ``edit``,
``glob_files`` and ``grep_files``.

The suite is built around the behaviours the tools exist to guarantee, and
five of them pin defects that were recently fixed:

**A named pipe must be refused, never opened.**  ``read_text`` on a FIFO
blocks the worker thread forever, so Read, Edit and Write refuse a
non-regular file in prose naming the object.  Those tests run the call on a
daemon thread with a join deadline (``_finish_within``): with the guard in
place the refusal arrives instantly, and against a regression the test fails
instead of freezing the suite.

**The read byte cap.**  ``limit`` caps lines after the file is in memory, so
a whole-file read is bounded separately: anything above 10 MiB is refused
from its ``st_size`` alone.

**Containment is decided in exactly one place.**  ``paths.resolve`` is the
sole authority: an in-root path reaches the approver ZERO times, an escape
-- including an in-root symlink pointing out -- reaches it EXACTLY ONCE,
and an approved escape is actually operated on.  Every count is asserted
through a recording approver, never just the outcome.

**The documented walk asymmetry.**  The same out-of-workspace symlink that
sends ``read`` to the approver is silently dropped by ``glob_files`` and
``grep_files`` with no approver call, because the walker can only omit a
candidate.  Pinned so that any change to it is deliberate.

**Edit replaces with one ``str.replace``.**  The substring case
(``old_string`` contained in ``new_string``) must terminate;
``"ab".replace("a", "aa")`` is ``"aab"``, not ``"aa"`` -- the expansion
tests run under the same deadline guard.

Expected values are hand-computed from the tools' documented output shapes,
not copied from a run.  Nothing mocks ``open`` or ``pathlib``: every case is
built on ``tmp_path`` so the filesystem itself decides what resolves, and on
macOS every expected path derives from an already-resolved root so the
``/var`` vs ``/private/var`` spellings can never disagree.
"""

import asyncio
import concurrent.futures
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.core.tool.builtin.context as context_module
from mitocode.core.tool.builtin.approval import ApprovalDecision
from mitocode.core.tool.builtin.context import ToolContext, set_current
from mitocode.core.tool.builtin.fs_tools import (
    edit,
    glob_files,
    grep_files,
    read,
    write,
)
from mitocode.core.workspace.workspace import Workspace

#: Seconds a hang-risk call (a FIFO target, an expansion replacement) gets
#: before the test declares a hang.  Generous against a slow machine, far
#: below the "forever" a regression would produce.
DEADLINE_SECONDS = 10.0

#: The documented whole-file read cap: ten mebibytes, refused from
#: ``st_size`` alone before anything is read.
READ_CAP_BYTES = 10 * 1024 * 1024

#: The refusal sentence an escape produces under a denying approver, as
#: documented on ``paths.resolve``: the action, the resolved path and the
#: root, in that order.
DENIAL_TEMPLATE = (
    "Error: {action} on {path} was declined: "
    "it resolves outside the workspace root {root}."
)

#: The five tools, in the order their escape behaviour is parametrized.
TOOL_NAMES = ("read", "write", "edit", "glob", "grep")


# --- doubles and helpers -----------------------------------------------------


class _RecordingApprover:
    """An approver that records every request and returns a fixed decision.

    The tests assert on ``requests`` -- its length and, where it matters, the
    recorded ``action`` -- because the containment ruling is about how many
    times the approver is consulted, not merely about the final sentence.
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
            req: The ``ApprovalRequest`` the chokepoint asked about.

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
        workspace: The workspace whose root bounds every path.
        approver: The policy consulted when a request escapes the root.
        max_output_chars: The clamp budget the tools read from the context.

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


def _make(root: Path, rel: str, content: str = "") -> Path:
    """Create *root* / *rel* with *content*, making parents as needed.

    Args:
        root: The workspace root to build inside.
        rel: The workspace-relative path of the file.
        content: The text to write.

    Returns:
        The created file's path.
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _fifo(path: Path) -> Path:
    """Create a named pipe at *path* and return it.

    Args:
        path: Where the FIFO should be created.

    Returns:
        The same path, for fluent use in tests.
    """
    os.mkfifo(path)
    return path


class _DaemonExecutor(concurrent.futures.Executor):
    """An executor whose workers are daemon threads nothing waits for.

    ``asyncio.to_thread`` runs its work on an executor, and
    ``concurrent.futures`` joins every pooled worker at interpreter exit --
    so a worker blocked forever on a FIFO (the exact regression these tests
    exist to catch) would hang the *suite's exit* even though the deadline
    already failed the test.  This executor spawns a bare daemon thread per
    submission and registers it nowhere: a blocked worker is simply killed
    at exit, and the failed assertion is the last word.
    """

    def __init__(self) -> None:
        """Initialise the base executor (a lock and a shutdown flag)."""
        super().__init__()

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        """Run *fn* on a fresh daemon thread and report through *future*.

        Args:
            fn: The callable the loop wants off the event loop thread.
            *args: Positional arguments for *fn*.
            **kwargs: Keyword arguments for *fn*.

        Returns:
            A future resolved by the worker thread.
        """
        future: concurrent.futures.Future = concurrent.futures.Future()

        def _run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        threading.Thread(target=_run, daemon=True).start()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Do nothing: there are no pooled threads to wait for.

        Args:
            wait: Accepted for the ``Executor`` protocol; ignored.
            cancel_futures: Accepted for the protocol; ignored.
        """


class _DaemonExecutorLoop(asyncio.SelectorEventLoop):
    """A loop whose default-executor work runs on :class:`_DaemonExecutor`.

    ``asyncio.to_thread`` hands its work to ``loop.run_in_executor(None,
    ...)``; this loop substitutes the daemon executor exactly when the
    caller asks for the default one, so a worker blocked forever on a FIFO
    is a daemon thread registered nowhere and cannot hang interpreter exit.
    ``loop.set_default_executor`` cannot be used for this: Python 3.14
    rejects any default executor that is not a ``ThreadPoolExecutor``.
    """

    def run_in_executor(self, executor, func, *args):
        """Route default-executor work to :class:`_DaemonExecutor`.

        Args:
            executor: The caller's executor choice; ``None`` -- the default
                executor -- is the case this loop reroutes.
            func: The callable to run off the event loop thread.
            *args: Positional arguments for *func*.

        Returns:
            A future wrapped for this loop.
        """
        if executor is None:
            executor = _DaemonExecutor()
        return super().run_in_executor(executor, func, *args)


def _finish_within(call, timeout: float = DEADLINE_SECONDS) -> str:
    """Run ``await call()`` on a daemon thread with a join deadline.

    A hang inside the tool -- the exact defect the FIFO and expansion tests
    exist to catch -- would otherwise outlive the assertion that should have
    failed.  The coroutine runs on a :class:`_DaemonExecutorLoop` inside a
    daemon worker, so even a tool worker blocked forever dies at interpreter
    exit instead of blocking it.

    Args:
        call: A zero-argument callable returning a coroutine.
        timeout: Seconds to wait before declaring a hang.

    Returns:
        The coroutine's result, when it finished in time.

    Raises:
        AssertionError: When the call outlives *timeout*.
        BaseException: Whatever the call raised, re-raised on the main thread.
    """
    outcome: dict = {}

    def _target() -> None:
        loop = _DaemonExecutorLoop()
        try:
            outcome["value"] = loop.run_until_complete(call())
        except BaseException as exc:  # re-raised on the main thread below
            outcome["error"] = exc
        finally:
            loop.close()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), (
        f"the tool call did not finish within {timeout}s; it hung where the "
        "documented prose refusal was required"
    )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_module_state():
    """Reset the context module's process-wide state around every test.

    ``_CURRENT`` survives between test files in one pytest process, so a
    leaked registration would hand this file's tools another file's context
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
        A ``Workspace`` whose ``is_within``/``resolve_within`` decide
        containment for every path in these tests.  A real instance rather
        than a double: the containment ruling is precisely that the tools
        defer to this boundary and add no check of their own.
    """
    return Workspace(root, "fs-tools-tests", 2)


@pytest.fixture(name="spy")
def spy_fixture():
    """A recording approver that allows every escape.

    Returns:
        A fresh ``_RecordingApprover`` per test, so request counts never leak
        between cases.
    """
    return _RecordingApprover(ApprovalDecision.ALLOW)


@pytest.fixture(name="walk_tree")
def walk_tree_fixture(root):
    """A tree exercising every walk pruning rule at once.

    Layout: ``a.txt``, ``c.txt``, ``sub/b.txt`` as content; ``m.txt`` as a
    *directory* matching ``*.txt``; ``.git/HEAD``, ``.hg/x``, ``.svn/y``,
    ``__pycache__/z.pyc`` as never-entered directories; ``.hidden/secret.txt``
    as a dot-directory that only a dot-starting pattern may reach.

    Args:
        root: The resolved root from :func:`root_fixture`.

    Returns:
        The root, populated.
    """
    _make(root, "a.txt", "a\n")
    _make(root, "c.txt", "c\n")
    _make(root, "sub/b.txt", "b\n")
    (root / "m.txt").mkdir()
    for skip in (".git", ".hg", ".svn", "__pycache__", ".hidden"):
        (root / skip).mkdir()
    _make(root, ".git/HEAD", "ref\n")
    _make(root, ".hg/x", "x\n")
    _make(root, ".svn/y", "y\n")
    _make(root, "__pycache__/z.pyc", "z\n")
    _make(root, ".hidden/secret.txt", "s\n")
    return root


@pytest.fixture(name="grep_tree")
def grep_tree_fixture(root):
    """Three files whose single first lines all contain ``hello``.

    Args:
        root: The resolved root from :func:`root_fixture`.

    Returns:
        The root, populated.
    """
    _make(root, "code.py", "# hello\n")
    _make(root, "note_a.txt", "hello world\n")
    _make(root, "note_b.txt", "say hello\n")
    return root


@pytest.fixture(name="escape")
def escape_fixture(root):
    """An outside directory reached only through in-root symlinks.

    Layout: ``outside/loot.txt`` (content ``outside treasure\\n``) reached by
    the file symlink ``link_out.txt``; ``outside/fresh.txt`` (absent) reached
    by ``link_fresh.txt``; ``outside/editable.txt`` reached by
    ``link_edit.txt``; and the directory symlink ``outlink`` reaching the
    whole outside tree for the walked tools.

    Args:
        root: The resolved root from :func:`root_fixture`.

    Returns:
        A namespace with ``outside_dir``, ``loot``, ``fresh`` and ``editable``
        -- all resolved-spelling absolute paths.
    """
    outside_dir = root.parent / "outside"
    outside_dir.mkdir(exist_ok=True)
    loot = outside_dir / "loot.txt"
    loot.write_text("outside treasure\n", encoding="utf-8")
    fresh = outside_dir / "fresh.txt"
    editable = outside_dir / "editable.txt"
    editable.write_text("keep this line\n", encoding="utf-8")
    (root / "link_out.txt").symlink_to(loot)
    (root / "link_fresh.txt").symlink_to(fresh)
    (root / "link_edit.txt").symlink_to(editable)
    (root / "outlink").symlink_to(outside_dir, target_is_directory=True)
    return SimpleNamespace(
        outside_dir=outside_dir, loot=loot, fresh=fresh, editable=editable
    )


# --- read --------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [0, 1])
async def test_read_numbers_lines_from_one_and_offset_zero_and_one_agree(
    root, workspace, spy, offset
):
    """Offset 0 and offset 1 both start at the first line, numbered from 1."""
    _install(workspace, spy)
    _make(root, "note.txt", "alpha\nbeta\ngamma\n")

    result = await read("note.txt", offset=offset)

    assert (
        result == "1\talpha\n2\tbeta\n3\tgamma"
    ), f"offset={offset} must read the whole file numbered from 1, got {result!r}"


@pytest.mark.asyncio
async def test_read_window_keeps_absolute_line_numbers(root, workspace, spy):
    """A windowed read reports absolute positions, not window-relative ones."""
    _install(workspace, spy)
    _make(root, "five.txt", "l1\nl2\nl3\nl4\nl5\n")

    result = await read("five.txt", offset=2, limit=2)

    assert (
        result == "2\tl2\n3\tl3"
    ), f"the window must keep the file's own line numbers, got {result!r}"


@pytest.mark.asyncio
async def test_read_limit_caps_the_number_of_lines(root, workspace, spy):
    """``limit`` bounds the returned window to that many lines."""
    _install(workspace, spy)
    _make(root, "five.txt", "l1\nl2\nl3\nl4\nl5\n")

    result = await read("five.txt", limit=2)

    assert (
        result == "1\tl1\n2\tl2"
    ), f"limit=2 must return exactly the first two lines, got {result!r}"


@pytest.mark.asyncio
async def test_read_offset_equal_to_the_line_count_still_reads_the_last_line(
    root, workspace, spy
):
    """The boundary: offset == line count is the last line, not past the end."""
    _install(workspace, spy)
    _make(root, "three.txt", "l1\nl2\nl3\n")

    result = await read("three.txt", offset=3)

    assert (
        result == "3\tl3"
    ), f"offset 3 of a 3-line file is its last line, got {result!r}"


@pytest.mark.asyncio
async def test_read_offset_past_the_end_reports_the_line_count(root, workspace, spy):
    """An offset beyond EOF is an error that says how many lines there are."""
    _install(workspace, spy)
    _make(root, "three.txt", "l1\nl2\nl3\n")

    result = await read("three.txt", offset=9)

    assert result == (
        "Error: offset 9 is past the end of three.txt: it has 3 lines. "
        "Offsets count from 1; use offset 1 to start from the first line."
    ), f"the past-the-end refusal must state the file's line count, got {result!r}"


@pytest.mark.asyncio
async def test_read_missing_file_is_an_error_not_empty_output(root, workspace, spy):
    """A missing file is refused in prose; it never reads as an empty success."""
    _install(workspace, spy)

    result = await read("ghost.txt")

    assert (
        result == "Error: read failed: ghost.txt does not exist."
    ), f"a missing file must be named as such, got {result!r}"


@pytest.mark.asyncio
async def test_read_empty_file_is_distinguishable_from_a_missing_one(
    root, workspace, spy
):
    """An empty file and a missing file produce two different error sentences."""
    _install(workspace, spy)
    _make(root, "hollow.txt", "")

    result = await read("hollow.txt")

    assert (
        result == "Error: hollow.txt is empty; there is nothing to read."
    ), f"an empty file must be reported as empty, got {result!r}"
    assert (
        result != "Error: read failed: ghost.txt does not exist."
    ), "empty and missing must stay distinguishable"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -3])
async def test_read_refuses_a_limit_below_one(root, workspace, spy, limit):
    """A non-positive limit is refused, naming the value that arrived."""
    _install(workspace, spy)
    _make(root, "note.txt", "alpha\nbeta\n")

    result = await read("note.txt", limit=limit)

    assert result == (
        f"Error: read of note.txt: limit must be at least 1, got {limit}."
    ), f"limit={limit} must be refused with the documented sentence, got {result!r}"


@pytest.mark.asyncio
async def test_read_negative_offset_starts_at_the_first_line(root, workspace, spy):
    """Anything below 1 for offset means the first line."""
    _install(workspace, spy)
    _make(root, "note.txt", "alpha\nbeta\n")

    result = await read("note.txt", offset=-5)

    assert (
        result == "1\talpha\n2\tbeta"
    ), f"a negative offset must read from the first line, got {result!r}"


def test_read_refuses_a_named_pipe_by_name_without_hanging(root, workspace, spy):
    """A FIFO is refused in prose naming the object, before anything opens."""
    _install(workspace, spy)
    _fifo(root / "pipe.fifo")

    result = _finish_within(lambda: read("pipe.fifo"))

    assert result == (
        "Error: read refused: pipe.fifo is a named pipe (fifo), not a regular "
        "file; only regular files can be read as text."
    ), f"the FIFO must be refused by name, got {result!r}"


@pytest.mark.asyncio
async def test_read_refuses_a_directory(root, workspace, spy):
    """A directory target is refused before any open, naming the object."""
    _install(workspace, spy)
    (root / "sub").mkdir()

    result = await read("sub")

    assert result == (
        "Error: read refused: sub is a directory, not a regular file; only "
        "regular files can be read as text."
    ), f"a directory must be refused by name, got {result!r}"


@pytest.mark.asyncio
async def test_read_refuses_a_file_above_the_cap_from_its_size_alone(
    root, workspace, spy
):
    """A regular file above the 10 MiB cap is refused before anything is read."""
    _install(workspace, spy)
    big = root / "big.bin"
    big.write_bytes(b"x" * (READ_CAP_BYTES + 1))

    result = await read("big.bin")

    assert (
        big.stat().st_size == READ_CAP_BYTES + 1
    ), "precondition: the file is one byte above the cap"
    assert result == (
        f"Error: read refused: big.bin is {READ_CAP_BYTES + 1} bytes, above "
        f"the {READ_CAP_BYTES}-byte read cap; a file this large is never "
        f"loaded whole."
    ), f"the cap refusal must state both sizes, got {result!r}"


@pytest.mark.asyncio
async def test_read_allows_a_file_exactly_at_the_cap(root, workspace, spy):
    """The cap is exclusive: exactly 10 MiB is a legitimate read."""
    _install(workspace, spy)
    big = root / "edge.bin"
    big.write_bytes(b"x" * READ_CAP_BYTES)

    result = await read("edge.bin")

    assert not result.startswith(
        "Error: "
    ), f"a file exactly at the cap must be read, got {result!r}"
    assert result.startswith(
        "1\t"
    ), f"the read must come back as numbered lines, got {result[:20]!r}"


@pytest.mark.asyncio
async def test_read_clamps_output_to_the_context_budget(root, workspace, spy):
    """An oversized read carries the documented truncation marker."""
    _install(workspace, spy, max_output_chars=40)
    _make(root, "wide.txt", "0123456789\n" * 5)

    result = await read("wide.txt")

    assert result.startswith(
        "1\t0123456789"
    ), f"the clamped output keeps its head, got {result[:20]!r}"
    assert (
        "[truncated:" in result
    ), f"an output above the budget must carry the truncation marker, got {result!r}"


# --- write -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_creates_a_file_and_reports_the_character_count(
    root, workspace, spy
):
    """A successful create reports the exact character count."""
    _install(workspace, spy)

    result = await write("made.txt", "abc")

    assert (
        result == "Wrote 3 characters to made.txt."
    ), f"the success sentence must report the character count, got {result!r}"
    assert (root / "made.txt").read_text(
        encoding="utf-8"
    ) == "abc", "the file must hold exactly the content that was written"


@pytest.mark.asyncio
async def test_write_creates_missing_parent_directories(root, workspace, spy):
    """Parents are created as needed, and the sentence names the full path."""
    _install(workspace, spy)

    result = await write("deep/nested/made.txt", "abc")

    assert (
        result == "Wrote 3 characters to deep/nested/made.txt."
    ), f"parents must be created and the path named in full, got {result!r}"
    assert (root / "deep" / "nested" / "made.txt").read_text(
        encoding="utf-8"
    ) == "abc", "the nested file must hold exactly the content that was written"


@pytest.mark.asyncio
async def test_write_refuses_to_clobber_and_names_edit(root, workspace, spy):
    """An existing file is refused, and the refusal points at Edit."""
    _install(workspace, spy)
    _make(root, "note.txt", "original\n")

    result = await write("note.txt", "clobber\n")

    assert result == (
        "Error: write refused: note.txt already exists. Write never "
        "overwrites: read the file and use Edit to change it, because "
        "replacing a whole file to make one change destroys the rest without "
        "a trace."
    ), f"the clobber refusal must name Edit, got {result!r}"
    assert (root / "note.txt").read_text(
        encoding="utf-8"
    ) == "original\n", "a refused write must leave the existing file byte-identical"


@pytest.mark.asyncio
async def test_write_empty_content_creates_an_empty_file(root, workspace, spy):
    """Empty content is legitimate and produces a 0-byte file."""
    _install(workspace, spy)

    result = await write("hollow.txt", "")

    assert (
        result == "Wrote 0 characters to hollow.txt."
    ), f"empty content must succeed with a zero count, got {result!r}"
    assert (root / "hollow.txt").stat().st_size == 0, "the created file must be empty"


@pytest.mark.asyncio
async def test_write_through_a_broken_symlink_lands_on_the_target(root, workspace, spy):
    """A broken in-root link resolves away, and the write names the target.

    ``resolve`` collapses the symlink before Write sees it, so the write is
    never "through" anything untracked: the file is created at the link's
    target and the success sentence names the TARGET, never the link.
    """
    _install(workspace, spy)
    (root / "broken.txt").symlink_to(root / "never_target.txt")

    result = await write("broken.txt", "boom\n")

    assert (
        result == "Wrote 5 characters to never_target.txt."
    ), f"the sentence must name the resolved target, not the link, got {result!r}"
    assert (root / "never_target.txt").read_text(
        encoding="utf-8"
    ) == "boom\n", "the file must exist at the link's target"
    assert (
        spy.requests == []
    ), f"an in-root target must not reach the approver, got {spy.requests!r}"


@pytest.mark.asyncio
async def test_write_refuses_a_symlink_loop_target(root, workspace, spy):
    """A symlink loop survives resolution as a symlink, and Write refuses it.

    The documented ``is_symlink`` branch: a loop comes back un-resolved (on
    3.10-3.12) or partially resolved (on 3.13+), so the path Write is handed
    is still a symlink and writing through it would be untrackable.  A loop
    is also an escape, so it reaches the approver exactly once first.
    """
    _install(workspace, spy)
    (root / "loop_a").symlink_to(root / "loop_b")
    (root / "loop_b").symlink_to(root / "loop_a")

    result = await write("loop_a", "x")

    assert len(spy.requests) == 1, (
        "a loop is unresolvable and must be escalated exactly once, "
        f"got {len(spy.requests)}"
    )
    assert result.startswith(
        "Error: write refused: loop_a already exists."
    ), f"a symlink loop target must be refused, got {result!r}"
    assert not (root / "loop_a").exists(), "nothing may be written through a loop"


def test_write_refuses_a_named_pipe_by_name_without_hanging(root, workspace, spy):
    """An existing FIFO is refused as a non-regular file, naming the object."""
    _install(workspace, spy)
    _fifo(root / "pipe.fifo")

    result = _finish_within(lambda: write("pipe.fifo", "x"))

    assert result == (
        "Error: write refused: pipe.fifo is a named pipe (fifo), not a "
        "regular file; Write only creates regular files."
    ), f"the FIFO must be refused by name, got {result!r}"


@pytest.mark.asyncio
async def test_write_refuses_a_directory_target(root, workspace, spy):
    """A directory target is refused; nothing is written inside it."""
    _install(workspace, spy)
    (root / "sub").mkdir()

    result = await write("sub", "x")

    assert (
        result == "Error: write refused: sub is a directory."
    ), f"a directory target must be refused, got {result!r}"


# --- edit --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_replaces_one_occurrence_and_reports_the_line_movement(
    root, workspace, spy
):
    """A unique match is replaced once, and the sentence reports the count."""
    _install(workspace, spy)
    _make(root, "lines.txt", "alpha\nbeta\ngamma\n")

    result = await edit("lines.txt", "beta", "BETA")

    assert result == (
        "Replaced 1 occurrence in lines.txt: the file went from 3 to 3 lines."
    ), f"the success sentence must report one occurrence, got {result!r}"
    assert (root / "lines.txt").read_text(encoding="utf-8") == (
        "alpha\nBETA\ngamma\n"
    ), "the file must hold the replacement"


@pytest.mark.asyncio
async def test_edit_replaces_across_lines_in_one_exact_swap(root, workspace, spy):
    """``old_string`` may span lines; the swap reports the line-count move."""
    _install(workspace, spy)
    _make(root, "swap.txt", "a\nb\nc\n")

    result = await edit("swap.txt", "a\nb", "X")

    assert result == (
        "Replaced 1 occurrence in swap.txt: the file went from 3 to 2 lines."
    ), f"a cross-line swap must report the line movement, got {result!r}"
    assert (root / "swap.txt").read_text(
        encoding="utf-8"
    ) == "X\nc\n", "the file must hold the collapsed replacement"


@pytest.mark.asyncio
async def test_edit_ambiguous_match_reports_the_count_and_names_replace_all(
    root, workspace, spy
):
    """More than one match without replace_all is refused with the count."""
    _install(workspace, spy)
    _make(root, "dups.txt", "dup\ndup\n")

    result = await edit("dups.txt", "dup", "unique")

    assert result == (
        "Error: edit on dups.txt: old_string matches 2 places. Include more "
        "surrounding text to make it unique, or pass replace_all to replace "
        "every occurrence."
    ), f"the ambiguity refusal must state the count, got {result!r}"
    assert (root / "dups.txt").read_text(
        encoding="utf-8"
    ) == "dup\ndup\n", "a refused ambiguous edit must leave the file byte-identical"


@pytest.mark.asyncio
async def test_edit_replace_all_replaces_every_occurrence(root, workspace, spy):
    """With replace_all every occurrence goes, and the count is reported."""
    _install(workspace, spy)
    _make(root, "many.txt", "a b a b b a\n")

    result = await edit("many.txt", "a", "z", replace_all=True)

    assert result == (
        "Replaced 3 occurrences in many.txt: the file went from 1 to 1 lines."
    ), f"replace_all must report every occurrence, got {result!r}"
    assert (root / "many.txt").read_text(
        encoding="utf-8"
    ) == "z b z b b z\n", "every occurrence must be replaced"


@pytest.mark.asyncio
async def test_edit_no_match_suggests_the_closest_lines(root, workspace, spy):
    """Zero matches produce an error quoting the most similar existing lines."""
    _install(workspace, spy)
    _make(root, "poem.txt", "apple pie\nbanana bread\n")

    result = await edit("poem.txt", "aple", "REPLACED")

    assert result.startswith(
        "Error: edit on poem.txt: old_string was not found."
    ), f"the no-match refusal must say so, got {result!r}"
    assert (
        "'apple pie'" in result
    ), f"the closest line must be suggested, got {result!r}"
    assert (root / "poem.txt").read_text(encoding="utf-8") == (
        "apple pie\nbanana bread\n"
    ), "a refused edit must leave the file byte-identical"


@pytest.mark.asyncio
async def test_edit_on_an_empty_file_reports_not_found_and_empty(root, workspace, spy):
    """An empty file with a non-empty old_string reports both facts."""
    _install(workspace, spy)
    _make(root, "hollow.txt", "")

    result = await edit("hollow.txt", "anything", "x")

    assert result == (
        "Error: edit on hollow.txt: old_string was not found and the file " "is empty."
    ), f"the empty-file refusal must state both facts, got {result!r}"


@pytest.mark.asyncio
async def test_edit_refuses_an_empty_old_string(root, workspace, spy):
    """The empty string matches between every character and is refused."""
    _install(workspace, spy)
    _make(root, "note.txt", "stable\n")

    result = await edit("note.txt", "", "x")

    assert (
        result == "Error: edit on note.txt: old_string must not be empty."
    ), f"an empty old_string must be refused, got {result!r}"
    assert (root / "note.txt").read_text(
        encoding="utf-8"
    ) == "stable\n", "the file must be untouched"


@pytest.mark.asyncio
async def test_edit_refuses_identical_old_and_new(root, workspace, spy):
    """An edit that would change nothing is refused."""
    _install(workspace, spy)
    _make(root, "note.txt", "same text\n")

    result = await edit("note.txt", "same", "same")

    assert result == (
        "Error: edit on note.txt: new_string is identical to old_string; "
        "nothing would change."
    ), f"an identical old/new pair must be refused, got {result!r}"
    assert (root / "note.txt").read_text(
        encoding="utf-8"
    ) == "same text\n", "the file must be untouched"


def test_edit_substring_replacement_terminates_and_replaces_once(root, workspace, spy):
    """old_string contained in new_string must terminate: one str.replace."""
    _install(workspace, spy)
    _make(root, "grow.txt", "one a two\n")

    result = _finish_within(lambda: edit("grow.txt", "a", "aa"))

    assert result == (
        "Replaced 1 occurrence in grow.txt: the file went from 1 to 1 lines."
    ), f"the substring replacement must succeed, got {result!r}"
    assert (root / "grow.txt").read_text(
        encoding="utf-8"
    ) == "one aa two\n", 'str.replace semantics: "one a two" becomes "one aa two"'


def test_edit_substring_replace_all_terminates(root, workspace, spy):
    """The expansion case under replace_all terminates and expands, not loops.

    ``"ab".replace("a", "aa")`` is ``"aab"`` -- each original occurrence is
    replaced exactly once; the produced text is never rescanned.
    """
    _install(workspace, spy)
    _make(root, "abab.txt", "abab\n")

    result = _finish_within(lambda: edit("abab.txt", "a", "aa", True))

    assert result == (
        "Replaced 2 occurrences in abab.txt: the file went from 1 to 1 lines."
    ), f"the expansion must replace the two original occurrences, got {result!r}"
    assert (root / "abab.txt").read_text(
        encoding="utf-8"
    ) == "aabaab\n", (
        'str.replace semantics: "abab" becomes "aabaab", never "aa" or a hang'
    )


def test_edit_refuses_a_named_pipe_by_name_without_hanging(root, workspace, spy):
    """A FIFO target is refused before the read that would block forever."""
    _install(workspace, spy)
    _fifo(root / "pipe.fifo")

    result = _finish_within(lambda: edit("pipe.fifo", "a", "b"))

    assert result == (
        "Error: edit on pipe.fifo refused: it is a named pipe (fifo), not a "
        "regular file; Edit only rewrites regular files."
    ), f"the FIFO must be refused by name, got {result!r}"


@pytest.mark.asyncio
async def test_edit_refuses_a_file_that_is_not_valid_utf8(root, workspace, spy):
    """A file Edit cannot decode is refused, and its bytes are untouched."""
    _install(workspace, spy)
    raw = b"caf\xe9 menu"
    (root / "latin.txt").write_bytes(raw)

    result = await edit("latin.txt", "caf", "tea")

    assert result == (
        "Error: edit on latin.txt failed: the file is not valid UTF-8 text; "
        "Edit refuses to rewrite a file it cannot decode."
    ), f"a non-UTF-8 file must be refused, got {result!r}"
    assert (
        root / "latin.txt"
    ).read_bytes() == raw, "the file's bytes must be untouched"


@pytest.mark.asyncio
async def test_edit_missing_file_is_an_error(root, workspace, spy):
    """A missing target is refused in prose naming the path."""
    _install(workspace, spy)

    result = await edit("ghost.txt", "a", "b")

    assert (
        result == "Error: edit failed: ghost.txt does not exist."
    ), f"a missing file must be named as such, got {result!r}"


# --- glob --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_lists_matching_files_sorted_and_omits_directories(walk_tree):
    """Matches come back sorted, workspace-relative, and never directories."""
    _install(Workspace(walk_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await glob_files("*.txt")

    assert (
        result == "a.txt\nc.txt\nsub/b.txt"
    ), f"matches must be sorted relative paths, got {result!r}"
    assert "m.txt" not in result, "a directory matching the pattern is not listed"


@pytest.mark.asyncio
async def test_glob_skips_git_and_dot_directories(walk_tree):
    """Version-control internals and dot-directories are never entered."""
    _install(Workspace(walk_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await glob_files("*")

    assert (
        result == "a.txt\nc.txt\nsub/b.txt"
    ), f"only files outside skipped directories may be listed, got {result!r}"
    for banned in (".git", ".hg", ".svn", "__pycache__", ".hidden", "HEAD"):
        assert (
            banned not in result
        ), f"{banned} must never appear in the walk output, got {result!r}"


@pytest.mark.asyncio
async def test_glob_honours_gitignore_negation_with_last_match_wins(root):
    """``!keep.log`` after ``*.log`` keeps keep.log: the last rule decides."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, ".gitignore", "*.log\n!keep.log\n")
    _make(root, "a.log", "a\n")
    _make(root, "keep.log", "k\n")
    _make(root, "note.txt", "n\n")

    result = await glob_files("*.log")

    assert (
        result == "keep.log"
    ), f"the negated rule must keep keep.log and drop a.log, got {result!r}"


@pytest.mark.asyncio
async def test_glob_honours_a_gitignore_rule_containing_a_slash(root):
    """A slash rule matches the walk-relative posix path, not the basename."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, ".gitignore", "sub/*.tmp\n")
    _make(root, "sub/x.tmp", "x\n")
    _make(root, "sub/keep.txt", "k\n")

    excluded = await glob_files("*.tmp")
    kept = await glob_files("*.txt")

    assert (
        excluded == "No files matched '*.tmp' under .."
    ), f"the slash rule must exclude sub/x.tmp, got {excluded!r}"
    assert (
        kept == "sub/keep.txt"
    ), f"files the rule does not name must still be listed, got {kept!r}"


@pytest.mark.asyncio
async def test_glob_no_matches_says_so_without_an_error_prefix(walk_tree):
    """An empty result is the documented sentence, not an 'Error: ' string."""
    _install(Workspace(walk_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await glob_files("*.zzz")

    assert (
        result == "No files matched '*.zzz' under .."
    ), f"the no-match sentence is not an error, got {result!r}"


@pytest.mark.asyncio
async def test_glob_dot_pattern_unlocks_hidden_directories(root):
    """A pattern starting with a dot walks dot-directories; a plain one does not."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, ".config/settings.txt", "s\n")
    _make(root, "note.txt", "n\n")

    plain = await glob_files("settings.txt")
    widened = await glob_files(".config/*")

    assert (
        plain == "No files matched 'settings.txt' under .."
    ), f"a non-dot pattern must not see .config/, got {plain!r}"
    assert (
        widened == ".config/settings.txt"
    ), f"a dot-starting pattern must unlock .config/, got {widened!r}"


@pytest.mark.asyncio
async def test_glob_refuses_an_empty_pattern(walk_tree):
    """The empty pattern is refused before any walk."""
    _install(Workspace(walk_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await glob_files("")

    assert (
        result == "Error: glob: pattern must not be empty."
    ), f"an empty pattern must be refused, got {result!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "no_such_dir",
            "Error: glob failed: no_such_dir does not exist, so "
            "there is nothing to walk.",
        ),
        ("note.txt", "Error: glob on note.txt failed: it is not a directory."),
    ],
    ids=["missing", "not-a-directory"],
)
async def test_glob_refuses_a_missing_or_non_directory_root(root, path, expected):
    """A root that is missing or not a directory is refused in prose."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, "note.txt", "n\n")

    result = await glob_files("*.txt", path)

    assert result == expected, f"the root refusal must be exact, got {result!r}"


@pytest.mark.asyncio
async def test_glob_clamps_output_to_the_context_budget(root):
    """An oversized listing carries the documented truncation marker."""
    _install(
        Workspace(root, "fs-tools-tests", 2),
        _RecordingApprover(),
        max_output_chars=5,
    )
    _make(root, "aaa.txt", "a\n")
    _make(root, "bbb.txt", "b\n")

    result = await glob_files("*.txt")

    assert (
        "[truncated:" in result
    ), f"an output above the budget must carry the truncation marker, got {result!r}"


# --- grep --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_with_matches_lists_each_matching_path_once(grep_tree):
    """The default mode lists every matching path once, in walk order."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("hello")

    assert (
        result == "code.py\nnote_a.txt\nnote_b.txt"
    ), f"each matching path must appear once, sorted, got {result!r}"


@pytest.mark.asyncio
async def test_grep_content_mode_reports_path_line_number_and_text(grep_tree):
    """Content mode emits ``path:lineno:line`` per matching line."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("hello", output_mode="content")

    assert result == (
        "code.py:1:# hello\nnote_a.txt:1:hello world\nnote_b.txt:1:say hello"
    ), f"content mode must be path:lineno:line, got {result!r}"


@pytest.mark.asyncio
async def test_grep_count_mode_reports_the_matching_lines_per_file(grep_tree):
    """Count mode emits ``path:n`` where n counts matching lines."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("hello", output_mode="count")

    assert (
        result == "code.py:1\nnote_a.txt:1\nnote_b.txt:1"
    ), f"count mode must be path:n per file, got {result!r}"


@pytest.mark.asyncio
async def test_grep_glob_filter_restricts_which_files_are_read(grep_tree):
    """The glob argument is a basename filter on which files are read."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    only_py = await grep_files("hello", glob="*.py")
    only_txt = await grep_files("hello", glob="*.txt")

    assert only_py == "code.py", f"the filter must restrict to *.py, got {only_py!r}"
    assert (
        only_txt == "note_a.txt\nnote_b.txt"
    ), f"the filter must restrict to *.txt, got {only_txt!r}"


@pytest.mark.asyncio
async def test_grep_head_limit_truncates_the_output(grep_tree):
    """head_limit caps output lines in every mode that produces them."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    paths = await grep_files("hello", head_limit=2)
    content = await grep_files("hello", output_mode="content", head_limit=2)

    assert (
        paths == "code.py\nnote_a.txt"
    ), f"head_limit=2 must keep a stable sorted prefix, got {paths!r}"
    assert (
        content == "code.py:1:# hello\nnote_a.txt:1:hello world"
    ), f"content mode must truncate to the same stable prefix, got {content!r}"


@pytest.mark.asyncio
async def test_grep_head_limit_below_one_yields_the_no_matches_error(grep_tree):
    """A non-positive head_limit truncates to nothing, which is the error."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("hello", head_limit=0)

    assert (
        result == "Error: no matches for 'hello' under .."
    ), f"head_limit=0 must truncate to the no-matches error, got {result!r}"


@pytest.mark.asyncio
async def test_grep_case_insensitive_matches_without_regard_to_case(root):
    """case_insensitive widens matching; the default stays case-sensitive."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, "soft.txt", "hello world\n")
    _make(root, "shout.txt", "HELLO world\n")

    sensitive = await grep_files("hello")
    insensitive = await grep_files("hello", case_insensitive=True)

    assert (
        sensitive == "soft.txt"
    ), f"the default must stay case-sensitive, got {sensitive!r}"
    assert (
        insensitive == "shout.txt\nsoft.txt"
    ), f"case_insensitive must match both spellings, got {insensitive!r}"


@pytest.mark.asyncio
async def test_grep_invalid_regex_falls_back_to_a_substring_match(root):
    """A pattern re cannot compile degrades to a literal substring search."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, "odd.txt", "value h(llo end\n")

    paths = await grep_files("h(llo")
    content = await grep_files("h(llo", output_mode="content")

    assert paths == "odd.txt", f"the fallback must still find the text, got {paths!r}"
    assert (
        content == "odd.txt:1:value h(llo end"
    ), f"the fallback must match the literal text, got {content!r}"


@pytest.mark.asyncio
async def test_grep_skips_binary_files_and_marks_them_in_content_mode(root):
    """A NUL in the first 8 KiB makes a file binary: skipped, marked in content."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, "hit.txt", "hello\n")
    (root / "bin.dat").write_bytes(b"hello\x00world")

    paths = await grep_files("hello")
    content = await grep_files("hello", output_mode="content")
    count = await grep_files("hello", output_mode="count")

    assert (
        paths == "hit.txt"
    ), f"a binary file must be skipped in files_with_matches, got {paths!r}"
    assert (
        content == "bin.dat: [binary]\nhit.txt:1:hello"
    ), f"content mode must note the skipped binary, got {content!r}"
    assert (
        count == "hit.txt:1"
    ), f"count mode must skip the binary silently, got {count!r}"


@pytest.mark.asyncio
async def test_grep_treats_a_nul_after_the_sniff_window_as_text(root):
    """A NUL beyond the first 8192 bytes does not make a file binary."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    (root / "big.bin").write_bytes(b"a" * 8192 + b"\x00tail\nneedle\n")

    result = await grep_files("needle", output_mode="content")

    assert (
        result == "big.bin:2:needle"
    ), f"a NUL past the sniff window must leave the file searchable, got {result!r}"


@pytest.mark.asyncio
async def test_grep_dot_starting_pattern_widens_the_walk_into_hidden_directories(
    root,
):
    """The dot rule is the pattern's own: '.' widens, 'needle' does not."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _make(root, ".hidden/h.txt", "needle hidden\n")
    _make(root, "visible.txt", "needle visible\n")

    widened = await grep_files(".")
    narrowed = await grep_files("needle")

    assert (
        widened == ".hidden/h.txt\nvisible.txt"
    ), f"a dot-starting pattern must search .hidden/ too, got {widened!r}"
    assert (
        narrowed == "visible.txt"
    ), f"a non-dot pattern must not enter .hidden/, got {narrowed!r}"


@pytest.mark.asyncio
async def test_grep_refuses_an_unknown_output_mode(grep_tree):
    """A direct call with an undocumented output_mode is refused."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("hello", output_mode="bogus")

    assert result == (
        "Error: grep: unknown output_mode 'bogus'; use 'files_with_matches', "
        "'content' or 'count'."
    ), f"an unknown output_mode must be refused, got {result!r}"


@pytest.mark.asyncio
async def test_grep_refuses_an_empty_pattern(grep_tree):
    """The empty pattern is refused before any walk."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("")

    assert (
        result == "Error: grep: pattern must not be empty."
    ), f"an empty pattern must be refused, got {result!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "no_such_dir",
            "Error: grep failed: no_such_dir does not exist, so "
            "there is nothing to search.",
        ),
        ("note_a.txt", "Error: grep on note_a.txt failed: it is not a directory."),
    ],
    ids=["missing", "not-a-directory"],
)
async def test_grep_refuses_a_missing_or_non_directory_root(grep_tree, path, expected):
    """A root that is missing or not a directory is refused in prose."""
    _install(Workspace(grep_tree, "fs-tools-tests", 2), _RecordingApprover())

    result = await grep_files("hello", path)

    assert result == expected, f"the root refusal must be exact, got {result!r}"


def test_grep_skips_a_named_pipe_without_hanging(root):
    """A FIFO in the tree is never opened; the walk simply omits it."""
    _install(Workspace(root, "fs-tools-tests", 2), _RecordingApprover())
    _fifo(root / "pipe.fifo")
    _make(root, "note.txt", "needle here\n")

    result = _finish_within(lambda: grep_files("needle"))

    assert (
        result == "note.txt"
    ), f"the walk must skip the FIFO and still match the text file, got {result!r}"


@pytest.mark.asyncio
async def test_grep_clamps_output_to_the_context_budget(grep_tree):
    """An oversized result carries the documented truncation marker."""
    _install(
        Workspace(grep_tree, "fs-tools-tests", 2),
        _RecordingApprover(),
        max_output_chars=20,
    )

    result = await grep_files("hello")

    assert (
        "[truncated:" in result
    ), f"an output above the budget must carry the truncation marker, got {result!r}"


# --- containment: one chokepoint, one approver --------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOL_NAMES)
async def test_in_root_paths_reach_the_approver_zero_times(root, workspace, spy, tool):
    """Ordinary in-root work must never consult the approver, for any tool."""
    _install(workspace, spy)
    _make(root, "note.txt", "hello there\n")

    if tool == "read":
        result = await read("note.txt")
    elif tool == "write":
        result = await write("fresh.txt", "new\n")
    elif tool == "edit":
        result = await edit("note.txt", "hello", "goodbye")
    elif tool == "glob":
        result = await glob_files("*.txt")
    else:
        result = await grep_files("hello")

    assert spy.requests == [], (
        f"{tool}: an in-root path must reach the approver zero times, "
        f"got {spy.requests!r}"
    )
    assert not result.startswith(
        "Error: "
    ), f"{tool}: in-root work must succeed, got {result!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOL_NAMES)
async def test_escalated_escapes_reach_the_approver_exactly_once_and_are_honoured(
    root, workspace, spy, escape, tool
):
    """An in-root symlink pointing out is one request, and the verdict rules."""
    _install(workspace, spy)

    if tool == "read":
        result = await read("link_out.txt")
        assert (
            "outside treasure" in result
        ), f"read: an approved escape must be actually read, got {result!r}"
    elif tool == "write":
        result = await write("link_fresh.txt", "written through the link\n")
        assert escape.fresh.read_text(encoding="utf-8") == (
            "written through the link\n"
        ), "write: the approved escape must create the outside file"
    elif tool == "edit":
        result = await edit("link_edit.txt", "keep this line", "kept line")
        assert (
            escape.editable.read_text(encoding="utf-8") == "kept line\n"
        ), "edit: the approved escape must change the outside file"
    elif tool == "glob":
        result = await glob_files("*.txt", "outlink")
        assert (
            str(escape.loot) in result
        ), f"glob: the approved outside root must be walked, got {result!r}"
    else:
        result = await grep_files("treasure", "outlink")
        assert result == str(
            escape.loot
        ), f"grep: the approved outside root must be searched, got {result!r}"

    assert len(spy.requests) == 1, (
        f"{tool}: an escape must reach the approver exactly once, "
        f"got {len(spy.requests)}"
    )
    assert (
        spy.requests[0].action == tool
    ), f"{tool}: the request must name the action that was attempted"
    assert not result.startswith(
        "Error: "
    ), f"{tool}: an allowed escape must not come back as a refusal, got {result!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOL_NAMES)
async def test_denied_escapes_are_refused_in_prose_and_change_nothing(
    root, workspace, escape, tool
):
    """A denied escape yields the documented refusal and leaves everything put."""
    denying = _RecordingApprover(ApprovalDecision.DENY)
    _install(workspace, denying)
    targets = {
        "read": escape.loot,
        "write": escape.fresh,
        "edit": escape.editable,
        "glob": escape.outside_dir,
        "grep": escape.outside_dir,
    }

    if tool == "read":
        result = await read("link_out.txt")
    elif tool == "write":
        result = await write("link_fresh.txt", "must not land\n")
    elif tool == "edit":
        result = await edit("link_edit.txt", "keep this line", "kept line")
    elif tool == "glob":
        result = await glob_files("*.txt", "outlink")
    else:
        result = await grep_files("treasure", "outlink")

    assert (
        len(denying.requests) == 1
    ), f"{tool}: a denial still escalates exactly once, got {len(denying.requests)}"
    assert result == DENIAL_TEMPLATE.format(
        action=tool, path=targets[tool], root=root
    ), f"{tool}: the refusal must name the action, path and root, got {result!r}"

    if tool == "write":
        assert (
            not escape.fresh.exists()
        ), "write: a denied escape must create nothing outside the root"
    elif tool == "edit":
        assert (
            escape.editable.read_text(encoding="utf-8") == "keep this line\n"
        ), "edit: a denied escape must leave the outside file byte-identical"


# --- the documented walk asymmetry --------------------------------------------


@pytest.mark.asyncio
async def test_read_escalates_but_the_walks_drop_the_same_symlink(
    root, workspace, spy, escape
):
    """The same out-link is one approver call for read, zero for the walks.

    Deliberate, documented behaviour: the walker's containment comparison
    scopes what the walk enumerates and can only omit a candidate, so
    ``glob_files`` and ``grep_files`` drop the out-of-workspace symlink
    silently -- no approver call at all -- while ``read`` of the very same
    path is an escape decided by the approver.
    """
    _install(workspace, spy)

    escalated = await read("link_out.txt")
    assert (
        len(spy.requests) == 1
    ), f"read: the symlink out of the root is an escape, got {spy.requests!r}"
    assert (
        "outside treasure" in escalated
    ), f"read: the approved escape is honoured, got {escalated!r}"

    spy.requests.clear()
    walked = await glob_files("*.txt")
    assert (
        walked == "No files matched '*.txt' under .."
    ), f"glob: the outside symlink must be dropped, got {walked!r}"
    assert (
        spy.requests == []
    ), f"glob: the walk must never consult the approver, got {spy.requests!r}"

    searched = await grep_files("treasure")
    assert (
        searched == "Error: no matches for 'treasure' under .."
    ), f"grep: the outside symlink must be dropped, got {searched!r}"
    assert (
        spy.requests == []
    ), f"grep: the walk must never consult the approver, got {spy.requests!r}"
