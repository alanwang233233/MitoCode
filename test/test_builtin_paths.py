"""Unit tests for ``paths.resolve``, the single approval chokepoint for file operations.

Two properties carry the module and are the reason these tests exist rather than a
smoke test of the happy path:

**The approver is reached only on escape.**  An in-root path must never produce
an ``ApprovalRequest``.  That is structural — the shipped design reserves a
future dialog for escapes so that ordinary work cannot slow to a keystroke — so
it is proven with a recording approver that counts: exactly zero requests for
every in-root spelling, exactly one for every escape spelling.  Asserting only
that the happy path returns a non-``None`` path would pass against an
implementation that asked the approver every time.

**Containment is decided by ``is_within``, not ``resolve_within``.**  On Python
3.13+ ``Path.resolve()`` no longer raises on a symlink loop: it returns the path
with the looping component left in place, still spelled under the root, so a bare
containment check accepts it.  (Probed, not assumed: on 3.10.20
``Path.resolve`` raises ``RuntimeError``; on 3.14.0 it returns
``<root>/loop_a`` and ``resolve_within`` *succeeds* — only ``is_within``
answers False.  ``test_a_symlink_loop_reaches_the_approver`` is the test that
fails if the two are ever swapped.)

Two expectations are deliberately written against ``Path(...).resolve()`` rather
than a literal spelling: on macOS ``/etc`` is a symlink to ``/private/etc``, so a
literal comparison would pass on Linux CI and fail here.

Expected values are hand-computed from the module's docstrings — the refusal
template is copied out of ``resolve``'s own documented example, not out of its
output.  Nothing here mocks ``open`` or ``pathlib``; every path case is built on
``tmp_path`` so the filesystem itself decides what resolves.
"""

import asyncio
import errno
import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.exception.mitocode_exception import MitoCodeWorkspacePathError
from mitocode.core.tool.builtin.approval import (
    AlwaysAllow,
    ApprovalDecision,
    ApprovalRequest,
    DenyAll,
)
from mitocode.core.tool.builtin.context import ToolContext
from mitocode.core.tool.builtin.paths import resolve
from mitocode.core.workspace.workspace import Workspace

#: The refusal sentence documented in ``resolve``'s docstring, parameterised by
#: the three things it names.  Copied from the documented example — "Error: read
#: on /etc/hostname was declined: it resolves outside the workspace root /proj."
#: — rather than from the string the function happens to build, so a change in
#: wording fails a test instead of being adopted by it.
REFUSAL_TEMPLATE = (
    "Error: {action} on {path} was declined: "
    "it resolves outside the workspace root {root}."
)

#: Ids of :func:`_in_root_cases`, written out so ``pytest.mark.parametrize`` can
#: name them.  A key added to the table without a matching entry here is a
#: ``KeyError`` rather than a silently untested case.
IN_ROOT_CASE_IDS = (
    "relative",
    "absolute-path-object",
    "absolute-string",
    "dot",
    "empty-string",
    "root-as-absolute-string",
    "root-as-path-object",
    "dot-prefixed",
    "dotdot-landing-back-inside",
    "symlink-inside-pointing-inside",
    "symlink-inside-pointing-inside-absolute",
    "not-created-yet",
    "subdirectory",
)

#: Ids of :func:`_escape_cases`, same relationship to the table.
ESCAPE_CASE_IDS = (
    "traversal",
    "absolute-outside-string",
    "absolute-outside-path-object",
    "symlink-inside-pointing-outside",
    "dangling-symlink-pointing-outside",
    "symlink-loop",
    "embedded-nul-byte",
)


class _RecordingApprover:
    """An approver that records every request and returns a fixed decision.

    The ``asyncio.sleep(0)`` is load-bearing rather than decorative: a verdict
    produced only after a suspension point can be consumed by an ``await`` and by
    nothing else.  So every escape test in this file also proves that ``resolve``
    really suspends on ``ctx.approver.request(...)`` — an implementation that
    called it without awaiting would hold a coroutine, fail the ``is ALLOW``
    identity check and refuse every escape, which is exactly what the allow-path
    assertions catch.
    """

    def __init__(self, decision: ApprovalDecision = ApprovalDecision.ALLOW) -> None:
        """Record requests and answer with *decision*.

        Args:
            decision: The verdict every recorded request receives.
        """
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        """Record *req* and return the configured verdict after one suspension.

        Args:
            req: The request ``resolve`` asked about.

        Returns:
            The decision this approver was built with.
        """
        self.requests.append(req)
        await asyncio.sleep(0)
        return self.decision


class _TripwireApprover:
    """An approver that fails the test if it is ever consulted."""

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        """Fail the test: an in-root target must not need a verdict.

        Args:
            req: Unused; the call itself is the failure.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError(
            f"the approver was consulted for an in-root path: {req!r}"
        )


class _FlakyWorkspace:
    """A workspace that passes containment and then refuses to resolve the same target.

    Stands in for the documented race in which the filesystem changes between
    ``is_within`` and ``resolve_within``.  Only the surface ``resolve`` uses is
    implemented — ``path``, ``is_within``, ``resolve_within`` — because
    ``ToolContext`` reads the workspace structurally rather than by type.
    """

    def __init__(self, path: Path, error: BaseException) -> None:
        """Bind the root and the error the second call will raise.

        Args:
            path: The workspace root ``ToolContext.root()`` reads.
            error: The exception ``resolve_within`` will raise.
        """
        self.path = path
        self.error = error
        self.resolve_within_targets: list[Path] = []

    def is_within(self, target: Path) -> bool:
        """Accept everything: the race is in the second call, not this one.

        Args:
            target: The candidate ``resolve`` checked.

        Returns:
            Always ``True``.
        """
        return True

    def resolve_within(self, target: Path) -> Path:
        """Record the candidate and raise the configured error.

        Args:
            target: The candidate that passed the containment check.

        Raises:
            BaseException: The exception this double was built with.
        """
        self.resolve_within_targets.append(target)
        raise self.error


@pytest.fixture(name="root")
def root_fixture(tmp_path):
    """A resolved workspace root populated with every path shape under test.

    The root is resolved up front so that expectations can be written against
    it directly: on macOS ``tmp_path`` is reached through ``/var`` while
    ``Path.cwd()`` reports ``/private/var``, and a test that mixed the two
    spellings would be asserting a symlink expansion rather than a path.

    Returns:
        The workspace root directory, with these children:

        * ``src/main.py`` — an ordinary in-root file.
        * ``hostname`` — a decoy with the same basename as ``/etc/hostname``, so
          a re-rooted absolute target would be caught.
        * ``src/link_to_main`` — a symlink to an in-root file.
        * ``escape_link`` — a symlink to ``../outside.txt``.
        * ``dangling_escape`` — a symlink to ``../never_created.txt``.
        * ``loop_a`` / ``loop_b`` — a symlink loop inside the root.
        * ``src/new_file.py`` — deliberately absent (the Write-a-new-file case).
    """
    root = (tmp_path / "ws").resolve()
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "hostname").write_text("decoy inside the root\n", encoding="utf-8")
    (root / "src" / "link_to_main").symlink_to(root / "src" / "main.py")
    (root.parent / "outside.txt").write_text("secret\n", encoding="utf-8")
    (root / "escape_link").symlink_to(root.parent / "outside.txt")
    (root / "dangling_escape").symlink_to(root.parent / "never_created.txt")
    (root / "loop_a").symlink_to(root / "loop_b")
    (root / "loop_b").symlink_to(root / "loop_a")
    return root


@pytest.fixture(name="workspace")
def workspace_fixture(root):
    """A real ``Workspace`` rooted at *root*.

    Args:
        root: The directory from :func:`root_fixture`.

    Returns:
        A ``Workspace`` whose root bounds every path in these tests.  A real
        instance rather than a double: ``resolve``'s whole job is to defer to
        ``is_within``/``resolve_within``, and stubbing those would test the stub.
    """
    return Workspace(root, "paths-tests", 2)


@pytest.fixture(name="spy")
def spy_fixture():
    """A recording approver that allows everything.

    Returns:
        A fresh :class:`_RecordingApprover`; parametrized tests each get their own.
    """
    return _RecordingApprover(ApprovalDecision.ALLOW)


def _context(workspace, approver) -> ToolContext:
    """Build a context over *workspace* with *approver* as its policy.

    Args:
        workspace: The bounding workspace, or ``None`` for no boundary.
        approver: The approver ``resolve`` may consult on escape.

    Returns:
        A ``ToolContext`` wired to the two collaborators ``resolve`` reads.
    """
    return ToolContext(workspace=workspace, approver=approver)


def _root_of(workspace) -> Path:
    """Return the resolved root ``ToolContext`` would compute for *workspace*.

    Args:
        workspace: The workspace whose root is wanted.

    Returns:
        ``workspace.path`` resolved, matching ``ToolContext.root()``.
    """
    return Path(workspace.path).resolve()


def _in_root_cases(root):
    """Every spelling that must be answered in-root, with the path each resolves to.

    Each expected value is ``root``-relative by hand: ``Path("")`` and
    ``Path(".")`` both name the root, ``"src/../src/main.py"`` normalises to
    ``src/main.py``, and a symlink resolves to its target rather than staying a
    symlink.  ``"src/new_file.py"`` is absent from disk on purpose: a model about
    to create a file must not be asked for approval to do so.

    Args:
        root: The resolved workspace root.

    Returns:
        A mapping of case id to ``(target, expected path)``.
    """
    return {
        "relative": ("src/main.py", root / "src/main.py"),
        "absolute-path-object": (root / "src/main.py", root / "src/main.py"),
        "absolute-string": (str(root / "src/main.py"), root / "src/main.py"),
        "dot": (".", root),
        "empty-string": ("", root),
        "root-as-absolute-string": (str(root), root),
        "root-as-path-object": (root, root),
        "dot-prefixed": ("./src/main.py", root / "src/main.py"),
        "dotdot-landing-back-inside": ("src/../src/main.py", root / "src/main.py"),
        "symlink-inside-pointing-inside": (
            "src/link_to_main",
            root / "src/main.py",
        ),
        "symlink-inside-pointing-inside-absolute": (
            str(root / "src/link_to_main"),
            root / "src/main.py",
        ),
        "not-created-yet": ("src/new_file.py", root / "src/new_file.py"),
        "subdirectory": ("docs", root / "docs"),
    }


def _escape_cases(root):
    """Every spelling that must be escalated, with the path the approver is told about.

    Expectations are written as ``Path(...).resolve()`` because the documented
    contract is a *fully resolved* path, and resolution is platform-dependent:
    ``/etc/hostname`` is ``/private/etc/hostname`` on macOS.  The two unresolvable
    cases — the symlink loop and the NUL byte — expect the candidate back
    un-normalised, which is what ``_resolve_best_effort`` promises.  Both spellings
    were checked on CPython 3.10.20 (loop: ``RuntimeError``) and 3.14.0 (loop:
    resolved with the component left in place) and agree on ``<root>/loop_a``.

    Args:
        root: The resolved workspace root.

    Returns:
        A mapping of case id to ``(target, expected path)``.
    """
    return {
        "traversal": ("../../etc/passwd", (root / "../../etc/passwd").resolve()),
        "absolute-outside-string": (
            "/etc/hostname",
            Path("/etc/hostname").resolve(),
        ),
        "absolute-outside-path-object": (
            Path("/etc/hostname"),
            Path("/etc/hostname").resolve(),
        ),
        "symlink-inside-pointing-outside": (
            "escape_link",
            (root.parent / "outside.txt").resolve(),
        ),
        "dangling-symlink-pointing-outside": (
            "dangling_escape",
            (root.parent / "never_created.txt").resolve(),
        ),
        "symlink-loop": ("loop_a", root / "loop_a"),
        "embedded-nul-byte": ("in\0side.txt", root / "in\0side.txt"),
    }


# ──────────────────────────────────────────────────────────────
# the signature every later file tool calls
# ──────────────────────────────────────────────────────────────


def test_resolve_is_a_coroutine_with_the_signature_the_call_sites_use():
    """The dialog has to be awaitable later, so the shape is part of the contract.

    ``async`` is not incidental: every file tool calls ``await resolve(...)``,
    and turning it synchronous to save nothing would break all of them at the
    moment somebody adds the approver dialog that the await exists for.
    """
    signature = inspect.signature(resolve)

    assert inspect.iscoroutinefunction(resolve), (
        "resolve must stay awaitable; a future approver is a dialog that waits "
        "for a keystroke"
    )
    assert list(signature.parameters) == ["ctx", "target", "action"], (
        "the parameter names are what keyword call sites depend on, got "
        f"{list(signature.parameters)}"
    )


# ──────────────────────────────────────────────────────────────
# in-root traffic: answered, never asked about
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", IN_ROOT_CASE_IDS)
async def test_resolve_returns_the_in_root_path_without_asking_the_approver(
    workspace, spy, root, case_id
):
    """Every in-root spelling resolves to its expected path and records no request.

    The count is the assertion that matters: ``spy.requests == []`` after each
    case, not merely "the path came back".  A gate that asked about every path
    would pass a return-value-only test and would turn a future dialog into a
    prompt per keystroke.
    """
    target, expected = _in_root_cases(root)[case_id]

    path, error = await resolve(_context(workspace, spy), target, "read")

    assert error is None, (
        f"{case_id}: an in-root target must not produce an error, got {error!r}"
    )
    assert path == expected, f"{case_id}: must resolve to {expected}, got {path!r}"
    assert spy.requests == [], (
        f"{case_id}: an in-root target must not reach the approver, "
        f"got {spy.requests!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "approver",
    [object(), _TripwireApprover()],
    ids=["approver-with-no-request-attribute", "approver-that-raises-if-asked"],
)
async def test_resolve_answers_in_root_traffic_with_no_working_approver_at_all(
    workspace, root, approver
):
    """In-root work needs nothing at all from the policy — not even a valid object.

    Stronger than counting requests: a context whose approver has no ``request``
    attribute and a context whose approver explodes if awaited must both resolve
    ordinary work.  Anything that consulted the approver before the containment
    verdict — a "check the policy first" refactor, say — fails here immediately.
    """
    ctx = _context(workspace, approver)

    path, error = await resolve(ctx, "src/main.py", "read")

    assert error is None, f"in-root work must not fail: {error!r}"
    assert path == root / "src/main.py", (
        "the containment verdict alone decides the path"
    )


@pytest.mark.asyncio
async def test_resolve_ignores_a_verdict_it_was_never_given(workspace, root):
    """An approver that would deny everything cannot change what in-root work returns.

    If the gate consulted the approver on in-root traffic, this is the test that
    fails: the path would come back ``None`` and the error would name a refusal
    nobody asked for.
    """
    denying = _RecordingApprover(ApprovalDecision.DENY)

    path, error = await resolve(_context(workspace, denying), "src/main.py", "read")

    assert path == root / "src/main.py", "the approver must not have vetoed the path"
    assert error is None, "no request means no verdict means no refusal"
    assert denying.requests == [], "the denying approver must never have been asked"


@pytest.mark.asyncio
async def test_resolve_returns_the_in_root_path_under_the_shipped_default_approver(
    workspace, root
):
    """``ToolContext``'s own default is enough to resolve ordinary work.

    No approver is injected here, which is what makes this the shipped
    configuration rather than a configured one: the built-in tools run under
    ``AlwaysAllow`` and must never feel it.
    """
    ctx = ToolContext(workspace=workspace)

    path, error = await resolve(ctx, "src/main.py", "read")

    assert isinstance(ctx.approver, AlwaysAllow), (
        "precondition: the default approver is AlwaysAllow"
    )
    assert path == root / "src/main.py", (
        "in-root work is unaffected by the default policy"
    )
    assert error is None, (
        f"the default policy must not refuse in-root work, got {error!r}"
    )


# ──────────────────────────────────────────────────────────────
# escapes: escalated exactly once, and the approver decides
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ESCAPE_CASE_IDS)
async def test_resolve_asks_the_approver_exactly_once_about_each_escape(
    workspace, spy, root, case_id
):
    """Every escape spelling produces one request, naming the resolved path.

    Exactly one: an implementation that asked per candidate segment, or retried
    after a denial, would turn one human decision into several dialogs.
    """
    target, expected = _escape_cases(root)[case_id]

    path, error = await resolve(_context(workspace, spy), target, "read")

    assert len(spy.requests) == 1, (
        f"{case_id}: exactly one approval request, got {len(spy.requests)}"
    )
    assert spy.requests[0].action == "read", f"{case_id}: the request names the action"
    assert spy.requests[0].detail == str(expected), (
        f"{case_id}: the request must name the resolved path {expected}"
    )
    assert path == expected, f"{case_id}: an allowed escape returns the resolved path"
    assert error is None, (
        f"{case_id}: an allowed escape produces no error, got {error!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ESCAPE_CASE_IDS)
async def test_resolve_refuses_every_escape_under_an_approver_that_denies(
    workspace, root, case_id
):
    """Under ``DenyAll`` the same escapes return no path and a refusal that reads as prose.

    The refusal is compared whole, not by substring: it is the only thing the
    model sees, and a truncated or reordered sentence still reads as a refusal
    while no longer telling the model which operation, which path or which root
    was involved.
    """
    target, expected = _escape_cases(root)[case_id]

    path, error = await resolve(_context(workspace, DenyAll()), target, "write")

    assert path is None, f"{case_id}: a denied escape must not hand back a usable path"
    assert error == REFUSAL_TEMPLATE.format(
        action="write", path=expected, root=root
    ), f"{case_id}: the refusal must name the action, the resolved path and the root"


@pytest.mark.asyncio
async def test_resolve_returns_the_path_when_the_shipped_default_permits_an_escape(
    workspace,
):
    """An escape is permitted, not refused, under the shipped default — by design.

    ``../../etc/passwd`` and ``/etc/hostname`` are not special cases: they reach
    the approver like any other escape, and the shipped policy allows it.  A test
    asserting a refusal here would contradict
    ``test_resolve_refuses_every_escape_under_an_approver_that_denies`` and would
    pin behaviour the module explicitly reserves for a dialog that does not exist
    yet.
    """
    ctx = ToolContext(workspace=workspace)

    path, error = await resolve(ctx, "../../etc/passwd", "read")

    assert error is None, f"the shipped default permits escapes, got {error!r}"
    assert path is not None, "an allowed escape yields the path the model asked about"


@pytest.mark.asyncio
async def test_resolve_asks_about_an_absolute_target_without_re_rooting_it(
    workspace, root
):
    """``/etc/hostname`` is judged as ``/etc/hostname``, not as ``<root>/hostname``.

    The fixture holds an in-root file literally named ``hostname``.  An
    implementation that joined every target to the root would answer about that
    decoy instead, and the model would be told it may read a workspace file when
    it asked about a system one.
    """
    decoy = root / "hostname"

    path, error = await resolve(_context(workspace, DenyAll()), "/etc/hostname", "read")

    assert decoy.exists(), "precondition: an in-root file of the same name exists"
    assert decoy != Path("/etc/hostname").resolve(), (
        "precondition: the decoy is a different path than the requested one"
    )
    assert path is None, "an absolute target outside the root is refused"
    assert error == REFUSAL_TEMPLATE.format(
        action="read", path=Path("/etc/hostname").resolve(), root=root
    ), "the refusal must describe the path that was asked for"


@pytest.mark.asyncio
async def test_resolve_treats_a_sibling_extending_the_root_name_as_an_escape(
    workspace, root
):
    """``<root>_evil`` is outside the root even though it starts with the root's name.

    Containment is per path component.  A string-prefix check would accept every
    sibling whose name extends the root's — ``ws_evil``, ``ws_backup`` — and hand
    the model a file outside the workspace while reporting it as inside.
    """
    sibling_file = root.parent / "ws_evil" / "loot.txt"
    sibling_file.parent.mkdir()
    sibling_file.write_text("secret\n", encoding="utf-8")
    assert str(sibling_file).startswith(str(root)), (
        "precondition: the two paths share a string prefix, so a startswith check "
        "would wrongly accept this"
    )

    path, error = await resolve(
        _context(workspace, DenyAll()), str(sibling_file), "read"
    )

    assert path is None, "a sibling sharing the root's prefix is not inside the root"
    assert error == REFUSAL_TEMPLATE.format(
        action="read", path=sibling_file, root=root
    ), "the refusal must name the sibling path the model asked for"


@pytest.mark.asyncio
async def test_resolve_carries_the_resolved_path_the_action_and_the_root_in_the_request(
    workspace, spy, root
):
    """The request a future dialog renders names the action, the path and the boundary.

    ``detail`` is the concrete path, so a human can answer sensibly; ``reason``
    is asserted only as far as the docstrings specify.  ``resolve``'s docstring
    documents the refusal sentence but never ``ApprovalRequest.reason`` itself, so
    the wording is pinned no further than "it says where the boundary is" — the
    exact phrasing is an open question, recorded here rather than guessed at.
    """
    target = "escape_link"
    expected = (root.parent / "outside.txt").resolve()

    await resolve(_context(workspace, spy), target, "grep")

    assert len(spy.requests) == 1, "one request, one dialog"
    request = spy.requests[0]
    assert isinstance(request, ApprovalRequest), (
        "the approver is handed an ApprovalRequest"
    )
    assert request.action == "grep", "the action is passed through verbatim"
    assert request.detail == str(expected), "the detail is the resolved path"
    assert isinstance(request.reason, str), "the reason is prose for a human"
    assert str(root) in request.reason, (
        f"the reason must name the boundary, got {request.reason!r}"
    )
    assert "workspace root" in request.reason, (
        f"the reason must say what the boundary is, got {request.reason!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["frobnicate", ""], ids=["unrecognised-verb", "empty-verb"]
)
async def test_resolve_passes_an_unvalidated_action_through_to_approver_and_refusal(
    workspace, root, action
):
    """The verb is displayed, never validated, so an unknown one is inert.

    The docstring is explicit that this is deliberate: a whitelist here would be a
    second source of truth for which verbs exist, alongside the tool definitions.
    What must still hold is that the verb is not silently dropped or rewritten —
    the approver is asked about the operation the model asked for, and the model
    is told which operation was refused.
    """
    denying = _RecordingApprover(ApprovalDecision.DENY)
    expected = (root.parent / "outside.txt").resolve()

    path, error = await resolve(_context(workspace, denying), "escape_link", action)

    assert len(denying.requests) == 1, "an escape is still escalated whatever the verb"
    assert denying.requests[0].action == action, (
        "the verb reaches the approver verbatim"
    )
    assert path is None, "the escape is refused by the denying policy"
    assert error == REFUSAL_TEMPLATE.format(
        action=action, path=expected, root=root
    ), "the verb is quoted verbatim, including when it is empty"


@pytest.mark.asyncio
async def test_resolve_names_the_resolved_root_when_the_workspace_path_is_a_symlink(
    tmp_path,
):
    """The refusal names ``ctx.root()``, which is resolved, not the raw workspace path.

    Both spellings name the same directory, so either reads correctly to a human;
    the resolved one is what a model comparing the refusal against a later path
    needs, and it is the one the module's own root-resolution contract promises.
    """
    real_root = (tmp_path / "real").resolve()
    real_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    assert str(alias) != str(real_root), (
        "precondition: the alias is not the physical path"
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")

    ctx = _context(Workspace(alias, "aliased", 2), DenyAll())

    path, error = await resolve(ctx, "../outside.txt", "read")

    assert path is None, "a traversal out of an aliased root is still an escape"
    assert error == REFUSAL_TEMPLATE.format(
        action="read", path=outside.resolve(), root=real_root
    ), "the refusal must name the resolved root, not the symlink the caller passed"
    assert str(alias) not in error, (
        f"the alias is a spelling, not a location; the model must be given {real_root}"
    )


# ──────────────────────────────────────────────────────────────
# the symlink loop: the case is_within exists for
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_asks_the_approver_about_a_symlink_loop_instead_of_accepting_it(
    workspace, spy, root
):
    """A loop is in-root yet unusable, so it must escalate rather than be taken at its word.

    The discriminator between the two containment entry points.  On Python 3.13+
    ``Path.resolve()`` no longer raises on a loop: it returns the path with the
    looping component left in place, still spelled under the root, so
    ``resolve_within`` *accepts* it (verified on CPython 3.14.0).  An
    implementation that gated on ``resolve_within`` inside a try/except — or on a
    bare containment test — would hand the model a path that can only ever fail
    with ELOOP and would never consult a human about it.  ``is_within`` answers
    False on every supported version, so this test fails the moment the predicate
    is swapped.
    """
    loop = root / "loop_a"

    path, error = await resolve(_context(workspace, spy), "loop_a", "read")

    assert loop.is_symlink(), "precondition: the fixture really is a symlink loop"
    assert len(spy.requests) == 1, (
        "a loop is unresolvable, so it must be escalated; accepting it would "
        "hand the model a path that only ever fails with ELOOP"
    )
    assert path == loop, "an approved loop comes back spelled as the model wrote it"
    assert error is None, f"the recording approver allowed it, got {error!r}"


@pytest.mark.asyncio
async def test_resolve_refuses_a_symlink_loop_with_a_reason_the_module_documents(
    workspace, root
):
    """A denied loop is refused in the same words as a real escape — and that is known.

    The module docstring records this inaccuracy: collapsing "resolves outside the
    root" and "sits inside the root but cannot be resolved" into one branch makes
    the reason misdescribe an in-root target.  It is pinned here on purpose so the
    sentence reaches the interactive approver milestone as a checked state rather
    than as a surprise, and so fixing it later is a deliberate edit.
    """
    loop = root / "loop_a"

    path, error = await resolve(_context(workspace, DenyAll()), "loop_a", "read")

    assert path is None, "a denied loop yields no usable path"
    assert error == REFUSAL_TEMPLATE.format(action="read", path=loop, root=root), (
        "the refusal currently claims the loop resolves outside the root, which the "
        "module docstring tracks as a known inaccuracy"
    )


@pytest.mark.asyncio
async def test_resolve_hands_back_an_approved_loop_un_normalised_so_open_still_fails(
    workspace, root
):
    """An approved unresolvable escape is returned as written, and is still unopenable.

    The module's argument for collapsing both refusals into one branch: a loop is
    inside the root, so nothing outside it is reachable through one, and returning
    the path costs the model a precise ``ELOOP`` at ``open()`` instead of a
    refusal that would misreport an in-root file as an escape.
    """
    loop = root / "loop_a"

    path, error = await resolve(_context(workspace, AlwaysAllow()), "loop_a", "read")

    assert error is None, f"AlwaysAllow permits the escape, got {error!r}"
    assert path == loop, (
        "an unresolvable target comes back as the candidate, un-normalised"
    )
    with pytest.raises(OSError) as excinfo:
        open(path).close()
    assert excinfo.value.errno == errno.ELOOP, (
        "the point of returning the loop path is that open() names the real "
        f"failure; got errno {excinfo.value.errno}"
    )


# ──────────────────────────────────────────────────────────────
# no workspace: no boundary, nothing to escape from
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id",
    [
        "relative",
        "absolute-outside",
        "traversal",
        "symlink-loop",
        "nul-relative",
        "nul-absolute",
    ],
)
async def test_resolve_without_a_workspace_answers_every_path_and_asks_nothing(
    tmp_path, monkeypatch, case_id
):
    """With no workspace every target is trivially in-root, including real escapes.

    Documented behaviour, not a hole: the boundary *is* the workspace root, and
    with no root there is nothing to enforce — which is also why the approver is
    unreachable on this branch.  The approver here denies everything, so if it were
    consulted even one of these would come back refused.
    """
    cwd = tmp_path.resolve()
    monkeypatch.chdir(cwd)
    loop = cwd / "loop_a"
    loop.symlink_to(cwd / "loop_b")
    (cwd / "loop_b").symlink_to(loop)
    cases = {
        "relative": ("src/main.py", cwd / "src/main.py"),
        "absolute-outside": ("/etc/hostname", Path("/etc/hostname").resolve()),
        "traversal": ("../../etc/passwd", (cwd / "../../etc/passwd").resolve()),
        # Unresolvable, so the candidate comes back as the caller spelled it.
        "symlink-loop": ("loop_a", cwd / "loop_a"),
        "nul-relative": ("in\0side.txt", cwd / "in\0side.txt"),
        "nul-absolute": ("/in\0side.txt", Path("/in\0side.txt")),
    }
    target, expected = cases[case_id]
    ctx = ToolContext(
        workspace=None, approver=_RecordingApprover(ApprovalDecision.DENY)
    )
    assert ctx.workspace is None, "precondition: the boundary is absent"

    path, error = await resolve(ctx, target, "read")

    assert error is None, (
        f"{case_id}: no boundary means nothing to refuse, got {error!r}"
    )
    assert path == expected, f"{case_id}: must resolve to {expected}, got {path!r}"
    assert ctx.approver.requests == [], (
        f"{case_id}: with no workspace there is no escape to approve"
    )


@pytest.mark.asyncio
async def test_resolve_without_a_workspace_raises_when_the_working_directory_is_gone(
    tmp_path, monkeypatch,
):
    """A deleted cwd fails at ``root()``, before any candidate or verdict exists.

    ``ToolContext.root()`` falls back to ``Path.cwd()``, and ``Path.cwd()`` has
    nothing left to read when the process is sitting in a directory that has been
    removed.  Nothing is returned in that case: the root is needed before a
    candidate even exists, and the approver must not be asked about a path that
    was never built.
    """
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    spy = _RecordingApprover(ApprovalDecision.ALLOW)

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        await resolve(_context(None, spy), "src/main.py", "read")

    assert spy.requests == [], "no root means no candidate means no request"


# ──────────────────────────────────────────────────────────────
# documented escapes and failures
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", [None, 123, b"src/main.py"], ids=["none", "int", "bytes"]
)
async def test_resolve_propagates_type_error_for_a_target_path_cannot_take(
    workspace, spy, target
):
    """A non-path target is a bug in the calling tool, and stays an exception.

    The docstring draws the line deliberately: a *refusal* is prose for the model,
    but a ``TypeError`` here means the tool passed something ``Path`` cannot
    accept, which ``errors.tool_boundary`` renders as ``"Error: TypeError: ..."``.
    Swallowing it into a refusal would tell the model its argument was refused
    when in fact the tool was called wrongly.
    """
    with pytest.raises(TypeError, match="os.PathLike"):
        await resolve(_context(workspace, spy), target, "read")

    assert spy.requests == [], "a target that never became a path cannot be an escape"


@pytest.mark.asyncio
async def test_resolve_propagates_an_approver_failure_verbatim(workspace, root):
    """An approver that raises is not wrapped, not swallowed, and not turned into prose.

    The ``Approver`` protocol disclaims raising, so an approver that breaks is a
    bug in the approver — and it reaches the model through ``tool_boundary`` like
    any other escaped exception.  The identity assertion is the point: a gate that
    caught and re-rendered it would report a refusal the approver never gave.
    """
    boom = RuntimeError("approver wire is down")

    class _BrokenApprover:
        """An approver that honours no part of the protocol but its signature."""

        def __init__(self) -> None:
            self.requests: list[ApprovalRequest] = []

        async def request(self, req: ApprovalRequest) -> ApprovalDecision:
            """Record the request and fail.

            Args:
                req: The request that will not be answered.

            Raises:
                RuntimeError: Always.
            """
            self.requests.append(req)
            raise boom

    approver = _BrokenApprover()

    with pytest.raises(RuntimeError, match="approver wire is down") as excinfo:
        await resolve(_context(workspace, approver), "escape_link", "read")

    assert excinfo.value is boom, "the very exception raised must be what escapes"
    assert len(approver.requests) == 1, (
        "the approver was asked, and it was the one that failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict", [None, "allow", 0], ids=["none", "string-allow", "falsey-int"]
)
async def test_resolve_refuses_a_verdict_that_is_not_the_allow_decision(
    workspace, spy, verdict
):
    """Only ``ApprovalDecision.ALLOW`` permits; anything else is refused.

    Fail-closed, and it is worth pinning: ``"allow"`` is the enum's *value*, so a
    comparison written as ``decision == "allow"`` would match it, and one written
    as ``if decision is DENY: refuse`` would wave through every other object.  An
    approver that returns something unexpected must block the operation, never
    allow it.
    """

    class _SloppyApprover:
        """An approver returning a verdict that is not a member of the enum."""

        def __init__(self, value: object) -> None:
            self.value = value
            self.requests: list[ApprovalRequest] = []

        async def request(self, req: ApprovalRequest) -> object:
            """Record the request and answer with the wrong kind of object.

            Args:
                req: The request being answered.

            Returns:
                Whatever this double was built with, ALLOW excluded.
            """
            self.requests.append(req)
            return self.value

    approver = _SloppyApprover(verdict)
    expected = (_root_of(workspace) / ".." / "outside.txt").resolve()

    path, error = await resolve(_context(workspace, approver), "escape_link", "read")

    assert path is None, (
        f"a verdict of {verdict!r} is not ALLOW, so the escape is refused"
    )
    assert error == REFUSAL_TEMPLATE.format(
        action="read", path=expected, root=_root_of(workspace)
    ), "the refusal is the same sentence an explicit DENY produces"
    assert len(approver.requests) == 1, "the escape was escalated exactly once"


@pytest.mark.asyncio
async def test_resolve_propagates_a_workspace_that_accepts_and_then_refuses(tmp_path):
    """Containment passing and then resolution failing propagates, and is not re-asked.

    The documented race: ``is_within`` said yes and ``resolve_within`` then said
    no, which only a filesystem change between the two calls can cause.  The escape
    is deliberately not re-submitted to the approver, because asking about a path
    whose answer depends on when the question was asked is how a human ends up
    approving a different path than the one the tool then opens.  The double
    implements only ``path``, ``is_within`` and ``resolve_within``, because that is
    all ``ToolContext`` and ``resolve`` use.
    """
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    (root / "main.py").write_text("x", encoding="utf-8")
    error = MitoCodeWorkspacePathError("the file moved mid-check")
    flaky = _FlakyWorkspace(root, error)
    spy = _RecordingApprover(ApprovalDecision.ALLOW)

    with pytest.raises(
        MitoCodeWorkspacePathError, match="the file moved mid-check"
    ) as excinfo:
        await resolve(_context(flaky, spy), "main.py", "read")

    assert excinfo.value is error, "the workspace's own exception must be what escapes"
    assert flaky.resolve_within_targets == [root / "main.py"], (
        "the resolved-path call receives the root-joined candidate"
    )
    assert spy.requests == [], "a race must not produce a second approval request"