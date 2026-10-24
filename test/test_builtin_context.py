"""Unit tests for the tool context, the error boundary, and output clamping."""

import asyncio
import contextlib
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.core.tool.builtin.context as context_module
from mitocode.core.tool.builtin.approval import AlwaysAllow
from mitocode.core.tool.builtin.background import BackgroundRegistry
from mitocode.core.tool.builtin.context import (
    ToolContext,
    clamp,
    current,
    current_agent_context,
    current_agent_id,
    set_current,
)
from mitocode.core.tool.builtin.errors import (
    MitoCodeBuiltinToolError,
    tool_boundary,
)
from mitocode.core.tool.tool_manager import ToolManager, declare_tool
from mitocode.core.workspace.workspace import Workspace

CONTEXT = ToolContext()


#: Value the ContextVar carries when no agent is currently running a turn.
UNKNOWN_AGENT_ID = -1

#: Dotted path :func:`clamp` inserts between the head and the tail.  Spelled out
#: here rather than imported, because the tests that use it assert on the
#: length the marker adds to the budget — a value copied from the module's own
#: f-string would pass on any implementation of it, including a wrong one.
TRUNCATION_MARKER = "\n\n[truncated: {dropped} characters omitted]\n\n"


@pytest.fixture(autouse=True)
def _isolated_module_state():
    """Reset the module-level mutables around every test in this file.

    ``context`` keeps two pieces of process-wide state: the registered current
    ``ToolContext`` and the ``current_agent_id`` ContextVar.  Both survive
    between test files in a single pytest process, so a test that registers a
    context would otherwise hand that context to an unrelated test in another
    file, and a test that sets an agent id would make "no agent" assertions
    fail depending on execution order.  Restoring ``None`` — rather than a
    fresh ``ToolContext`` — is what makes ``current()`` take its lazy-default
    path in the next test, which is one of the behaviours under test.

    Yields:
        Nothing; the fixture exists only for its setup and teardown.
    """
    context_module._CURRENT = None  # pylint: disable=protected-access
    token = current_agent_id.set(UNKNOWN_AGENT_ID)
    try:
        yield
    finally:
        current_agent_id.reset(token)
        context_module._CURRENT = None  # pylint: disable=protected-access


@contextlib.contextmanager
def _captured_logs():
    """Collect every loguru line emitted inside the block.

    loguru does not propagate to the stdlib ``logging`` tree, so pytest's
    ``caplog`` fixture sees nothing from it.  This mirrors the helper already
    used by ``test_tool_manager.py`` and ``test_mcp_manager.py``.

    Yields:
        The list every emitted line is appended to.
    """
    lines = []
    sink_id = loguru_logger.add(lines.append, level="DEBUG")
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)


# --- tool_boundary ---


@pytest.mark.asyncio
async def test_tool_boundary_renders_oserror_as_error_string():
    @tool_boundary
    async def boom():
        raise OSError("disk gone")

    result = await boom()
    assert result.startswith("Error: OSError: disk gone")


@pytest.mark.asyncio
async def test_tool_boundary_passes_through_normal_return():
    @tool_boundary
    async def fine():
        return "all good"

    assert await fine() == "all good"


@pytest.mark.asyncio
async def test_tool_boundary_passes_through_arguments_and_keyword_arguments():
    """A tool is registered from its signature, so arguments must survive."""

    @tool_boundary
    async def echo(first, second="?"):
        return f"{first}{second}"

    assert await echo("a") == "a?"
    assert await echo("a", second="b") == "ab"
    assert await echo(first="a", second="b") == "ab"


@pytest.mark.asyncio
async def test_tool_boundary_renders_a_builtin_tool_error_like_any_other():
    """The boundary does not special-case its own error type."""

    @tool_boundary
    async def boom():
        raise MitoCodeBuiltinToolError("no orchestrator attached")

    result = await boom()
    assert result.startswith("Error: MitoCodeBuiltinToolError: no orchestrator")


@pytest.mark.asyncio
async def test_tool_boundary_preserves_the_cause_in_the_log():
    """An escaped exception must still be logged, not silently swallowed."""

    @tool_boundary
    async def boom():
        raise ValueError("nope")

    with _captured_logs() as lines:
        await boom()
    logged = "\n".join(lines)
    assert "nope" in logged
    assert "boom" in logged


@pytest.mark.asyncio
async def test_tool_boundary_does_not_swallow_cancellation():
    """Cancelling a tool must cancel it, not turn the cancel into a string.

    ``AgentRuntime`` cancels a tool task to stop an agent turn.  A boundary
    that rendered ``CancelledError`` as an ``"Error: ..."`` string would let
    the cancelled turn run to completion and report success.
    """

    @tool_boundary
    async def sleeper():
        await asyncio.sleep(10)
        return "never reached"

    task = asyncio.create_task(sleeper())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


def test_cancelled_error_is_not_an_exception_subclass():
    """Why a bare ``except Exception`` is safe inside the boundary.

    Verified on this repository's interpreter (CPython 3.10.20); the same
    hierarchy holds on 3.11 through 3.14.  ``asyncio.CancelledError`` moved
    from ``Exception`` to ``BaseException`` in 3.8, so ``except Exception``
    never sees it.  If this assertion ever fails, the boundary in
    ``errors.tool_boundary`` would start eating cancellations and this
    assertion is the tripwire for it.
    """
    assert not issubclass(asyncio.CancelledError, Exception)
    assert issubclass(asyncio.CancelledError, BaseException)


def test_tool_boundary_keeps_the_wrapped_signature_and_name():
    """``ToolManager.register`` derives the JSON schema from the annotation.

    ``functools.wraps`` sets ``__wrapped__``, which is what
    :func:`inspect.signature` follows, so the registered schema is the tool's
    real signature rather than ``(*args, **kwargs)``.
    """

    @tool_boundary
    async def sample(file_path: str, limit: int = 10) -> str:
        """Read a slice of a file."""
        return file_path

    assert sample.__name__ == "sample"
    assert sample.__doc__ == "Read a slice of a file."
    assert sample.__wrapped__ is not None
    params = inspect.signature(sample).parameters
    assert list(params) == ["file_path", "limit"]
    assert params["file_path"].annotation is str
    assert params["limit"].default == 10


def test_builtin_tool_error_is_a_mitocode_error():
    from mitocode.core.exception.mitocode_exception import MitoCodeBaseError

    assert issubclass(MitoCodeBuiltinToolError, MitoCodeBaseError)


# --- clamp ---


def test_clamp_keeps_short_text_untouched():
    assert clamp("hello", 100) == "hello"


def test_clamp_truncates_head_and_tail_and_names_the_loss():
    text = "A" * 100 + "B" * 100
    out = clamp(text, 120)
    assert out.startswith("A")
    assert out.endswith("B")
    assert "truncated" in out
    assert len(out) < len(text)


def test_clamp_at_exact_limit_is_untouched():
    assert clamp("x" * 10, 10) == "x" * 10


def test_clamp_keeps_limit_over_two_chars_at_the_head_and_the_remainder_at_the_tail():
    """An odd limit rounds the remainder up into the tail, not the head."""
    text = "".join(chr(ord("a") + (i % 26)) for i in range(1000))
    out = clamp(text, 101)
    assert text[:50] in out
    assert text[-51:] in out
    assert text[50:949] not in out


def test_clamp_marker_names_the_number_of_characters_dropped():
    text = "y" * 500
    out = clamp(text, 100)
    assert "[truncated: 400 characters omitted]" in out


def test_clamp_of_empty_text_is_empty():
    assert clamp("", 10) == ""


def test_clamp_of_a_non_positive_limit_is_only_the_marker():
    """No head and no tail, but the model is still told something was dropped."""
    out = clamp("abcdef", 0)
    assert "truncated" in out
    assert "abcdef" not in out


# --- ToolContext ---


def test_root_falls_back_to_cwd_when_no_workspace():
    assert ToolContext().root() == __import__("pathlib").Path.cwd().resolve()


def test_root_uses_workspace_path(tmp_path):
    ws = Workspace(tmp_path, "t", 2)
    assert ToolContext(workspace=ws).root() == tmp_path.resolve()


def test_defaults_are_the_documented_values():
    ctx = ToolContext()
    assert ctx.workspace is None
    assert ctx.tool_manager is None
    assert isinstance(ctx.approver, AlwaysAllow)
    assert ctx.orchestrator is None
    assert ctx.max_output_chars == 30_000
    assert ctx.max_sub_agent_depth == 4


def test_each_context_gets_its_own_approver():
    """A shared default would let one context's approver state leak sideways."""
    first = ToolContext()
    second = ToolContext()
    assert first.approver is not second.approver


def test_background_defaults_to_a_fresh_registry_and_keeps_an_explicit_one():
    """The field is a real registry, built per context, and never clobbered.

    ``background.py`` exists now, so ``context.py`` imports
    :class:`BackgroundRegistry` at module level and builds one through
    ``default_factory``.  Two properties have to hold, and neither is free:
    a *shared* registry would let one agent collect or cancel another agent's
    detached calls, since handles are process-wide strings; and an explicit
    value must survive construction verbatim, so that re-adding a
    ``__post_init__`` which overwrites the field would fail here rather than
    silently retarget a caller's registry.

    The registry's own behaviour is covered by Task 6's dedicated test module;
    what is asserted here is only that the field is wired to it.
    """

    assert isinstance(ToolContext().background, BackgroundRegistry)

    first = ToolContext()
    second = ToolContext()
    assert (
        first.background is not second.background
    ), "the registry is built per context, so one cannot hand state to another"

    sentinel = object()
    assert ToolContext(background=sentinel).background is sentinel, (
        "an explicitly supplied registry must be stored verbatim, not replaced "
        "by a fresh one"
    )


def test_attach_orchestrator_installs_the_orchestrator():
    ctx = ToolContext()
    sentinel = object()
    ctx.attach_orchestrator(sentinel)
    assert ctx.orchestrator is sentinel


# --- the module-level current context ---


def test_current_agent_id_defaults_to_unknown():
    assert current_agent_context()[0] is None
    assert current_agent_id.get() == UNKNOWN_AGENT_ID


def test_current_agent_context_reports_the_running_agent_id():
    current_agent_id.set(7)
    agent_id, ctx = current_agent_context()
    assert agent_id == 7
    assert isinstance(ctx, ToolContext)


def test_set_current_replaces_the_module_default():
    set_current(CONTEXT)
    assert current() is CONTEXT


def test_current_creates_a_default_when_nothing_is_registered():
    """A tool called with no registration must still work."""
    first = current()
    assert isinstance(first, ToolContext)
    assert current() is first


def test_set_current_replaces_rather_than_mutates_the_previous_context():
    """Two contexts must not end up sharing one approver or one root."""
    first = ToolContext()
    second = ToolContext()
    set_current(first)
    assert current() is first
    set_current(second)
    assert current() is second


def test_root_of_the_lazily_created_default_is_the_cwd():
    assert current().root() == Path.cwd().resolve()


# --- tool_boundary: the safety property ---
#
# Everything above checks the half of the boundary's contract that converts a
# failure into prose.  This section checks the other half: the classes asyncio
# and the interpreter use to *stop* a running turn must never be converted,
# because a turn that was cancelled has to stop rather than report an outcome.


@pytest.mark.asyncio
async def test_tool_boundary_reraises_cancelled_error_raised_inside_the_tool():
    """Cancellation escapes even when the tool raises it itself.

    The test above cancels the task from the outside, which proves the boundary
    does not intercept a cancellation in flight.  This one proves the other
    direction: a tool whose body raises ``CancelledError`` — a timeout shim
    that re-raises it, a library mapping its own cancellation onto it — must
    not have it rewritten into ``"Error: CancelledError: "``, which would let
    ``AgentRuntime`` resume a turn that was already being stopped.
    """

    @tool_boundary
    async def cancelled() -> str:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_exception",
    [KeyboardInterrupt, SystemExit],
    ids=["KeyboardInterrupt", "SystemExit"],
)
async def test_tool_boundary_reraises_base_exceptions_that_halt_the_process(
    base_exception,
):
    """``except Exception`` must not reach the two that stop the interpreter.

    Both are ``BaseException`` subclasses for the same reason
    ``CancelledError`` is, and for the same reason: catching them would swallow
    the operator's ``Ctrl-C`` and a ``sys.exit()`` raised inside a tool body.
    """

    @tool_boundary
    async def halted() -> str:
        raise base_exception("stop")

    with pytest.raises(base_exception, match="stop"):
        await halted()


@pytest.mark.asyncio
async def test_tool_boundary_renders_an_exception_with_an_empty_message():
    """A bare ``raise KeyError`` still yields the type prefix.

    ``str(KeyError())`` is empty, so the rendered line ends in a trailing
    space.  That is the contract: the *type* is what the model branches on, and
    dropping the ``": "`` when the message is empty would make an empty message
    indistinguishable from a malformed line.
    """

    @tool_boundary
    async def bare() -> str:
        raise KeyError

    assert await bare() == "Error: KeyError: "


class _ExplosiveStr(Exception):
    """An exception whose ``__str__`` raises, as some proxy classes do."""

    def __str__(self) -> str:
        """Fail on purpose: the boundary renders with the exception itself.

        Raises:
            RuntimeError: Always, which is the whole point of this class.
        """
        raise RuntimeError("__str__ exploded")


@pytest.mark.asyncio
async def test_tool_boundary_renders_even_when_the_exception_str_raises():
    """A hostile ``__str__`` must not defeat the one guarantee this file exists for.

    The render is ``f"Error: {type(exc).__name__}: {exc}"``, which calls
    ``str(exc)`` inside the ``except`` clause, so the failure happens while the
    decorator is already unwinding.  Note that the log line is lost too:
    loguru formats the same exception for ``logger.exception`` and swallows its
    own formatting error, so the operator gets neither the transcript entry nor
    the traceback.
    """

    @tool_boundary
    async def hostile() -> str:
        raise _ExplosiveStr("original message")

    result = await hostile()

    assert result.startswith(
        "Error: _ExplosiveStr:"
    ), "an exception with a broken __str__ must still reach the model as prose"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_value",
    [None, 0, False, []],
    ids=["none", "zero", "false", "empty-list"],
)
async def test_tool_boundary_passes_through_a_falsy_result_untouched(result_value):
    """The decorator must not convert, re-wrap or "improve" a falsy result.

    ``None``, ``0``, ``False`` and ``[]`` are all falsy and all plausible tool
    returns, and the docstring is explicit that the return type is ``Any``
    because the decorator "only replaces an escaped exception".  Identity is
    asserted rather than equality, because a wrapper that rebuilt the value —
    ``str(result)``, a copy of the list — would pass an equality check on three
    of these four and only the fresh list would catch it.
    """

    @tool_boundary
    async def produces() -> object:
        return result_value

    result = await produces()

    assert result is result_value, "the wrapper must hand back the very same object"
    assert type(result) is type(result_value), "the type must not be coerced to str"


@pytest.mark.asyncio
async def test_tool_boundary_logs_nothing_when_the_tool_succeeds():
    """The log call lives inside the ``except``, so a healthy tool is silent.

    A boundary that logged on every call would bury the one traceback that
    matters under a line per tool invocation, which defeats the reason it logs
    at all.
    """

    @tool_boundary
    async def fine() -> str:
        return "all good"

    with _captured_logs() as lines:
        assert await fine() == "all good"

    assert not [
        line for line in lines if "fine" in line
    ], f"a successful call must not log the tool name, got {lines!r}"


@pytest.mark.asyncio
async def test_tool_boundary_passes_a_multi_line_message_through_verbatim():
    """Subprocess errors carry newlines, and the model has to see them.

    Nothing escapes the message: no escaping, no indentation, no joining.  An
    error naming a line of stderr reaches the model exactly as it was raised,
    which is the only reason a model can point at the line it names.
    """

    @tool_boundary
    async def boom() -> str:
        raise RuntimeError("stdout: ok\nstderr: broken")

    assert await boom() == "Error: RuntimeError: stdout: ok\nstderr: broken"


def test_tool_boundary_preserves_the_metadata_the_registry_reads():
    """Everything ``ToolManager.register`` looks at must survive the wrapper.

    ``register`` reads ``func.__name__`` for the tool name and builds the JSON
    Schema from ``func``'s annotations, following ``__wrapped__`` for the
    signature.  A boundary that lost any of these would register a tool called
    ``_wrapper`` that advertises no parameters, and the only symptom would be a
    model calling a tool it has no way to pass arguments to.
    """

    @tool_boundary
    async def sample(file_path: str) -> str:
        """Read a file."""
        return file_path

    assert sample.__name__ == "sample", "the tool name is read off __name__"
    assert sample.__module__ == __name__, "the defining module must survive"
    assert sample.__qualname__.endswith("sample"), "the qualified name must survive"
    assert sample.__doc__ == "Read a file.", "the human doc is the docstring"
    assert sample.__annotations__ == {
        "file_path": str,
        "return": str,
    }, "the schema is derived from these annotations, not from the wrapper's"
    assert inspect.iscoroutinefunction(
        sample
    ), "the registry awaits the registered callable"


@pytest.mark.asyncio
async def test_a_boundary_wrapped_tool_registers_under_its_declared_name_and_schema():
    """End-to-end: the boundary is transparent to the real registry.

    Deliberately uses the real :class:`ToolManager` rather than a fake, because
    the claim under test is that ``functools.wraps`` merged the wrapped
    function's ``__dict__`` — which is where ``declare_tool`` publishes its
    metadata — into the wrapper.  If that ever stopped being true the tool would
    register under the name ``_wrapper`` and the failure would be invisible
    until a model tried to call it.
    """

    manager = ToolManager()

    @tool_boundary
    @declare_tool(
        name="Read",
        human_doc="Read a file",
        agent_doc="Read a file for the agent.",
    )
    async def read(file_path: str, limit: int = 10) -> str:
        """Read a slice of a file."""
        return file_path

    spec = manager.register(read)

    assert spec.name == "Read", "declare_tool metadata must survive the boundary"
    assert spec.human_doc == "Read a file", "human_doc comes from the same metadata"
    assert spec.agent_doc == "Read a file for the agent."
    assert spec.func is read, "dispatch must go through the boundary, not the bare body"
    schema = spec.parameters_schema
    assert list(schema["properties"]) == [
        "file_path",
        "limit",
    ], "the schema must be derived from the real signature"
    assert schema["required"] == ["file_path"], "a defaulted parameter is optional"
    assert schema["properties"]["limit"]["default"] == 10
    assert await read("a.txt") == "a.txt", "the registered callable must still run"


@pytest.mark.asyncio
async def test_tool_boundary_reports_a_non_coroutine_body_as_an_error_string():
    """Decorating a plain function degrades to prose instead of crashing.

    ``await func(...)`` on a non-awaitable raises ``TypeError`` inside the
    ``try``, so a mis-decorated tool reaches the model as a line it can react
    to rather than taking the whole turn down with a traceback.  The exact
    wording is the interpreter's, so only the type and the mention of ``await``
    are asserted.
    """

    @tool_boundary
    def not_a_coroutine() -> str:
        return "computed synchronously"

    result = await not_a_coroutine()

    assert result.startswith(
        "Error: TypeError: "
    ), f"a non-awaitable body must render as an error line, got {result!r}"
    assert "await" in result, f"the TypeError must name the await, got {result!r}"


# --- clamp: the edges of the budget ---


@pytest.mark.parametrize(
    "limit", [0, -1, -30_000], ids=["zero", "negative-one", "far-negative"]
)
def test_clamp_of_empty_text_is_empty_at_every_non_positive_budget(limit):
    """The no-truncation branch must be reached before the marker is built.

    The budget is clamped to zero first, which puts an empty string exactly on
    the boundary of ``len(text) <= limit``.  Answering with a truncation marker
    would tell the model characters were dropped from a string that had none.
    """

    assert clamp("", limit) == ""


def test_clamp_of_a_negative_budget_behaves_exactly_like_a_zero_budget():
    """``max(limit, 0)`` makes a negative budget zero, not a reversed slice.

    ``text[:-5]`` would be the *opposite* of an empty tail, so a caller that
    passed a negative budget by accident must not get most of the text back.
    """

    text = "abcdefghij"

    assert clamp(text, -5) == clamp(
        text, 0
    ), "a negative budget must be clamped to zero, exactly"
    assert clamp(text, -5) == TRUNCATION_MARKER.format(dropped=10)


def test_clamp_one_character_over_the_budget_drops_exactly_one():
    """The tightest truncation: ten characters in, a budget of nine.

    Hand-computed: ``head = 9 // 2 = 4`` and ``tail = 9 - 4 = 5``, so the result
    is the first four characters, a marker naming the single dropped
    character, and the last five.
    """

    result = clamp("abcdefghij", 9)

    assert result == "abcd" + TRUNCATION_MARKER.format(dropped=1) + "fghij"
    assert len(result) == 9 + len(
        TRUNCATION_MARKER.format(dropped=1)
    ), "the marker sits on top of the budget rather than inside it"


@pytest.mark.parametrize(
    "character",
    ["\u00e9", "\U0001f600"],
    ids=["two-byte-accent", "four-byte-emoji"],
)
def test_clamp_budgets_code_points_and_not_bytes(character):
    """Multi-byte characters are one unit of budget each.

    The two representatives are two and four bytes wide in UTF-8 but one code
    point each.  A byte budget would hand back fewer characters than the limit
    allows and would also cut a character in half; the contract is characters as
    Python counts them, so a model asking for 30 000 characters of CJK or emoji
    text gets 30 000 of them.
    """

    text = character * 10

    result = clamp(text, 4)

    assert (
        result == character * 2 + TRUNCATION_MARKER.format(dropped=6) + character * 2
    ), "head and tail must each keep limit // 2 characters"
    assert len(result.encode("utf-8")) > len(
        result
    ), "the result is longer in bytes than in code points, so this cannot be a byte budget"


def test_clamp_can_cut_between_a_base_character_and_its_combining_mark():
    """Slicing is by code point, so a cut can orphan a combining mark.

    ``"e"`` plus a combining acute accent is one glyph and two code points.  A
    budget of five splits the twenty-code-point string two from the end, which
    lands inside a sequence and leaves the tail beginning with a bare accent.
    This is documented behaviour rather than a defect — the contract is
    characters as Python counts them and no normalisation is attempted — but a
    caller that cares about grapheme clusters has to know it is not getting
    them.
    """

    # "e" followed by COMBINING ACUTE ACCENT: one glyph, two code points.
    base = "e\u0301"
    text = base * 10
    assert len(text) == 20, "twenty code points, ten visible characters"

    result = clamp(text, 5)

    tail = "\u0301" + base  # the cut lands between a base and its accent
    assert (
        result == base + TRUNCATION_MARKER.format(dropped=15) + tail
    ), "the tail begins with an orphaned combining mark, which is the caveat"


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101])
def test_clamp_splits_the_budget_with_the_odd_character_going_to_the_tail(limit):
    """For every budget the head is ``limit // 2`` and the tail takes the rest.

    Parametrised rather than one test per limit: this is one rule observed ten
    times, and writing the rule once is what makes it a rule.  The remainder
    always goes to the tail because ``head = limit // 2`` rounds down, so an odd
    budget cannot make the head depend on the text that went through it.
    """

    text = "".join(chr(ord("a") + (index % 26)) for index in range(500))
    head, tail = limit // 2, limit - limit // 2

    result = clamp(text, limit)

    assert head + tail == limit, "the split must consume the whole budget"
    assert result.startswith(text[:head]), f"limit {limit}: head must be {head} chars"
    assert result.endswith(
        text[len(text) - tail :]
    ), f"limit {limit}: tail must be {tail} chars"
    assert (
        TRUNCATION_MARKER.format(dropped=500 - limit) in result
    ), f"limit {limit}: the marker must survive the split"


def test_the_default_output_budget_is_the_one_clamp_is_given():
    """``max_output_chars`` is meant to be passed straight to :func:`clamp`.

    Linked to the literal default already asserted above rather than repeating
    it, so that retuning the default in one place cannot leave the other stale.
    """

    budget = ToolContext().max_output_chars
    on_budget = "x" * budget

    assert (
        clamp(on_budget, budget) == on_budget
    ), "a result exactly on budget is not clamped"
    over_budget = clamp(on_budget + "x", budget)
    assert "truncated" in over_budget, "one character over the budget is clamped"
    assert len(over_budget) > budget, "the marker is added on top of the budget"


# --- ToolContext.root(): paths that are awkward on purpose ---


def test_root_of_a_workspace_whose_directory_never_existed_is_still_returned(tmp_path):
    """A workspace may point at a directory that is not on disk.

    ``Path.resolve`` is non-strict by default, so the root is the absolute form
    of a path that need not exist.  The alternative — raising — would make a
    planned or scratch workspace unusable, which is what the field's own
    docstring promises by allowing a context to be built without a workspace on
    disk.
    """

    planned = tmp_path / "not" / "created"
    workspace = Workspace(planned, "planned", 2)
    assert not planned.exists(), "the fixture must really be missing from disk"

    root = ToolContext(workspace=workspace).root()

    assert root == planned.resolve(), "a missing directory still has a resolvable root"
    assert root.is_absolute(), "root() must never return a relative path"


def test_root_resolves_a_relative_workspace_path_against_the_cwd(tmp_path, monkeypatch):
    """``Workspace`` keeps a relative path relative; ``root`` must anchor it.

    ``Workspace._convert_to_path`` stores whatever it is handed, so a caller
    passing ``"src/pkg"`` ends up with a relative ``workspace.path``.  Every
    containment comparison downstream is between two resolved paths, so an
    unanchored root would be measured against whatever the process cwd happened
    to be and every relative path would look like an escape.
    """

    (tmp_path / "src" / "pkg").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    workspace = Workspace(Path("src") / "pkg", "relative", 2)
    assert not workspace.path.is_absolute(), "the fixture must really be relative"

    root = ToolContext(workspace=workspace).root()

    assert (
        root == (tmp_path / "src" / "pkg").resolve()
    ), "a relative workspace path must be anchored to the cwd"
    assert root.is_absolute(), "root() must never return a relative path"


def test_root_of_a_symlinked_cwd_is_the_physical_directory(tmp_path, monkeypatch):
    """The macOS ``/var`` versus ``/private/var`` case the docstring names.

    The working directory is entered through a symlink here, which is how a
    ``cd`` into a linked checkout looks to every path comparison.  ``root()``
    must hand back the physical directory, and agree with what ``Path.cwd()``
    reports, or a naive ``==`` against a non-resolved workspace path would call
    an in-root file out-of-root.
    """

    physical = tmp_path / "physical"
    physical.mkdir()
    link = tmp_path / "link"
    link.symlink_to(physical, target_is_directory=True)
    monkeypatch.chdir(link)

    root = ToolContext().root()

    assert root == physical.resolve(), "the physical directory, not the symlink to it"
    assert root == Path.cwd().resolve(), "root() must agree with the resolved cwd"
    assert (
        not root.is_symlink()
    ), "root() must not hand back a path containing a symlink"


def test_root_accepts_any_object_exposing_a_path(tmp_path):
    """``ToolContext`` never imports ``Workspace`` at runtime, by design.

    The type appears only under ``TYPE_CHECKING``, so ``root`` has to work
    against anything with a ``.path``.  Wrapping it in ``Path`` rather than
    assuming it already is one is what lets a plain string path through, and
    that is the shape a test double or a future workspace implementation is
    likely to have.
    """

    stub = SimpleNamespace(path=str(tmp_path))

    assert ToolContext(workspace=stub).root() == tmp_path.resolve()


# --- ToolContext fields ---


def test_explicit_budget_overrides_are_stored_verbatim_including_zero():
    """``0`` is a real setting here, not an absent one.

    Neither field is validated, so an implementation that defaulted a falsy
    value with ``or`` would be silently wrong: a caller asking for no sub-agents
    at all would quietly get four.  It is the same falsy-zero trap the agent id
    carries, on two values every later task reads.
    """

    context = ToolContext(max_output_chars=0, max_sub_agent_depth=0)

    assert context.max_output_chars == 0, "a zero budget must not fall back to 30_000"
    assert context.max_sub_agent_depth == 0, "depth 0 must not fall back to 4"


# --- current_agent_context(): the id and the context together ---


def test_current_agent_context_reports_agent_zero_rather_than_treating_it_as_unknown():
    """``0`` is falsy: a truthiness check here silently loses the first agent.

    The sentinel is ``-1`` and has to be compared, because every other int is a
    real agent id — and every other id in this file is non-zero, which is exactly
    why a truthiness bug here survives: agent 0 would report "no agent", and a
    sub-agent it launched would come back unstamped.
    """

    current_agent_id.set(0)

    agent_id, _ = current_agent_context()

    assert agent_id == 0, "agent 0 is a real agent, not the unknown sentinel"
    assert agent_id is not None, "None is reserved for the -1 sentinel"


@pytest.mark.parametrize(
    "agent_id", [-2, 2**31], ids=["another-negative", "large-positive"]
)
def test_current_agent_context_reports_every_other_id_verbatim(agent_id):
    """Only ``-1`` maps to ``None``; every other int is passed through unchanged.

    ``-2`` is the near-miss case: it is negative and therefore a plausible
    sentinel someone might reach for, and a truthiness check happens to answer
    correctly for it.  What makes that luck rather than correctness is the ``0``
    case above, which the same check gets wrong.
    """

    current_agent_id.set(agent_id)

    assert (
        current_agent_context()[0] == agent_id
    ), f"agent {agent_id} must be reported as itself"


@pytest.mark.asyncio
async def test_the_agent_id_is_isolated_between_concurrent_turns():
    """Two sub-agents running at once must not see each other's id.

    ``current_agent_id`` is a ``ContextVar`` for exactly this: a module-level
    global would let a background sub-agent relabel the main turn's tools for
    the rest of its output.  ``asyncio`` copies the current context into every
    task, so the two ``set`` calls below cannot escape into each other or back
    into the turn that spawned them.
    """

    async def sub_agent(agent_id: int) -> int:
        current_agent_id.set(agent_id)
        await asyncio.sleep(0)  # let the sibling task run its own set first
        return current_agent_context()[0]

    first, second = await asyncio.gather(sub_agent(3), sub_agent(4))

    assert (first, second) == (3, 4), "each task must read back only its own id"
    assert (
        current_agent_context()[0] is None
    ), "the parent turn still has no agent of its own"


@pytest.mark.asyncio
async def test_the_registered_context_is_process_wide_while_the_agent_id_is_not():
    """The two pieces of module state are scoped differently, on purpose.

    ``current_agent_id`` is a ``ContextVar`` so concurrent turns cannot relabel
    each other, while the registered context is a plain module global because
    the module docstring names the trade outright — "the context is
    process-wide rather than per-call".  A sub-agent therefore inherits its
    parent's root, which is only safe because the nesting depth is bounded
    elsewhere.  Pinning both halves in one test is what stops someone "fixing"
    one of them in isolation and silently changing the other's guarantee.
    """

    context = ToolContext()
    set_current(context)

    async def sub_agent() -> tuple:
        return current() is context, current_agent_context()[0]

    inherited_context, inherited_agent_id = await asyncio.create_task(sub_agent())

    assert (
        inherited_context is True
    ), "the registered context is process-wide by design, so a sub-agent shares the root"
    assert (
        inherited_agent_id is None
    ), "the agent id is task-local, so the sub-agent sees no agent of its own"


def test_current_agent_context_arms_the_registered_context_together_with_the_id():
    """The two halves of the return value travel as one.

    Reading the id and the context in separate calls is how a caller ends up
    stamping a sub-agent with one turn's id and launching it against another
    turn's context, so the pair is asserted as a pair.
    """

    context = ToolContext()
    set_current(context)

    agent_id, returned_context = current_agent_context()

    assert agent_id is None, "no agent is running, so only the sentinel is reported"
    assert (
        returned_context is context
    ), "the context must be the registered one, not a lazily created default"


# --- global state hygiene ---


def test_a_registered_context_is_visible_to_unrelated_callers():
    """The hazard the fixture exists for: the registration is process-wide.

    Asserts the leak *happens*, before anything resets it.  If the module ever
    grew per-task or per-agent storage this would fail, and that is the intent:
    the module docstring trades isolation for a tool signature the model cannot
    fill in, and that trade should not be changed silently — it would make the
    fixture and every test that depends on it meaningless rather than better.
    """

    context = ToolContext()

    set_current(context)

    assert current() is context, "the registration must be visible module-wide"
    assert (
        current_agent_context()[1] is context
    ), "the agent tools read the same global, so they inherit it too"


def test_the_module_state_is_reset_between_tests():
    """Canary for the autouse fixture, kept last so it watches every test above.

    It fires when the reset is no longer installed at all.  It deliberately does
    not fire when only the setup or only the teardown half is removed: the two
    are individually redundant — either one alone keeps every test clean — so a
    failure here means the guarantee is genuinely gone rather than merely
    implemented in one place instead of two.  Verified by deleting the whole
    reset block from the fixture in a scratch copy, which fails this test and
    two others with it.
    """

    assert (
        context_module._CURRENT is None
    ), "a previous test leaked its context"  # pylint: disable=protected-access
    assert (
        current_agent_id.get() == UNKNOWN_AGENT_ID
    ), "a previous test leaked its agent id"
