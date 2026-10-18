"""Unit tests for the nested tool-call mini-DSL parser.

The parser handles text a model produced and we consume whole, so the text is
untrusted.  The suite is therefore split the way the risk is:

* The attack suite tries to get **code execution** or an unexpected AST shape
  through ``ast.parse`` + ``ast.literal_eval``.  Every attempt must be refused
  at the *literal* step, with nothing written to disk, nothing added to
  ``sys.modules``, and nothing echoed raw into an error message.
* The contract suite pins the points of ``parse_call`` / ``bind_arguments`` that
  the docstrings leave open: which exception step refuses an input, positional
  binding order, schema ``default`` versus ``required``, and which value wins
  when the same parameter is supplied twice.

Several behaviours are traps rather than defects-with-a-right-answer, so they
are pinned as-is and flagged by name: grep for ``silent_``, ``DEFECT``, ``not
checked``, ``without_type_check`` or ``FINDING`` to list every one.  They are
described individually in the docstring of the test that pins them.
"""

import copy
import math
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.tool.builtin.call_dsl import ParsedCall, bind_arguments, parse_call
from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError
from mitocode.core.tool.tool_manager import ToolManager, ToolSpec


def _two_param_tool(first: str, second: str) -> str:
    return first + second


# The binding tests register the fixture as "Two" because that is the name the
# DSL call uses: `register` otherwise falls back to the function's own name,
# `_two_param_tool`, and the model only ever sees the registered name.


def _mixed_tool(first: str, second: str = "DEFAULT", third: int = 7) -> tuple:
    """One required parameter followed by two defaulted ones."""
    return first, second, third


def _flags_tool(flag: bool, count: int = 1, note: str = "x") -> tuple:
    """A required boolean plus two defaulted parameters."""
    return flag, count, note


def _background_tool(call: str, timeout: int = 300_000) -> str:
    """Stand-in for the Background tool named in the module docstring."""
    return call


def _spec_named(name: str, schema: Dict[str, Any]) -> ToolSpec:
    """Build a ``ToolSpec`` directly, for schema shapes ``register`` cannot emit.

    Args:
        name: The registered tool name the error messages will use.
        schema: A ``parameters_schema`` that ``ToolManager.register`` would
            never derive from annotations -- MCP servers and explicit
            ``parameters_schema=`` callers can both supply these.

    Returns:
        A spec whose ``parameters_schema`` is *schema* verbatim.
    """
    return ToolSpec(
        name=name,
        func=_mixed_tool,
        human_doc="",
        agent_doc="",
        parameters_schema=schema,
    )


@pytest.fixture()
def mixed_spec() -> ToolSpec:
    """A registered tool with a required parameter ahead of two optional ones."""
    manager = ToolManager()
    spec = manager.register(_mixed_tool, name="Mixed")
    assert spec is not None, "the Mixed fixture tool must register"
    return spec


@pytest.fixture()
def canary_file(tmp_path: Path) -> Path:
    """A file whose contents stand in for "did anything actually execute?"."""
    path = tmp_path / "canary.txt"
    path.write_text("original", encoding="utf-8")
    return path


def test_single_quoted_positional_argument():
    parsed = parse_call('Bash("ls -la")')
    assert parsed.tool_name == "Bash"
    assert parsed.args == ["ls -la"]
    assert parsed.kwargs == {}


def test_keyword_arguments():
    parsed = parse_call('Grep(pattern="TODO", path="src")')
    assert parsed.tool_name == "Grep"
    assert parsed.kwargs == {"pattern": "TODO", "path": "src"}


def test_nested_quotes_survive():
    parsed = parse_call('Bash("echo \'hello world\'")')
    assert parsed.args == ["echo 'hello world'"]


def test_escaped_quotes_survive():
    parsed = parse_call(r'Bash("echo \"hi\"")')
    assert parsed.args == ['echo "hi"']


def test_numeric_argument_is_literal_evaluated():
    parsed = parse_call("Background(call='Bash(\"ls\")', timeout=300000)")
    assert parsed.kwargs["timeout"] == 300000
    assert isinstance(parsed.kwargs["timeout"], int)


def test_bare_name_is_rejected():
    with pytest.raises(MitoCodeBuiltinToolError, match="not a tool call"):
        parse_call("rm -rf /")


def test_non_literal_argument_is_rejected():
    with pytest.raises(MitoCodeBuiltinToolError):
        parse_call('Bash("ls" + os.sep)')


def test_attribute_call_is_rejected():
    with pytest.raises(MitoCodeBuiltinToolError, match="not a tool call"):
        parse_call("os.system('ls')")


def test_syntax_error_is_reported_as_a_builtin_error():
    with pytest.raises(MitoCodeBuiltinToolError):
        parse_call('Bash("unclosed')


def test_positional_arguments_bind_in_schema_order():
    manager = ToolManager()
    spec = manager.register(_two_param_tool, name="Two")
    assert spec is not None
    bound = bind_arguments(spec, parse_call('Two("alpha", "beta")'))
    assert bound == {"first": "alpha", "second": "beta"}


def test_too_many_positional_arguments_is_rejected():
    manager = ToolManager()
    spec = manager.register(_two_param_tool, name="Two")
    with pytest.raises(MitoCodeBuiltinToolError, match="Two"):
        bind_arguments(spec, parse_call('Two("a", "b", "c")'))


def test_missing_required_argument_is_rejected():
    manager = ToolManager()
    spec = manager.register(_two_param_tool, name="Two")
    with pytest.raises(MitoCodeBuiltinToolError, match="second"):
        bind_arguments(spec, parse_call('Two("a")'))


def test_unknown_keyword_is_rejected():
    manager = ToolManager()
    spec = manager.register(_two_param_tool, name="Two")
    with pytest.raises(MitoCodeBuiltinToolError, match="nope"):
        bind_arguments(spec, parse_call('Two(first="a", second="b", nope="c")'))


# ---------------------------------------------------------------------------
# Code execution: the input is model-generated and therefore untrusted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dsl",
    [
        'Bash(eval("1+1"))',
        'Bash(exec("import os"))',
        'Bash(__import__("os"))',
        'Bash(__import__("os").system("ls"))',
        'Bash(open("/etc/passwd"))',
        'Bash(open("/etc/passwd").read())',
        'Bash(os.system("rm -rf /"))',
        "Bash((1).__class__.__mro__[1].__subclasses__())",
        'Bash(globals()["__builtins__"])',
        "Bash(self.__init__.__globals__)",
    ],
    ids=[
        "eval",
        "exec",
        "import-call",
        "import-attribute",
        "open",
        "open-read",
        "os-system",
        "dunder-subclasses",
        "globals-subscript",
        "self-globals",
    ],
)
def test_code_execution_argument_is_rejected_at_the_literal_step(dsl):
    """A call used as an argument must be refused by literal_eval, not run.

    The refusal has to happen at the ``ast.literal_eval`` step *specifically*.
    ``_PARSE_FAILURES`` and ``_LITERAL_FAILURES`` hold the same five exception
    types, so the chained cause cannot tell the two steps apart -- the message
    can, and so can the control at the end.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "is not a literal value" in message, (
        f"the argument must be refused by literal_eval, got: {message}"
    )
    assert "not a tool call" not in message, (
        f"the AST shape is a legal call, so the shape gate must not be what refused it: {message}"
    )
    assert isinstance(excinfo.value.__cause__, ValueError), (
        f"literal_eval signals a non-literal node with ValueError, got "
        f"{type(excinfo.value.__cause__).__name__}"
    )
    # Control: the same wrapper parses cleanly, so the failure above is the
    # argument and not the DSL syntax around it.
    assert parse_call("Bash(1)").args == [1], "control call must parse"


@pytest.mark.parametrize(
    "template",
    [
        'Bash(open({path}, "w").write("PWNED"))',
        'Bash(__import__("pathlib").Path({path}).write_text("PWNED"))',
    ],
    ids=["open-write", "pathlib-write"],
)
def test_writing_payload_argument_performs_no_filesystem_write(template, canary_file):
    """``open(...).write(...)`` as an argument must leave the disk untouched."""
    dsl = template.format(path=repr(str(canary_file)))

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    assert "is not a literal value" in str(excinfo.value), (
        f"the write must be refused as a non-literal, got: {excinfo.value}"
    )
    assert canary_file.read_text(encoding="utf-8") == "original", (
        "the payload's write() ran; the canary was overwritten"
    )
    assert sorted(path.name for path in canary_file.parent.iterdir()) == ["canary.txt"], (
        "the payload created a file"
    )


def test_exec_argument_does_not_import_a_module():
    """``exec("import wave")`` as an argument must not populate ``sys.modules``."""
    modules_before = set(sys.modules)

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call('Bash(exec("import wave"))')

    new_modules = set(sys.modules) - modules_before
    assert new_modules == set(), f"the payload's exec ran and imported {sorted(new_modules)}"
    assert "wave" not in sys.modules, "the payload's import statement actually executed"
    assert isinstance(excinfo.value.__cause__, ValueError), (
        "the refusal must come from literal_eval, not from a partially-run exec"
    )


def test_builtin_name_is_carried_through_as_an_inert_tool_name():
    """FINDING: a builtin callee parses as a "tool call" and is only inert by
    the caller's discipline.

    ``parse_call`` performs no name resolution -- the name is carried forward as
    a string -- so ``__import__("os")`` is accepted and yields
    ``tool_name == "__import__"``.  Nothing here is exploitable, but the
    dispatcher MUST resolve the name with an exact registry lookup; anything
    that reaches for ``getattr`` or ``eval`` on this string would turn a parser
    bug straight into remote code execution.
    """
    parsed = parse_call('__import__("os")')

    assert parsed.tool_name == "__import__", (
        f"the callee name is forwarded verbatim, got {parsed.tool_name!r}"
    )
    assert parsed.args == ["os"], f"its argument is still literal-evaluated, got {parsed.args!r}"


def test_bytes_input_is_parsed_because_ast_accepts_bytes():
    """FINDING: ``bytes`` slips past the ``text: str`` signature.

    ``ast.parse`` accepts a bytes source, so a caller that passes a raw response
    body gets a successfully parsed call rather than a type error.  Worth
    knowing; not a hole on its own.
    """
    parsed = parse_call(b'Bash("a")')

    assert parsed.tool_name == "Bash", f"got {parsed.tool_name!r}"
    assert parsed.args == ["a"], f"got {parsed.args!r}"


@pytest.mark.parametrize("bad", [None, 123, 4.5, ["x"]], ids=["none", "int", "float", "list"])
def test_non_string_input_is_wrapped_as_a_builtin_error(bad):
    """A non-``str`` ``text`` surfaces as ``MitoCodeBuiltinToolError``.

    This closed a defect in the error path itself: ``ast.parse`` does raise
    ``TypeError``, and ``TypeError`` *is* in ``_PARSE_FAILURES``, so the
    handler ran -- and then called ``_excerpt(text)``, whose ``len(text)``
    raised a *second* ``TypeError`` from inside the ``except`` block, which
    nothing caught.  ``_excerpt`` is now total, so a value with no ``len``
    renders as its type name instead of crashing the error path.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(bad)

    message = str(excinfo.value)
    assert type(excinfo.value) is MitoCodeBuiltinToolError, (
        f"nothing but the builtin error may escape, got {type(excinfo.value).__name__}"
    )
    assert "not a tool call" in message, f"got: {message}"
    assert f"<{type(bad).__name__}>" in message, (
        f"the message must name the offending value's type, got: {message}"
    )
    assert isinstance(excinfo.value.__cause__, TypeError), (
        f"the original TypeError must be chained, got {type(excinfo.value.__cause__).__name__}"
    )


# ---------------------------------------------------------------------------
# AST shapes that are not a plain call on a bare name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dsl",
    [
        "Bash",
        '"Bash()"',
        "42",
        "[1, 2]",
        '{"a": 1}',
        "lambda: 1",
        'f"{x}"',
        'Bash("a"), Bash("b")',
        'Bash("a")[0]',
        "x if y else z",
        "(1,)",
        "()",
        "Bash.__call__",
    ],
    ids=[
        "bare-name",
        "quoted-string",
        "number",
        "list",
        "dict",
        "lambda",
        "f-string",
        "tuple-of-calls",
        "subscript",
        "ifexp",
        "one-tuple",
        "empty-tuple",
        "dunder-attribute",
    ],
)
def test_non_call_shape_is_rejected_by_the_shape_gate(dsl):
    """Valid Python that is not a call must be refused by the ``ast.Call`` check.

    Every one of these parses cleanly, so a failure here proves the shape gate
    -- not the tokenizer -- is what refuses them.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "not a tool call" in message, f"shape rejection must say so, got: {message}"
    assert "expected a call of the form" in message, (
        f"the shape gate must be the one refusing this, got: {message}"
    )
    assert excinfo.value.__cause__ is None, (
        "the shape gate raises directly; a chained cause would mean the parser failed first"
    )


@pytest.mark.parametrize(
    "dsl",
    [
        "a.b.c(1)",
        'd["k"]()',
        "f()()",
        'self.bash("x")',
        "tools.Bash('x')",
        '"../../etc/passwd"("x")',
    ],
    ids=[
        "chained-attribute",
        "subscript-callee",
        "call-callee",
        "self-attribute",
        "module-attribute",
        "string-callee",
    ],
)
def test_non_name_callee_is_rejected(dsl):
    """Only a bare ``ast.Name`` callee is a tool call.

    Attribute, subscript and nested-call callees are the entire sandbox-escape
    surface; a regex-based parser would happily accept most of them.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "not a tool call" in message, f"got: {message}"
    assert "callee must be a bare tool name" in message, (
        f"the callee check must be the one refusing this, got: {message}"
    )


@pytest.mark.parametrize(
    "dsl",
    [
        "Bash(*args)",
        'Bash(*["a", "b"])',
        "Bash([i for i in y])",
        "Bash({k: v for k, v in y})",
        "Bash({x for x in y})",
        "Bash(x for x in y)",
        "Bash((x := 1))",
        'Bash(f"{x}")',
        "Bash(1 + 2)",
        'Bash("a" * 3)',
        'Bash(-"a")',
        "Bash(lambda: 1)",
        'Bash(float("nan"))',
        "Bash((1).__class__)",
        'Bash("a" + "b")',
        "Bash(x if y else z)",
    ],
    ids=[
        "starred-name",
        "starred-list",
        "list-comp",
        "dict-comp",
        "set-comp",
        "genexp",
        "walrus",
        "f-string",
        "add",
        "multiply",
        "unary-minus",
        "lambda",
        "float-call",
        "dunder-attribute",
        "concat",
        "ifexp-argument",
    ],
)
def test_non_literal_argument_is_rejected_at_the_literal_step(dsl):
    """Expressions are refused by ``literal_eval``, not by the parser.

    Every one of these is syntactically a legal call, so reaching the argument
    check at all is the point: a denylist of "dangerous" argument text would
    have let several of these through.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "is not a literal value" in message, f"got: {message}"
    assert "at position 0" in message, (
        f"the message must name which argument was refused, got: {message}"
    )
    assert isinstance(excinfo.value.__cause__, ValueError), (
        f"literal_eval raises ValueError, got {type(excinfo.value.__cause__).__name__}"
    )


@pytest.mark.parametrize(
    "dsl",
    ['Bash(**{"command": "ls"})', "Bash(**kwargs)", 'Bash("ls", **rest)'],
    ids=["dict-splat", "name-splat", "mixed-splat"],
)
def test_double_star_unpacking_is_rejected_by_its_own_branch(dsl):
    """``**`` is valid Python that names nothing, so it needs its own check.

    ``keyword.arg is None`` for every splat, so this is a distinct branch from
    the literal check -- and it raises directly rather than wrapping a cause.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "not a tool call" in message, f"got: {message}"
    assert "** unpacking is not supported" in message, (
        f"the dedicated splat branch must be the one refusing this, got: {message}"
    )
    assert excinfo.value.__cause__ is None, (
        "the splat branch raises directly; a chained cause means it fell "
        "through to _literal instead"
    )


def test_keyword_splat_with_a_literal_payload_is_not_flattened_into_kwargs():
    """A splat whose *value* is a literal must still be refused.

    ``dict(**literal_eval(value))`` would happily expand ``Bash(**{"a": 1})``
    into a real keyword; this asserts it never gets that far.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call('Bash(**{"a": 1})')

    assert "** unpacking is not supported" in str(excinfo.value), (
        f"a literal splat payload must not be expanded, got: {excinfo.value}"
    )


def test_empty_call_parses_with_no_arguments():
    """``Bash()`` is a legal call; the arity complaint belongs to the bind step.

    Separating the two stages is what lets a distinct, more useful error name
    the parameter that is missing.
    """
    parsed = parse_call("Bash()")

    assert parsed.tool_name == "Bash", f"got {parsed.tool_name!r}"
    assert parsed.args == [], f"got {parsed.args!r}"
    assert parsed.kwargs == {}, f"got {parsed.kwargs!r}"


def test_empty_call_binds_and_reports_the_missing_required_parameter(mixed_spec):
    """An empty call survives parsing and fails at binding, naming the tool."""
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(mixed_spec, parse_call("Mixed()"))

    message = str(excinfo.value)
    assert "Missing required parameter for tool 'Mixed'" in message, f"got: {message}"
    assert "first" in message, f"the message must name what was missing, got: {message}"


# ---------------------------------------------------------------------------
# literal_eval semantics: what the accepted value set actually is.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dsl", "expected"),
    [
        ("Bash(-1)", -1),
        ("Bash(1 + 2j)", 1 + 2j),
        ("Bash(None)", None),
        ("Bash(True)", True),
        ("Bash(1.5)", 1.5),
        ("Bash(-2.5)", -2.5),
    ],
    ids=["negative-int", "complex", "none", "true", "float", "negative-float"],
)
def test_signed_and_special_number_literals_are_accepted(dsl, expected):
    """Real ``literal_eval`` accepts unary-minus numbers and complex; a
    hand-rolled evaluator or a regex would not.

    This is what distinguishes "we call ``literal_eval``" from "we look like we
    call ``literal_eval``".  ``1 + 2`` is rejected (see the non-literal tests) --
    complex literals are the one ``BinOp`` ``literal_eval`` permits.
    """
    parsed = parse_call(dsl)

    assert len(parsed.args) == 1, f"{dsl} should yield one positional, got {parsed.args!r}"
    value = parsed.args[0]
    assert type(value) is type(expected), (
        f"{dsl} should yield a {type(expected).__name__}, got {type(value).__name__} {value!r}"
    )
    assert value == expected, f"{dsl} should yield {expected!r}, got {value!r}"


@pytest.mark.parametrize(
    ("dsl", "expected_type", "expected"),
    [
        ('Bash(b"\\x00\\xff")', bytes, b"\x00\xff"),
        ("Bash({1, 2})", set, {1, 2}),
        ("Bash((1, 2))", tuple, (1, 2)),
        ('Bash(["a", "b"])', list, ["a", "b"]),
        ('Bash({"a": 1})', dict, {"a": 1}),
        ("Bash([[1], [2]])", list, [[1], [2]]),
    ],
    ids=["bytes", "set", "tuple", "list", "dict", "nested-list"],
)
def test_container_and_bytes_literals_pass_through_unchanged(dsl, expected_type, expected):
    """FINDING: the accepted value set is wider than the schema types.

    ``bytes``, ``set`` and ``tuple`` are literals, so they are accepted -- and
    ``bind_arguments`` does not type-check, so one of these can reach a
    parameter the schema describes as a string.  Pinned so that adding schema
    validation later is a visible, deliberate change.
    """
    parsed = parse_call(dsl)

    assert len(parsed.args) == 1, f"{dsl} should yield one positional, got {parsed.args!r}"
    value = parsed.args[0]
    assert isinstance(value, expected_type), (
        f"{dsl} should yield a {expected_type.__name__}, got {type(value).__name__}"
    )
    assert value == expected, f"{dsl} should yield {expected!r}, got {value!r}"


def test_overflowing_float_literal_becomes_infinity():
    """DEFECT: nothing bounds the magnitude of a numeric literal.

    ``1e400`` is a perfectly good Python float literal that overflows to
    ``inf``.  Nothing downstream is told.
    """
    parsed = parse_call("Bash(1e400)")

    assert len(parsed.args) == 1, f"got {parsed.args!r}"
    value = parsed.args[0]
    assert isinstance(value, float), f"1e400 must still be a float, got {type(value).__name__}"
    assert math.isinf(value), f"1e400 should overflow to infinity, got {value!r}"


def test_infinite_timeout_keyword_reaches_the_bound_arguments():
    """DEFECT: ``timeout=inf`` survives the whole pipeline.

    The module docstring's own example is
    ``Background(call='Bash("ls")', timeout=300000)``; substituting an
    overflowing exponent for the integer produces a tool invocation whose
    timeout is infinite, with no error at any stage.
    """
    manager = ToolManager()
    spec = manager.register(_background_tool, name="Background")
    assert spec is not None, "the Background fixture tool must register"

    bound = bind_arguments(spec, parse_call('Background(call=\'Bash("ls")\', timeout=1e400)'))

    assert bound["call"] == 'Bash("ls")', f"the string argument must survive, got {bound!r}"
    assert math.isinf(bound["timeout"]), (
        f"an infinite timeout reaches the tool unchallenged, got {bound['timeout']!r}"
    )


# ---------------------------------------------------------------------------
# Duplicates, collisions and other silent resolutions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dsl", "repeated"),
    [
        ('Bash(command="a", command="b")', "command"),
        ('Bash(a="1", a="2", a="3")', "a"),
        ('Bash(a="1", b="2", a="3")', "a"),
    ],
    ids=["two-values", "three-values", "non-adjacent"],
)
def test_duplicate_keyword_is_rejected_instead_of_silently_last_wins(dsl, repeated):
    """A repeated keyword is refused, matching CPython.

    This closed a defect.  ``ast.parse(mode="eval")`` does not run CPython's
    duplicate-keyword check -- only ``compile()`` does -- so
    ``f(command="a", command="b")`` parsed cleanly and a plain dict assignment
    silently discarded ``"a"``.  The model got no signal that half of its call
    had been thrown away.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "not a tool call" in message, f"got: {message}"
    assert f"'{repeated}' is repeated" in message, (
        f"the message must name the repeated parameter, got: {message}"
    )
    assert excinfo.value.__cause__ is None, (
        "the duplicate check raises directly, before any literal evaluation"
    )


@pytest.mark.parametrize(
    "dsl",
    [
        'Bash(command="a")',
        'Bash(a="1", b="2")',
        'Bash("a", key="v")',
    ],
    ids=["single", "distinct", "positional-and-keyword"],
)
def test_distinct_keywords_are_unaffected(dsl):
    """The duplicate check must not fire on a keyword supplied exactly once.

    Present so that widening the rejection cannot be satisfied by rejecting all
    keyword arguments.
    """
    parsed = parse_call(dsl)

    assert parsed.tool_name == "Bash", f"got {parsed.tool_name!r}"
    assert len(parsed.kwargs) <= 2, f"got {parsed.kwargs!r}"


def test_duplicate_keyword_rejection_agrees_with_cpython_compile():
    """The control: the parser now refuses exactly what ``compile()`` refuses.

    ``ast.parse`` and ``compile`` disagree about duplicate keywords -- that
    disagreement was the defect.  ``parse_call`` now re-implements the check
    ``compile()`` performs but ``ast.parse`` skips.
    """
    source = 'Bash(command="a", command="b")'

    with pytest.raises(SyntaxError):
        compile(source, "<string>", "eval")

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(source)

    assert "is repeated" in str(excinfo.value), (
        f"the parser must refuse what CPython refuses, got: {excinfo.value}"
    )


def test_positional_and_keyword_for_one_parameter_is_rejected(mixed_spec):
    """A parameter supplied both ways is refused rather than silently resolved.

    This closed a defect.  CPython raises ``TypeError: got multiple values for
    argument 'first'`` for the equivalent Python call; here the positional was
    silently overwritten by the keyword and the model got neither an error nor
    any hint that its first argument had been thrown away -- and a
    positional/keyword pair is exactly what an LLM emits when it is unsure of a
    tool's arity.

    Detection necessarily lives in ``bind_arguments``: whether the first
    positional binds to ``first`` is a property of the spec, which
    ``parse_call`` never sees.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(mixed_spec, parse_call('Mixed("from-positional", first="from-keyword")'))

    message = str(excinfo.value)
    assert "Got multiple values for parameter 'first'" in message, f"got: {message}"
    assert "Mixed" in message, f"the message must name the tool, got: {message}"
    assert "from-positional" not in message, (
        f"neither value may be silently chosen; the message was: {message}"
    )


def test_a_keyword_for_an_earlier_parameter_than_the_positionals_is_rejected(mixed_spec):
    """The collision is detected wherever the keyword lands in the prefix.

    ``Mixed("a", "b", first="x")`` collides with the *first* of three
    positionals, not with the one adjacent to the keyword.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(mixed_spec, parse_call('Mixed("a", "b", first="x")'))

    assert "'first'" in str(excinfo.value), f"got: {excinfo.value}"


def test_several_colliding_parameters_are_all_reported(mixed_spec):
    """Every collided parameter is named, not just the first."""
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(mixed_spec, parse_call('Mixed("a", "b", first="x", second="y")'))

    message = str(excinfo.value)
    headline = message.split(" for tool ")[0]
    assert headline == "Got multiple values for parameters 'first', 'second'", (
        f"the headline must name exactly the collided parameters, got: {headline!r}"
    )
    assert "third" in message.split("expects parameters")[1], (
        f"'third' belongs in the expected-parameter list, got: {message}"
    )


def test_keyword_overlays_an_earlier_positional_without_error(mixed_spec):
    """The supported overlay feature: a *later* argument may be spelled out
    while an earlier one stays positional.

    This is the boundary the collision check must not cross.  ``second`` is
    outside the prefix the first positional consumes, so it is not a collision.
    """
    bound = bind_arguments(mixed_spec, parse_call('Mixed("from-positional", second="explicit")'))

    assert bound == {"first": "from-positional", "second": "explicit"}, (
        f"the positional should be kept and the later keyword overlaid, got {bound!r}"
    )


def test_a_keyword_beyond_the_positional_prefix_is_not_a_collision(mixed_spec):
    """``Mixed("pos", third=1)`` skips a slot and is legitimate."""
    bound = bind_arguments(mixed_spec, parse_call('Mixed("pos", third=1)'))

    assert bound == {"first": "pos", "third": 1}, (
        f"'third' is outside the consumed prefix, so it is not a collision, got {bound!r}"
    )


def test_call_name_is_checked_against_the_spec_name(mixed_spec):
    """A call naming a different tool than the spec is refused, not mis-bound.

    This closed a defect.  ``bind_arguments`` never compared the two names, so
    ``bind_arguments(spec_named_Two, parse_call('Write("payload")'))`` bound
    Write's argument to Two's parameter with no signal at all.  The check is a
    consistency assertion, not an allowlist: it adds no knowledge of which
    tools exist, it only confirms the handed spec is the spec the call named.
    """
    parsed = parse_call('Write("payload")')
    assert parsed.tool_name == "Write", "precondition: the call must name a different tool"
    assert parsed.tool_name != mixed_spec.name, (
        "precondition: the call name and the spec name must differ for this test to prove anything"
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(mixed_spec, parsed)

    message = str(excinfo.value)
    assert "names 'Write'" in message, f"the message must name what the call asked for, got: {message}"
    assert "this spec is 'Mixed'" in message, (
        f"the message must name the spec that was handed in, got: {message}"
    )
    assert "payload" not in message, (
        f"no argument may be bound before the mismatch is caught, got: {message}"
    )


def test_a_matching_call_name_still_binds(mixed_spec):
    """The companion: equal names are unaffected by the consistency check.

    Present so the check cannot be satisfied by rejecting every call.
    """
    bound = bind_arguments(mixed_spec, parse_call('Mixed("payload")'))

    assert bound == {"first": "payload"}, f"got {bound!r}"


def test_name_mismatch_is_reported_before_any_other_validation():
    """The mismatch is checked first, so it wins over a would-be arity error.

    Fail fast: a dispatcher bug is the more fundamental problem and the one the
    operator needs to see first.
    """
    manager = ToolManager()
    spec = manager.register(_two_param_tool, name="Two")
    assert spec is not None, "the Two fixture tool must register"

    # Three positionals would otherwise raise "Too many positional arguments".
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, parse_call('Write("a", "b", "c")'))

    assert "refusing to bind" in str(excinfo.value), f"got: {excinfo.value}"


def test_name_mismatch_names_are_bounded_in_the_message():
    """A huge tool name is clipped rather than echoed whole.

    Both names are attacker-influenced -- the call's from the model, the spec's
    from a third-party MCP server -- so neither may be echoed unbounded.
    """
    manager = ToolManager()
    spec = manager.register(_two_param_tool, name="n" * 50_000)
    assert spec is not None, "the oversized-name fixture tool must register"

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, parse_call('Write("payload")'))

    message = str(excinfo.value)
    assert len(message) < 400, f"the message must stay bounded, got {len(message)} characters"
    assert "more characters" in message, f"the clipping marker is missing: {message}"


# ---------------------------------------------------------------------------
# Unicode, escapes, untrusted text handling and message hygiene.
# ---------------------------------------------------------------------------


def test_unicode_tool_name_and_argument_survive_without_normalisation():
    """Non-ASCII text must round-trip byte-for-byte in meaning.

    Silent NFC/NFKC folding here would rewrite tool arguments behind the
    model's back -- ``Bash("café")`` could stop matching a path on disk.
    """
    parsed = parse_call('Ünïcödé("héllo 🌍 中文")')

    assert parsed.tool_name == "Ünïcödé", f"got {parsed.tool_name!r}"
    assert parsed.args == ["héllo 🌍 中文"], f"got {parsed.args!r}"
    assert unicodedata.normalize("NFC", parsed.args[0]) == parsed.args[0], (
        "the parser must not normalise argument text"
    )


@pytest.mark.parametrize(
    ("dsl", "expected"),
    [
        (r'Bash("café")', "café"),
        (r'Bash("a\tb")', "a\tb"),
        (r'Bash("a\\b")', "a\\b"),
        (r'Bash("a\x00b")', "a\x00b"),
        (r'Bash("a\nb")', "a\nb"),
        (r'Bash("a\rb")', "a\rb"),
    ],
    ids=["unicode-escape", "tab", "backslash", "nul-escape", "newline-escape", "carriage-return"],
)
def test_escape_sequences_in_arguments_are_decoded(dsl, expected):
    """An escape is *decoded*, which is how a shell command can carry a newline."""
    parsed = parse_call(dsl)

    assert parsed.args == [expected], f"{dsl!r} should decode to {expected!r}, got {parsed.args!r}"


@pytest.mark.parametrize(
    "dsl",
    [
        'Bash("a\nb")',
        "Bash('a\nb')",
        'Bash("unterminated)',
        "Bash('mismatched\")",
        'Bash("a" \'b\nc")',
        'Bash("a\\',
    ],
    ids=[
        "raw-newline-double-quoted",
        "raw-newline-single-quoted",
        "unterminated-double",
        "mismatched-quotes",
        "implicit-concat-raw-newline",
        "trailing-backslash",
    ],
)
def test_string_that_cannot_be_tokenised_is_a_parse_failure(dsl):
    """Hostile text that the tokenizer rejects surfaces as the builtin error."""
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert "could not be parsed as a ToolName(...) expression" in message, f"got: {message}"
    assert isinstance(excinfo.value.__cause__, SyntaxError), (
        f"a tokeniser failure is a SyntaxError, got {type(excinfo.value.__cause__).__name__}"
    )
    assert "\n" not in message, (
        f"the message must stay on one line for the TUI error pane, got: {message!r}"
    )


def test_nul_byte_in_the_source_is_wrapped_and_not_echoed_raw():
    """A NUL is rejected by CPython before parsing, and never reaches the UI raw."""
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call('Bash("a\x00b")')

    message = str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError), (
        f"CPython rejects a NUL in the source with ValueError, got "
        f"{type(excinfo.value.__cause__).__name__}"
    )
    assert "\x00" not in message, "a raw NUL in a message breaks the TUI renderer"
    assert "\\x00" in message, f"the excerpt should still show where the NUL was, got: {message!r}"


def test_long_argument_value_is_preserved_intact():
    """The excerpt cap applies to *messages*, never to argument values.

    Truncating a value would silently corrupt a long command, so this asserts
    the full 20 000 characters come back.
    """
    value = "a" * 20_000

    parsed = parse_call(f'Bash("{value}")')

    assert len(parsed.args) == 1, f"got {len(parsed.args)} positional arguments"
    assert parsed.args[0] == value, "an unbounded argument must not be silently truncated"
    assert len(parsed.args[0]) == 20_000, f"got {len(parsed.args[0])} characters"


def test_error_message_bounds_an_unbounded_offending_body():
    """The message is capped, reports the real length, and leaks no more.

    The model controls this text and it is unbounded, so an error pane has no
    use for a megabyte of it.
    """
    hostile = "x" * 20_000
    dsl = f'Bash("{hostile}'  # unterminated, so this takes the parse-failure path

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert f"({len(dsl)} characters total)" in message, (
        f"the truncation marker must report the real length, got: {message[-80:]!r}"
    )
    assert hostile not in message, "the whole untrusted body leaked into the message"
    assert "x" * 20_000 not in message, "the untrusted body leaked into the message"
    assert len(message) < 1_000, f"the message should stay bounded, got {len(message)} characters"


@pytest.mark.parametrize(
    "dsl",
    [
        'Bash("a\nb")',
        "Bash(**kwargs)",
        "Bash(*args)",
        "Bash",
        'os.system("ls")',
        '"' + "x" * 500,
    ],
    ids=["raw-newline", "star-star", "star", "bare-name", "attribute-callee", "long-body"],
)
def test_error_message_is_one_non_empty_line(dsl):
    """Every failure path renders a single printable line.

    The error pane and the log are both line-oriented; a raw newline or a
    multi-kilobyte ``str(SyntaxError)`` breaks both.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    message = str(excinfo.value)
    assert message, "an empty message tells the model nothing"
    assert "\n" not in message, f"the message must be one line, got: {message!r}"
    assert len(message) < 600, f"the message should stay bounded, got {len(message)} characters"


@pytest.mark.parametrize(
    "dsl",
    [' Bash("ls")', 'Bash("ls") ', ' Bash("ls") ', '\n\t Bash("ls")\n', '\r\nBash("ls")\r\n'],
    ids=["leading", "trailing", "both", "indented-newline", "crlf"],
)
def test_surrounding_whitespace_is_stripped_before_parsing(dsl):
    """Leading and trailing whitespace no longer costs the model its call.

    This closed a defect: ``ast.parse`` reads a leading space as an indented
    statement block rather than an expression, so ``' Bash("ls")'`` was an
    ``IndentationError`` while ``'Bash("ls") '`` already worked.  The asymmetry
    is an artefact of Python's grammar, not a property of the DSL.
    """
    parsed = parse_call(dsl)

    assert parsed.tool_name == "Bash", f"got {parsed.tool_name!r}"
    assert parsed.args == ["ls"], f"got {parsed.args!r}"


def test_stripping_does_not_enable_a_smuggled_second_statement():
    """The strip is not a hole: only *outer* whitespace is removed.

    Stripping happens before parsing, so text that appends a statement is
    still rejected afterwards -- ``'  Bash("ls"); evil()  '`` must not become a
    successful parse just because its edges were trimmed.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call('  Bash("ls"); evil()  ')

    assert "could not be parsed" in str(excinfo.value), f"got: {excinfo.value}"
    assert isinstance(excinfo.value.__cause__, SyntaxError), (
        f"the appended statement must still be a syntax error, got "
        f"{type(excinfo.value.__cause__).__name__}"
    )


def test_whitespace_inside_quoted_arguments_is_untouched():
    """Only the outside of the call is trimmed; argument text is preserved.

    A strip applied after parsing, or one that reached inside quotes, would
    silently change a shell command.
    """
    parsed = parse_call('  Bash("  padded  ")  ')

    assert parsed.args == ["  padded  "], (
        f"whitespace inside the argument must survive, got {parsed.args!r}"
    )


@pytest.mark.parametrize(
    ("dsl", "expected_args", "expected_kwargs"),
    [
        ('Bash("ls")  # a trailing comment', ["ls"], {}),
        ('Bash(\n  "ls"\n)', ["ls"], {}),
        ('Bash( "a" , key = "v" )', ["a"], {"key": "v"}),
        ('Bash("  padded  ")', ["  padded  "], {}),
        ('Bash("")', [""], {}),
        ('Bash(\t"a",\tkey="v"\t)', ["a"], {"key": "v"}),
    ],
    ids=[
        "trailing-comment",
        "newlines-outside-quotes",
        "loose-spacing",
        "quoted-padding",
        "empty-string",
        "tabs",
    ],
)
def test_lexical_noise_outside_the_argument_is_tolerated(dsl, expected_args, expected_kwargs):
    """Whitespace, comments and empty strings around a legal call are fine.

    Quoted padding must survive untouched -- a stripping regex would eat it.
    """
    parsed = parse_call(dsl)

    assert parsed.tool_name == "Bash", f"got {parsed.tool_name!r}"
    assert parsed.args == expected_args, f"{dsl!r} should yield {expected_args!r}, got {parsed.args!r}"
    assert parsed.kwargs == expected_kwargs, (
        f"{dsl!r} should yield {expected_kwargs!r}, got {parsed.kwargs!r}"
    )


@pytest.mark.parametrize(
    "dsl",
    [
        'Bash("a"); import os',
        'Bash("a")\nimport os',
        'Bash("a") or __import__("os")',
        'Bash("a"), os.system("ls")',
        "import os; Bash('a')",
    ],
    ids=["semicolon", "newline-statement", "or-chain", "tuple-with-call", "leading-statement"],
)
def test_code_appended_to_a_call_is_rejected(dsl):
    """Text that smuggles a second statement in must never parse.

    ``mode="eval"`` makes the statement forms syntax errors and the
    ``or``/tuple forms the wrong AST shape; either way nothing is evaluated.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    assert "not a tool call" in str(excinfo.value) or "could not be parsed" in str(
        excinfo.value
    ), f"got: {excinfo.value}"


@pytest.mark.parametrize(
    "dsl",
    [
        "Bash(" + "[" * 2_000 + "]" * 2_000 + ")",
        "Bash(" + "(" * 2_000 + "1" + ")" * 2_000 + ")",
        "Bash(" + '"' + "a" * 100_000,
    ],
    ids=["deep-lists", "deep-parens", "huge-string"],
)
def test_pathologically_nested_input_is_contained(dsl):
    """Untrusted nesting must not escape as a raw interpreter error.

    ``RecursionError`` and ``MemoryError`` are named in ``_PARSE_FAILURES``
    precisely so that stack exhaustion near the parser cannot propagate into
    the caller.
    """
    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(dsl)

    assert "not a tool call" in str(excinfo.value), f"got: {excinfo.value}"
    # The interpreter's own limits are still intact afterwards.
    assert parse_call("Bash(1)").args == [1], "parsing must still work after a nested failure"


# ---------------------------------------------------------------------------
# bind_arguments: order, required versus default, and what is not checked.
# ---------------------------------------------------------------------------


def test_positional_binding_consumes_an_optional_parameter_rather_than_skipping_it(
    mixed_spec,
):
    """The second positional lands on the *optional* ``second``, not on ``third``.

    Binding iterates every declared property in signature order, so a positional
    never skips a defaulted parameter to reach a later required one.  A caller
    relying on skip-to-required semantics would silently fill the wrong slot.
    """
    bound = bind_arguments(mixed_spec, parse_call('Mixed("a", "b")'))

    assert bound == {"first": "a", "second": "b"}, (
        f"the second positional should consume the optional 'second', got {bound!r}"
    )
    assert "third" not in bound, f"'third' must be left alone, got {bound!r}"
    assert mixed_spec.parameters_schema["properties"]["second"]["default"] == "DEFAULT", (
        "precondition: 'second' really is optional"
    )


def test_positional_binding_fills_every_declared_property_in_order(mixed_spec):
    """All three positionals land on their declared slots, in order."""
    bound = bind_arguments(mixed_spec, parse_call('Mixed("a", "b", 3)'))

    assert bound == {"first": "a", "second": "b", "third": 3}, (
        f"positionals must fill first, second, third in order, got {bound!r}"
    )


def test_omitted_optional_parameter_is_absent_so_the_function_default_applies(
    mixed_spec,
):
    """The bound dict OMITS unsupplied optionals; the schema default is not copied.

    Omitting is the right choice -- the function's own default is the single
    source of truth, and baking the registration-time snapshot in would freeze
    it.
    """
    bound = bind_arguments(mixed_spec, parse_call('Mixed("a")'))

    assert bound == {"first": "a"}, f"only the supplied parameter should be bound, got {bound!r}"
    assert "second" not in bound, f"an unsupplied optional must be absent, got {bound!r}"
    assert "third" not in bound, f"an unsupplied optional must be absent, got {bound!r}"
    assert _mixed_tool(**bound) == ("a", "DEFAULT", 7), (
        "the function's own defaults must be what actually applies"
    )


def test_required_list_wins_over_a_property_default():
    """A parameter with a schema ``default`` that is also in ``required`` is
    treated as REQUIRED.

    ``ToolManager.register`` can never emit this shape -- a Python parameter
    either has a default or is required -- so only an MCP-supplied or explicit
    ``parameters_schema=`` tool can reach it.  There, ``required`` is the
    stricter statement and it wins.
    """
    spec = _spec_named(
        "Weird",
        {
            "type": "object",
            "properties": {"a": {"type": "string", "default": "D"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        },
    )
    assert spec.parameters_schema["properties"]["a"]["default"] == "D", (
        "precondition: 'a' really does carry a default"
    )
    assert spec.parameters_schema["required"] == ["a", "b"], (
        "precondition: 'a' really is listed as required"
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, parse_call("Weird()"))
    assert "Missing required parameters for tool 'Weird'" in str(excinfo.value), (
        f"both required names must be reported, got: {excinfo.value}"
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, parse_call('Weird(a="x")'))
    assert "Missing required parameter for tool 'Weird': b" in str(excinfo.value), (
        f"'a' must count as supplied so only 'b' is reported, got: {excinfo.value}"
    )

    assert bind_arguments(spec, parse_call('Weird(a="x", b="y")')) == {"a": "x", "b": "y"}, (
        "both supplied parameters must bind"
    )


def test_container_argument_binds_to_a_scalar_parameter_without_type_check(mixed_spec):
    """FINDING: the schema's ``type`` is ignored, so a list reaches a ``str``.

    Nothing in ``bind_arguments`` reads ``properties[name]["type"]``, so the
    tool runs with a value its own annotation forbids.
    """
    bound = bind_arguments(mixed_spec, parse_call('Mixed(["a", "b"])'))

    assert mixed_spec.parameters_schema["properties"]["first"]["type"] == "string", (
        "precondition: the schema declares this parameter as a string"
    )
    assert bound == {"first": ["a", "b"]}, f"a list should bind unchecked, got {bound!r}"
    assert _mixed_tool(**bound) == (["a", "b"], "DEFAULT", 7), (
        "the wrong type must reach the function body unchallenged"
    )


def test_falsy_argument_values_are_bound_and_not_treated_as_missing():
    """A falsy value is present, not absent.

    ``missing`` is computed with ``name not in bound``, so ``False``, ``0`` and
    ``""`` all count as supplied.  A ``if not value`` check anywhere in this
    path would report them as absent.
    """
    manager = ToolManager()
    spec = manager.register(_flags_tool, name="Flags")
    assert spec is not None, "the Flags fixture tool must register"

    bound = bind_arguments(spec, parse_call("Flags(False)"))
    assert bound == {"flag": False}, (
        f"a falsy required argument is present, not missing; got {bound!r}"
    )
    assert _flags_tool(**bound) == (False, 1, "x"), (
        "the defaults must survive alongside a falsy required argument"
    )

    bound = bind_arguments(spec, parse_call("Flags(True, 0, '')"))
    assert bound == {"flag": True, "count": 0, "note": ""}, (
        f"every falsy value must reach the bound dict unchanged, got {bound!r}"
    )


def test_bound_arguments_are_ready_to_pass_to_the_registered_function():
    """The end-to-end contract: the result splats straight into the tool."""
    manager = ToolManager()
    spec = manager.register(_mixed_tool, name="Mixed")
    assert spec is not None, "the Mixed fixture tool must register"

    bound = bind_arguments(spec, parse_call('Mixed("a", "b", 3)'))

    assert type(bound) is dict, f"expected a plain dict, got {type(bound).__name__}"
    assert bound == {"first": "a", "second": "b", "third": 3}, f"got {bound!r}"
    assert spec.func(**bound) == ("a", "b", 3), "the bound dict must splat into the tool"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "additionalProperties": True,
        },
    ],
    ids=["flag-absent", "flag-true"],
)
def test_unknown_keyword_is_rejected_whatever_additional_properties_says(schema):
    """``additionalProperties`` is never consulted; ``properties`` is the gate.

    A typo'd keyword must be caught here rather than surfacing as a late
    ``TypeError`` from inside the tool.
    """
    spec = _spec_named("Bash", schema)

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Bash", kwargs={"cmomand": "ls"}))

    message = str(excinfo.value)
    assert "Unknown parameter 'cmomand'" in message, f"got: {message}"
    assert "Bash" in message, f"the message must name the tool, got: {message}"


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "object"},
        {"properties": None},
        {"properties": []},
        {"properties": "command"},
        None,
    ],
    ids=["empty", "no-properties-key", "properties-none", "properties-list", "properties-string", "none"],
)
def test_schema_without_usable_properties_is_a_zero_parameter_tool(schema):
    """Every defensive ``isinstance`` guard collapses to the same safe outcome.

    A malformed or server-authored schema must not raise a raw ``KeyError`` or
    ``TypeError`` out of the tool layer; it degrades to "no parameters", where
    any keyword is reported as unknown.
    """
    spec = _spec_named("Bare", schema)

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Bare", kwargs={"anything": 1}))

    message = str(excinfo.value)
    assert "Unknown parameter 'anything'" in message, f"got: {message}"
    assert "expects parameters: (none)" in message, f"got: {message}"


@pytest.mark.parametrize("required", [None, "command", 7], ids=["none", "string", "int"])
def test_non_list_required_means_nothing_is_required(required):
    """``required`` is guarded by ``isinstance(list)``; anything else means the
    tool declares no required parameters.

    A server-authored schema with ``"required": "command"`` must not be read as
    a requirement, and must not crash either.
    """
    spec = _spec_named(
        "Bare",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": required},
    )

    bound = bind_arguments(spec, ParsedCall(tool_name="Bare", kwargs={"command": "ls"}))

    assert bound == {"command": "ls"}, f"got {bound!r}"


def test_positional_order_follows_properties_not_the_required_list():
    """Binding iterates ``properties``, never ``required``.

    An MCP schema is free to list ``required`` in a different order from its
    properties; the declaration order is what positions follow.
    """
    spec = _spec_named(
        "Two",
        {
            "type": "object",
            "properties": {"b": {"type": "string"}, "a": {"type": "string"}},
            "required": ["a", "b"],
        },
    )

    bound = bind_arguments(spec, ParsedCall(tool_name="Two", args=["x", "y"]))

    assert bound == {"b": "x", "a": "y"}, (
        f"the first positional must go to the first *declared* property, got {bound!r}"
    )


def test_too_many_positionals_message_is_singular_for_a_one_parameter_tool():
    """The count wording follows the number of declared parameters."""
    spec = _spec_named(
        "Bash",
        {"type": "object", "properties": {"only": {"type": "string"}}, "required": ["only"]},
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Bash", args=["a", "b"]))

    message = str(excinfo.value)
    assert "2 given but only 1 parameter is declared" in message, (
        f"a single declared parameter must read singularly, got: {message}"
    )
    assert "only" in message, f"the message must list what it expected, got: {message}"


def test_too_many_positionals_message_is_plural_for_a_multi_parameter_tool():
    """Two declared parameters must read plural, pinning the other side of the
    same ternary as the singular case."""
    spec = _spec_named(
        "Two",
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        },
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Two", args=["1", "2", "3"]))

    assert "3 given but only 2 parameters are declared" in str(excinfo.value), (
        f"two declared parameters must read plural, got: {excinfo.value}"
    )


def test_bind_arguments_does_not_mutate_the_spec_schema_or_the_parsed_call():
    """Binding is pure: the registry snapshot and the parsed call survive it.

    ``ToolManager`` hands out one shared ``ToolSpec`` per tool, so a bind that
    mutated its schema would corrupt every later call in the session.
    """
    spec = _spec_named(
        "Two",
        {
            "type": "object",
            "properties": {"b": {"type": "string"}, "a": {"type": "string"}},
            "required": ["b"],
        },
    )
    schema_before = copy.deepcopy(spec.parameters_schema)
    parsed = ParsedCall(tool_name="Two", args=["x"], kwargs={"a": "y"})

    bound = bind_arguments(spec, parsed)
    bound["b"] = "MUTATED"

    assert spec.parameters_schema == schema_before, (
        f"the registered schema must not change, got {spec.parameters_schema!r}"
    )
    assert parsed.args == ["x"] and parsed.kwargs == {"a": "y"}, (
        f"the parsed call must not change, got {parsed!r}"
    )
    assert bound == {"b": "MUTATED", "a": "y"}, (
        f"the returned dict must be the caller's own, got {bound!r}"
    )

# ---------------------------------------------------------------------------
# Error messages are bounded in the size AND the count of the identifiers they
# echo.  `_MAX_EXCERPT` capped the offending *body*; a parameter name is
# unbounded too, and Python has no identifier length limit.
# ---------------------------------------------------------------------------


def _huge_name() -> str:
    """A 50 000-character identifier, the size the review probe used."""
    return "n" * 50_000


def test_repeated_keyword_message_does_not_echo_the_whole_identifier():
    """A huge repeated keyword is clipped, not reproduced."""
    name = _huge_name()

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(f'Bash({name}=1, {name}=2)')

    message = str(excinfo.value)
    assert len(message) < 600, f"the message must stay bounded, got {len(message)} characters"
    assert name not in message, "the whole identifier leaked into the message"
    assert "is repeated" in message, f"got: {message}"
    assert "more characters" in message, f"the clipping marker is missing: {message}"


def test_literal_failure_label_does_not_echo_the_whole_identifier():
    """The `for parameter '<name>'` label is clipped too.

    This label reaches the message through ``_literal``, so it is a separate
    echo site from the repeated-keyword one.
    """
    name = _huge_name()

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(f'Bash({name}=eval("x"))')

    message = str(excinfo.value)
    assert len(message) < 600, f"the message must stay bounded, got {len(message)} characters"
    assert name not in message, "the whole identifier leaked into the message"
    assert "is not a literal value" in message, f"got: {message}"


def test_unknown_parameter_message_does_not_echo_the_whole_identifier():
    """A huge unknown keyword is clipped."""
    name = _huge_name()
    spec = _spec_named(
        "Two",
        {"type": "object", "properties": {"first": {"type": "string"}}, "required": ["first"]},
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Two", kwargs={name: "v"}))

    message = str(excinfo.value)
    assert len(message) < 400, f"the message must stay bounded, got {len(message)} characters"
    assert name not in message, "the whole identifier leaked into the message"
    assert "Unknown parameter" in message, f"got: {message}"


def test_summary_does_not_echo_a_whole_schema_supplied_parameter_name():
    """An MCP schema's property name is a second untrusted source.

    The name reaches the message through `summary`, which is appended to four
    different messages, so it is the widest-reaching echo site in the module.
    """
    spec = _spec_named("Mcp", {"type": "object", "properties": {_huge_name(): {"type": "string"}}})

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Mcp", kwargs={"nope": 1}))

    message = str(excinfo.value)
    assert len(message) < 400, f"the message must stay bounded, got {len(message)} characters"
    assert _huge_name() not in message, "the whole schema name leaked into the message"
    assert "expects parameters:" in message, f"got: {message}"


def test_summary_bounds_the_number_of_parameter_names_not_just_their_length():
    """Clipping each name is not enough; the *count* must be bounded too.

    A schema declaring ten thousand short properties would otherwise still
    produce a message proportional to the schema.
    """
    spec = _spec_named(
        "Wide",
        {"type": "object", "properties": {f"p{index}": {"type": "string"} for index in range(10_000)}},
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Wide", kwargs={"nope": 1}))

    message = str(excinfo.value)
    assert len(message) < 400, f"the message must stay bounded, got {len(message)} characters"
    assert "and 9992 more" in message, f"the overflow marker is missing: {message}"
    assert "p9999" not in message, f"names past the limit must be summarised, got: {message}"


def test_missing_required_message_bounds_the_number_of_names():
    """The missing-parameter list is bounded in count as well as length."""
    spec = _spec_named(
        "Wide",
        {
            "type": "object",
            "properties": {f"p{index}": {"type": "string"} for index in range(10_000)},
            "required": [f"p{index}" for index in range(10_000)],
        },
    )

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Wide"))

    message = str(excinfo.value)
    assert len(message) < 500, f"the message must stay bounded, got {len(message)} characters"
    assert "and 9992 more" in message, f"the overflow marker is missing: {message}"


def test_collision_message_bounds_the_names_it_lists():
    """The collision list is bounded too.

    Collided names come from the schema side of the intersection, so this is the
    MCP-sourced variant of the same bound.
    """
    spec = _spec_named("Mcp", {"type": "object", "properties": {_huge_name(): {"type": "string"}}})

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Mcp", args=["v"], kwargs={_huge_name(): "w"}))

    message = str(excinfo.value)
    assert len(message) < 500, f"the message must stay bounded, got {len(message)} characters"
    assert _huge_name() not in message, "the whole identifier leaked into the message"
    assert "Got multiple values" in message, f"got: {message}"


@pytest.mark.parametrize("size", [10, 1_000, 10_000, 100_000, 1_000_000])
def test_message_length_is_flat_in_the_identifier_length(size):
    """The bound must be constant, not merely smaller than the input.

    A cap that only trims by a fixed fraction would still grow with the input;
    this asserts the message length plateaus.
    """
    name = "n" * size

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        parse_call(f'Bash({name}=1, {name}=2)')

    assert len(str(excinfo.value)) < 600, (
        f"a {size}-character identifier produced a {len(str(excinfo.value))}-character message"
    )


def test_ordinary_parameter_names_are_not_clipped():
    """The bound must not fire on the names real tools actually use.

    Without this, a fix that clipped at an absurdly small limit would pass every
    test above.
    """
    parsed = parse_call('Bash(command="ls", timeout=300000)')
    assert parsed.kwargs == {"command": "ls", "timeout": 300000}, f"got {parsed.kwargs!r}"

    manager = ToolManager()
    spec = manager.register(_mixed_tool, name="Mixed")
    assert spec is not None, "the Mixed fixture tool must register"

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, parse_call('Mixed("a", "b", first="x")'))

    message = str(excinfo.value)
    assert "first" in message and "second" in message and "third" in message, (
        f"real parameter names must appear in full, got: {message}"
    )
    assert "more characters" not in message, (
        f"real parameter names must not be marked as truncated, got: {message}"
    )
    assert "and " not in message.split("expects parameters:")[1], (
        f"three real names fit under the count limit, got: {message}"
    )


def test_non_string_schema_keys_do_not_crash_the_message():
    """A non-``str`` schema key renders as its type instead of raising.

    ``', '.join(expected)`` would raise ``TypeError`` here, which is the same
    bug class as the error path that raises while reporting an error.
    """
    spec = _spec_named("Odd", {"type": "object", "properties": {7: {"type": "string"}}})

    with pytest.raises(MitoCodeBuiltinToolError) as excinfo:
        bind_arguments(spec, ParsedCall(tool_name="Odd", kwargs={"nope": 1}))

    assert "expects parameters: <int>" in str(excinfo.value), f"got: {excinfo.value}"
