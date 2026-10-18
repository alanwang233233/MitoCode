"""Parser for the nested tool-call mini-DSL produced by the model.

The DSL is the string a model emits when it wants one tool to call another
with a whole argument string already rendered, for example::

    Bash("python -m pytest test/")
    Grep(pattern="TODO", path="src")
    Background(call='Bash("ls")', timeout=300000)

The text is handed to us whole and is therefore untrusted.  It is parsed
with :func:`ast.parse` in ``mode="eval"`` rather than with a regex because
the format nests quotes and escapes (``Bash("echo 'hi'")``,
``Bash("echo \\"hi\\"")``) that a regular expression mangles.  Parsing builds
an AST and executes nothing; every value is then recovered with
:func:`ast.literal_eval`, which accepts only Python literals and cannot run
code even in principle.

Acceptance is deliberately narrow.  Only an :class:`ast.Call` whose callee
is a bare :class:`ast.Name` is a tool call, so ``os.system("ls")`` — an
:class:`ast.Attribute` — is rejected, as is anything that is not a call at
all.  A callee *name* is only ever carried forward as a string for the
caller to look up in a registry; nothing here resolves it to a callable, so
a name such as ``__import__`` is inert.  Any parse failure, including a
:class:`SyntaxError`, is re-raised as
:class:`~mitocode.core.tool.builtin.errors.MitoCodeBuiltinToolError` so a
malformed string cannot escape as a raw parser error.

    **What is deliberately *not* validated here.**  Two gaps were reviewed and
    accepted, because closing them here would duplicate a contract that already
    exists elsewhere rather than add safety:

    * **No schema type checking.**  :func:`bind_arguments` never reads
      ``properties[name]["type"]``, so ``Mixed(["a", "b"])`` binds a ``list`` to
      a parameter the schema describes as a ``string``.  That degrades safely:
      the tool runs, raises, and
      :func:`~mitocode.core.tool.builtin.errors.tool_boundary` turns the
      exception into an ``"Error: ..."`` line the model can read.  The tool's
      own signature is the real type contract; a second copy here would drift
      away from it.
    * **The accepted value set is wider than "strings, numbers and booleans".**
      :func:`ast.literal_eval` also admits ``bytes``, ``set``, ``tuple``,
      ``complex`` and nested containers, and a numeric literal can overflow --
      ``1e400`` becomes ``inf``, which is why the example above uses an integer
      ``timeout``.  The same degradation applies.  Callers that pass a parsed
      value onward to arithmetic (a timeout, a deadline, a poll interval) must
      clamp it themselves.

    Leading and trailing whitespace is stripped before parsing, because
    :func:`ast.parse` reads a leading space as an indented statement block
    rather than an expression, and the text is model-generated.
"""

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from mitocode.core.tool.builtin.errors import MitoCodeBuiltinToolError
from mitocode.core.tool.tool_manager import ToolSpec

#: Longest fragment of the offending text echoed back in an error message.
#: The input is untrusted and unbounded, and an error pane has no use for a
#: megabyte of model output.
_MAX_EXCERPT = 200

#: Longest single identifier echoed back in an error message.  The same
#: unbounded-input reasoning as :data:`_MAX_EXCERPT` applies verbatim to a
#: parameter name: ``keyword.arg`` comes from the model, and a property name
#: comes from a third-party MCP schema.  Python has no identifier length
#: limit, so without this a 50 000-character name lands in the message whole.
_MAX_NAME = 60

#: How many parameter names a single message will list.  Clipping each name is
#: not sufficient on its own: a schema declaring ten thousand properties would
#: still produce a message proportional to the schema, so the *count* is
#: bounded too.
_MAX_NAMES = 8

#: Failures :func:`ast.parse` can raise on hostile input.  ``SyntaxError``
#: covers malformed text; ``ValueError`` covers a NUL byte in the source,
#: which CPython rejects before parsing; the rest bound pathological
#: nesting, which is a property of untrusted input rather than a bug.
_PARSE_FAILURES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)

#: Failures :func:`ast.literal_eval` raises for a node that is not a literal.
#: Spelled out separately from :data:`_PARSE_FAILURES` rather than aliased, so
#: narrowing the parse-time catch does not silently narrow this one too.  A
#: binary expression, a name reference, a comprehension, a lambda, a nested
#: call, and a ``*args`` splat all land here; ``ValueError`` is the ordinary
#: "malformed node or string" case.
_LITERAL_FAILURES = (
    ValueError,
    TypeError,
    SyntaxError,
    MemoryError,
    RecursionError,
)


@dataclass
class ParsedCall:
    """The decomposed pieces of one DSL call.

    Attributes:
        tool_name: The bare callee name, e.g. ``"Bash"``.
        args: Positional argument values, already literal-evaluated.
        kwargs: Keyword argument values, keyed by the declared name.
    """

    tool_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _excerpt(text: Any) -> str:
    """Render untrusted *text* for inclusion in an error message.

    This function is total: it never raises.  It is called from inside
    ``except`` blocks, so an exception here would escape the error boundary
    entirely -- the same bug class as a ``__str__`` that raises, which
    :func:`~mitocode.core.tool.builtin.errors.safe_exception_message` exists to
    contain on the other side of this package.  Unlike that helper, the
    concern here is not an exception's message but *untrusted input*: the
    caller may hand over something that has no ``len`` at all, so anything
    without the string/bytes protocol is rendered as its type name instead of
    being measured.

    Args:
        text: The offending source, which may be arbitrarily long and may not
            be a string.

    Returns:
        A ``repr`` of at most :data:`_MAX_EXCERPT` characters, followed by a
        total-length marker when the text was truncated, or a ``<TypeName>``
        marker when *text* is neither ``str`` nor ``bytes``.
    """
    if not isinstance(text, (str, bytes)):
        return f"<{type(text).__name__}>"
    if len(text) <= _MAX_EXCERPT:
        return repr(text)
    return repr(text[:_MAX_EXCERPT]) + f"... ({len(text)} characters total)"


def _clip_name(name: Any, limit: int = _MAX_NAME) -> str:
    """Render one identifier for an error message, bounded.

    A sibling of :func:`_excerpt` rather than a reuse of it, because the two
    render different things.  :func:`_excerpt` renders *source text* and so
    wraps it in ``repr`` quotes and escapes non-ASCII; an identifier already
    carries its own quoting at the call site, so routing it through a ``repr``
    helper would emit doubled quotes and mangle a non-ASCII name into escape
    sequences.  This returns the name as written, clipped.

    Like :func:`_excerpt`, this is total and never raises: it builds error
    messages and must not be able to fail while reporting a failure.

    Args:
        name: The identifier to render.  Expected to be a ``str``, but a schema
            key need not be, so a non-string renders as its type.
        limit: Longest prefix to keep.

    Returns:
        *name* unchanged when it is within *limit*, otherwise its first *limit*
        characters followed by a marker reporting how many were dropped.
    """
    if not isinstance(name, str):
        return f"<{type(name).__name__}>"
    if len(name) <= limit:
        return name
    return f"{name[:limit]}... (+{len(name) - limit} more characters)"


def _clip_names(
    names: Sequence[Any], limit: int = _MAX_NAMES, quote: bool = False
) -> str:
    """Render a list of identifiers for an error message, bounded in size and count.

    Args:
        names: The identifiers to render.
        limit: How many names to show before summarising the remainder.
        quote: Whether to wrap each name in ``repr`` quotes, which the
            collision message does and the other messages do not.

    Returns:
        A comma-separated list of at most *limit* clipped names, a trailing
        ``"and N more"`` marker when *names* is longer, or ``"(none)"`` when
        *names* is empty.  The marker is never quoted even when *quote* is
        true: it is a statement *about* the list rather than an entry in it, so
        quoting it would render ``'... and 4 more'`` indistinguishable in kind
        from a real parameter name.
    """
    clipped = [_clip_name(name) for name in names[:limit]]
    overflow = len(names) - limit
    if not clipped and overflow <= 0:
        return "(none)"
    parts = [repr(name) if quote else name for name in clipped]
    if overflow > 0:
        parts.append(f"... and {overflow} more")
    return ", ".join(parts)


def _literal(node: ast.expr, label: str) -> Any:
    """Evaluate one argument node to a Python literal.

    Args:
        node: The AST node for a single positional or keyword argument.
        label: How to name *node*'s position in an error message.

    Returns:
        The literal value of *node*.

    Raises:
        MitoCodeBuiltinToolError: If *node* is not a Python literal, which
            covers arithmetic, name references, comprehensions, calls, and
            ``*`` splats.
    """
    try:
        return ast.literal_eval(node)
    except _LITERAL_FAILURES as exc:
        raise MitoCodeBuiltinToolError(
            f"Argument {label} is not a literal value; tool call arguments "
            "must be plain strings, numbers, booleans, lists or dicts "
            f"({type(exc).__name__})."
        ) from exc


def parse_call(text: str) -> ParsedCall:
    """Parse ``ToolName("arg", key=value)`` into its parts.

    Uses :func:`ast.parse` in ``mode="eval"`` rather than a regex: the mini-DSL
    nests quotes and escapes, which a regex mangles.  Only an ``ast.Call`` on a
    bare ``ast.Name`` with literal arguments is accepted, and every value goes
    through :func:`ast.literal_eval`, which cannot execute anything.

    Args:
        text: The raw DSL string emitted by the model.  Untrusted.

    Returns:
        A :class:`ParsedCall` whose arguments are real Python values, so
        ``timeout=300000`` is an ``int`` and not the string ``"300000"``.

    Raises:
        MitoCodeBuiltinToolError: If *text* cannot be parsed, is not a call,
            is not called on a bare name, carries a non-literal argument, or
            repeats a keyword argument.  The message names the offending input
            and always contains ``"not a tool call"`` for the shape rejections.
    """
    try:
        # Leading whitespace is stripped because `ast.parse` reads a leading
        # space as an indented statement block rather than an expression, so
        # `' Bash("ls")'` is an IndentationError while `'Bash("ls") '` already
        # works.  The asymmetry is an artefact of Python's grammar, not a
        # property of the DSL, and the text is model-generated: one stray space
        # should not cost the model its call.  Only outer whitespace is
        # removed, so an appended statement is still rejected afterwards --
        # stripping cannot turn `'Bash("ls"); evil()'` into anything valid.
        tree = ast.parse(
            text.strip() if isinstance(text, (str, bytes)) else text, mode="eval"
        )
    except _PARSE_FAILURES as exc:
        raise MitoCodeBuiltinToolError(
            f"{_excerpt(text)} is not a tool call: could not be parsed as a "
            f"ToolName(...) expression ({type(exc).__name__}: {exc})."
        ) from exc

    node = tree.body
    if not isinstance(node, ast.Call):
        raise MitoCodeBuiltinToolError(
            f"{_excerpt(text)} is not a tool call: expected a call of the form "
            'ToolName("argument"), for example Bash("ls -la").'
        )
    if not isinstance(node.func, ast.Name):
        raise MitoCodeBuiltinToolError(
            f"{_excerpt(text)} is not a tool call: the callee must be a bare "
            "tool name such as Bash(...); attribute and subscript calls are "
            "not tool calls."
        )

    args = [
        _literal(argument, f"at position {index}")
        for index, argument in enumerate(node.args)
    ]

    kwargs: Dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            # `Tool(**splat)` parses cleanly but names nothing; the splatted
            # value itself is a non-literal, so it cannot be recovered either.
            raise MitoCodeBuiltinToolError(
                f"{_excerpt(text)} is not a tool call: ** unpacking is not "
                "supported in a tool call; name each argument explicitly."
            )
        if keyword.arg in kwargs:
            # `ast.parse` does not run CPython's duplicate-keyword check --
            # `compile()` does, raising "keyword argument repeated" -- so this
            # text parses cleanly and a plain dict assignment would silently
            # discard the earlier value.  Refusing it here keeps the parser
            # aligned with the language the DSL borrows its syntax from.
            raise MitoCodeBuiltinToolError(
                f"{_excerpt(text)} is not a tool call: keyword argument "
                f"'{_clip_name(keyword.arg)}' is repeated; give each parameter "
                "exactly one value."
            )
        kwargs[keyword.arg] = _literal(
            keyword.value, f"for parameter '{_clip_name(keyword.arg)}'"
        )

    return ParsedCall(tool_name=node.func.id, args=args, kwargs=kwargs)


def bind_arguments(spec: ToolSpec, parsed: ParsedCall) -> Dict[str, Any]:
    """Map positional DSL arguments onto *spec*'s schema properties, in order.

    Property order comes from ``spec.parameters_schema["properties"]``, which
    for an annotation-derived schema is signature order, so ``Two("a", "b")``
    binds to ``first="a", second="b"``.  Positional arguments fill in that
    order first and explicit keywords are then overlaid on top, which lets a
    caller spell out a later argument while leaving an earlier one positional.
    A parameter supplied *both* ways is refused rather than resolved, because
    Python itself refuses it with ``TypeError: got multiple values for
    argument`` and silently keeping one of the two would drop a value the model
    believed it had passed.

    Whether a keyword collides with a positional is only knowable here and not
    in :func:`parse_call`, which has no spec: ``Tool("a", command="b")`` is a
    collision exactly when the first positional binds to ``command``.

    The call's own name is checked against *spec* before anything else happens.
    That is a consistency assertion, not an allowlist: it adds no knowledge of
    which tools exist -- the registry stays the single source of truth for that
    -- it only confirms that the spec this function was handed is the spec the
    call actually named.  Without it, a dispatcher that resolved the wrong spec
    would bind one tool's arguments to another's parameters silently.

    The schema's ``type`` is not consulted; see this module's docstring for why
    that widening is accepted.

    Args:
        spec: The registered tool the call is aimed at.
        parsed: The result of :func:`parse_call` for that tool.

    Returns:
        A dict of bound keyword arguments ready to be passed to the tool.

    Raises:
        MitoCodeBuiltinToolError: If the call names a different tool than
            *spec* does, has more positional arguments than the tool declares,
            names an unknown keyword, supplies one parameter both positionally
            and by keyword, or omits a required parameter.  Each message names
            the tool and lists the parameters it expected.
    """
    if parsed.tool_name != spec.name:
        raise MitoCodeBuiltinToolError(
            f"Tool call names '{_clip_name(parsed.tool_name)}' but this spec is "
            f"'{_clip_name(spec.name)}': refusing to bind arguments intended for "
            "one tool to another. Resolve the spec from the call's own name."
        )

    schema = spec.parameters_schema if isinstance(spec.parameters_schema, dict) else {}
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    expected = list(properties)
    required = schema.get("required")
    required = list(required) if isinstance(required, list) else []

    tool = _clip_name(spec.name)
    summary = f"Tool '{tool}' expects parameters: {_clip_names(expected)}."

    if len(parsed.args) > len(expected):
        raise MitoCodeBuiltinToolError(
            f"Too many positional arguments for tool '{tool}': "
            f"{len(parsed.args)} given but only {len(expected)} "
            f"{'parameter is' if len(expected) == 1 else 'parameters are'} "
            f"declared. {summary}"
        )

    # The collision check lives here rather than in `parse_call` because it
    # cannot live there: whether `Tool("a", command="b")` is a collision
    # depends on which parameter the first positional binds to, and only the
    # spec knows that.  Rejecting every mixture of positionals and keywords at
    # parse time would instead destroy the supported case of spelling out a
    # *later* argument while leaving an earlier one positional.
    positional_names = expected[: len(parsed.args)]
    repeated = [name for name in parsed.kwargs if name in positional_names]
    if repeated:
        raise MitoCodeBuiltinToolError(
            f"Got multiple values for parameter{'s' if len(repeated) > 1 else ''} "
            f"{_clip_names(repeated, quote=True)} for tool '{tool}': supplied both "
            "positionally and by keyword. Give each parameter exactly one value. "
            f"{summary}"
        )

    bound: Dict[str, Any] = dict(zip(expected, parsed.args))

    for name, value in parsed.kwargs.items():
        if name not in properties:
            raise MitoCodeBuiltinToolError(
                f"Unknown parameter '{_clip_name(name)}' for tool '{tool}'. {summary}"
            )
        bound[name] = value

    missing = [name for name in required if name not in bound]
    if missing:
        raise MitoCodeBuiltinToolError(
            f"Missing required parameter{'s' if len(missing) > 1 else ''} "
            f"for tool '{tool}': {_clip_names(missing)}. {summary}"
        )

    return bound
