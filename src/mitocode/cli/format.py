"""Pure formatting helpers for the MitoCode command line interface.

Nothing in this module touches a terminal, the filesystem, or global state:
each function turns values into a string or a :class:`rich.text.Text`, which
makes every rule directly testable through ``Text.plain``. The live UI layers
(``stream``, ``ui``, ``pump``) only add the ``●``/``⎿`` gutter and styles on
top of the plain shapes produced here.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Mapping

from rich.text import Text

PREVIEW_MAX_LINES = 8
"""Default number of result lines shown when a tool result is previewed."""

PREVIEW_MAX_CHARS = 1200
"""Default character budget for a previewed tool result."""

DIFF_PREVIEW_MAX_LINES = 12
"""Default number of unified-diff lines shown for an ``Edit`` call."""

CONTINUATION_INDENT = "    "
"""Indent prefixed to every line after the first in a tool result preview."""

ERROR_PREFIX = "error:"
"""Case-insensitive prefix that marks a tool result as a failure."""

MCP_TOOL_PREFIX = "mcp__"
"""Prefix the MCP registry puts in front of ``<server>__<tool>`` names."""

_PATH_PARAM_NAMES = frozenset({"file_path", "path", "source", "destination"})
"""Argument keys whose values are filesystem paths and must be relativised."""

_PARAM_BY_TOOL: Mapping[str, str] = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
    "Bash": "command",
    "Agent": "description",
    "QueryAgent": "agent_id",
    "Background": "call",
    "BackgroundResult": "handle",
}
"""Preferred argument key per built-in tool, matching its display metadata."""

_INLINE_PATTERN = re.compile(
    r"`(?P<code>[^`\n]+)`"
    r"|\*\*(?P<bold>(?!\s)[^\n]*?(?<=\S))\*\*"
    r"|\[(?P<label>[^\]\n]+)\]\((?P<url>[^)\s]*)\)"
    r"|(?<![\w*])\*(?P<italic>(?!\s)[^*\n]+?(?<=\S))\*(?![\w*])"
)
"""Inline markdown constructs, ordered so code wins over emphasis."""


def display_path(raw: str, root: Path) -> str:
    """Show a filesystem path relative to the workspace root when possible.

    ``~`` is expanded before the comparison, because the CLI reports real
    locations while the user typed a shorthand. Paths that resolve to the root
    itself become ``"."``; paths outside the workspace are returned as the
    expanded absolute path, since pretending they are relative would be
    misleading.

    Args:
        raw: The path as supplied by the model or the user.
        root: Workspace root the display should be relative to.

    Returns:
        A workspace-relative path, the unchanged relative path, or the expanded
        absolute path when the target lives outside *root*.
    """
    if not raw:
        return ""

    expanded = Path(str(raw)).expanduser()
    if not expanded.is_absolute():
        return str(expanded)

    try:
        target = expanded.resolve()
        base = Path(root).expanduser().resolve()
    except (OSError, RuntimeError):
        return str(expanded)

    if target == base:
        return "."
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(expanded)


def _string(value: Any) -> str:
    """Coerce an argument value to display text without raising.

    Args:
        value: Any value taken from a tool call's arguments.

    Returns:
        The value itself for strings, ``""`` for ``None``, and ``str(value)``
        for everything else.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # pylint: disable=broad-except
        return ""


def _is_path_key(key: str) -> bool:
    """Return whether an argument key holds a path that should be relativised.

    Args:
        key: Argument name from a tool call.

    Returns:
        ``True`` for known path keys and for names ending in ``_path``.
    """
    return key in _PATH_PARAM_NAMES or key.endswith("_path")


def _parameter_key(name: str, args: Mapping[str, Any], display_param: str) -> str:
    """Choose which argument should be shown next to a tool name.

    The built-in tools each name a preferred argument explicitly; MCP tools and
    unknown tools fall back to the registry's ``display_param`` and then to the
    first non-empty string argument, so no tool ever needs raw JSON on screen.

    Args:
        name: Registered tool name, possibly ``mcp__<server>__<tool>``.
        args: Arguments supplied by the model.
        display_param: Parameter name advertised by the tool registry.

    Returns:
        The argument key to display, or ``""`` when the call takes no visible
        argument.
    """
    preferred = _PARAM_BY_TOOL.get(name, "")
    if preferred and preferred in args:
        return preferred
    if display_param and display_param in args:
        return display_param
    for key, value in args.items():
        if isinstance(value, str) and value.strip():
            return key
    return preferred or display_param


def param_for(
    name: str,
    args: Mapping[str, Any],
    display_param: str,
    root: Path,
) -> str:
    """Return the plain-text parameter shown for a tool call.

    Multi-line values collapse to their first line, ``Glob``/``Grep`` append
    their search directory, and path-like values are shown relative to the
    workspace root. Commands are deliberately kept verbatim: rewriting an
    absolute executable path inside ``Bash`` would change the command.

    Args:
        name: Registered tool name.
        args: Arguments supplied by the model.
        display_param: Parameter name advertised by the tool registry.
        root: Workspace root used to relativise paths.

    Returns:
        The parameter as one line of plain text, or ``""`` when there is none.
    """
    key = _parameter_key(name, args, display_param)
    if not key:
        return ""

    value = _string(args.get(key))
    if value and key == "description":
        value = value.splitlines()[0] if value.splitlines() else ""

    if name in ("Glob", "Grep"):
        search_root = _string(args.get("path"))
        if search_root:
            return f"{value} in {display_path(search_root, root)}"
        return value

    if _is_path_key(key):
        return display_path(value, root)
    return value


def _display_name(name: str) -> str:
    """Convert an MCP registry name into the short form shown in the UI.

    Args:
        name: Registered tool name, possibly ``mcp__<server>__<tool>``.

    Returns:
        ``<server>/<tool>`` for MCP tools and *name* for everything else.
    """
    if not name.startswith(MCP_TOOL_PREFIX):
        return name
    parts = name.split("__", 2)
    if len(parts) == 3 and parts[1] and parts[2]:
        return f"{parts[1]}/{parts[2]}"
    return name


def format_tool_call(
    name: str,
    args: Mapping[str, Any],
    display_param: str,
    root: Path,
) -> Text:
    """Render the header line of a tool call, without the ``●`` gutter.

    Args:
        name: Registered tool name.
        args: Arguments supplied by the model.
        display_param: Parameter name advertised by the tool registry.
        root: Workspace root used to relativise paths.

    Returns:
        A styled text whose plain form is ``"<Name>  <param>"`` or just
        ``"<Name>"`` when the call takes no visible argument.
    """
    text = Text()
    text.append(_display_name(name), style="bold mito.accent")
    parameter = param_for(name, args, display_param, root)
    if parameter:
        text.append("  ")
        text.append(parameter, style="mito.code")
    return text


def _link_style(url: str) -> str:
    """Return the style for a markdown link label.

    Args:
        url: Destination from ``[label](url)``.

    Returns:
        An accent style, plus a terminal hyperlink for http(s) URLs so the label
        remains actionable in supporting terminals.
    """
    if url.startswith(("http://", "https://")):
        return f"link {url} mito.accent underline"
    return "mito.accent underline"


def inline_markdown(text: str) -> Text:
    """Convert a single line of markdown into styled text.

    Supports inline code, bold, italic and ``[label](url)`` links. Anything the
    pattern does not recognise -- including malformed brackets, unmatched
    markers and arithmetic asterisks -- is preserved literally. The function is
    total: it never raises, so a model streaming broken markdown cannot crash a
    turn.

    Args:
        text: One line of assistant or tool text.

    Returns:
        Styled text whose plain form has the markdown markers removed.
    """
    if not text:
        return Text()

    result = Text()
    position = 0
    try:
        for match in _INLINE_PATTERN.finditer(text):
            if match.start() > position:
                result.append(text[position : match.start()])
            code = match.group("code")
            bold = match.group("bold")
            label = match.group("label")
            italic = match.group("italic")
            if code is not None:
                result.append(code, style="mito.code")
            elif bold is not None:
                result.append(bold, style="bold")
            elif label is not None:
                result.append(label, style=_link_style(match.group("url") or ""))
            elif italic is not None:
                result.append(italic, style="italic")
            position = match.end()
        if position < len(text):
            result.append(text[position:])
    except Exception:  # pylint: disable=broad-except
        # Total by contract: fall back to the literal line rather than let any
        # unforeseen expression escape into the streaming renderer.
        return Text(text)
    return result


def truncate(
    text: str,
    *,
    max_lines: int = PREVIEW_MAX_LINES,
    max_chars: int = PREVIEW_MAX_CHARS,
) -> str:
    """Shorten a block of text to a preview budget.

    Lines beyond *max_lines* are dropped and, if the kept body is still longer
    than *max_chars*, it is cut at that character boundary. Any truncation ends
    with a marker so the model's full output is visibly not shown.

    Args:
        text: Full text to shorten.
        max_lines: Maximum number of content lines kept.
        max_chars: Maximum number of characters kept in those lines.

    Returns:
        The text unchanged when it fits, otherwise the shortened body followed
        by ``"… +N lines"`` (or ``"…"`` for a pure character cut).
    """
    if not text:
        return ""

    line_limit = max(0, max_lines)
    char_limit = max(0, max_chars)
    lines = text.splitlines()
    if len(lines) <= line_limit and len(text) <= char_limit:
        return text

    body = "\n".join(lines[:line_limit])
    if len(body) > char_limit:
        body = body[:char_limit].rstrip()

    kept_lines = len(body.splitlines()) if body else 0
    dropped_lines = max(0, len(lines) - kept_lines)
    marker = f"… +{dropped_lines} lines" if dropped_lines else "…"
    return f"{body}\n{marker}" if body else marker


def format_elapsed(seconds: float) -> str:
    """Format a duration the way the status line shows it.

    Args:
        seconds: Elapsed seconds; negative values are treated as zero.

    Returns:
        ``"12s"`` under a minute, ``"1m 02s"`` under an hour, and
        ``"1h 01m"`` beyond that.
    """
    total = int(seconds) if seconds > 0 else 0
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, remaining = divmod(total, 60)
        return f"{minutes}m {remaining:02d}s"
    hours, remaining = divmod(total, 3600)
    return f"{hours}h {remaining // 60:02d}m"


def estimate_tokens(text_chars: int) -> int:
    """Estimate tokens from a character count.

    Providers do not return usage for streamed responses, so the CLI reports an
    explicit approximation of four characters per token.

    Args:
        text_chars: Number of characters seen in the session.

    Returns:
        ``0`` for no text, otherwise at least one token.
    """
    if text_chars <= 0:
        return 0
    return max(1, text_chars // 4)


def _trim_decimal(value: float) -> str:
    """Format a scaled token count with at most one decimal place.

    Args:
        value: Token count already divided by its unit.

    Returns:
        ``"1"`` for ``1.0`` and ``"1.8"`` for ``1.75``-style values.
    """
    formatted = f"{value:.1f}"
    return formatted[:-2] if formatted.endswith(".0") else formatted


def format_tokens(tokens: int) -> str:
    """Format an approximate token count for the status line.

    Args:
        tokens: Estimated token count; negatives are treated as zero.

    Returns:
        ``"~950"`` below one thousand, ``"~1.8k"`` below a million, and a
        megabyte-scale suffix beyond that, always with the approximation mark.
    """
    value = max(0, tokens)
    if value < 1000:
        return f"~{value}"
    if value < 1_000_000:
        return f"~{_trim_decimal(value / 1000)}k"
    return f"~{_trim_decimal(value / 1_000_000)}M"


def _indented_text(body: str, style: str | None = None) -> Text:
    """Build text whose continuation lines carry the preview indentation.

    Args:
        body: Plain text, already truncated to the desired budget.
        style: Optional style applied to every line.

    Returns:
        Styled text; the first line has no indent and later lines start with
        :data:`CONTINUATION_INDENT`.
    """
    text = Text(style=style) if style else Text()
    for index, line in enumerate(body.splitlines()):
        if index:
            text.append("\n")
            text.append(CONTINUATION_INDENT)
        text.append(line)
    return text


def _is_diff_header(line: str) -> bool:
    """Return whether a unified-diff line is the ``---``/``+++`` file header.

    Args:
        line: One line of a unified diff.

    Returns:
        ``True`` for the two header lines, which carry no useful information
        here because both sides are the same file.
    """
    return line.startswith("--- ") or line.startswith("+++ ")


def _diff_counts(diff_lines: list[str]) -> tuple[int, int]:
    """Count added and removed lines in a unified diff.

    Args:
        diff_lines: Lines produced by :func:`difflib.unified_diff`.

    Returns:
        A ``(added, removed)`` pair, excluding the file headers.
    """
    added = sum(
        1 for line in diff_lines if line.startswith("+") and not line.startswith("+++ ")
    )
    removed = sum(
        1 for line in diff_lines if line.startswith("-") and not line.startswith("--- ")
    )
    return added, removed


def _format_edit_result(args: Mapping[str, Any], root: Path, *, verbose: bool) -> Text:
    """Render the ``Updated <path> (+A −B)`` summary and its unified diff.

    Args:
        args: ``Edit`` arguments, including ``old_string``/``new_string``.
        root: Workspace root used to relativise the path.
        verbose: Show the whole diff instead of the first
            :data:`DIFF_PREVIEW_MAX_LINES` lines.

    Returns:
        Styled text whose plain form starts with the summary line and continues
        with four-space-indented diff lines.
    """
    raw_path = _string(args.get("file_path"))
    shown_path = display_path(raw_path, root) if raw_path else "(unknown)"
    old = _string(args.get("old_string"))
    new = _string(args.get("new_string"))

    diff_lines = list(
        difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2)
    )
    added, removed = _diff_counts(diff_lines)
    body = [line for line in diff_lines if not _is_diff_header(line)]

    shown = body if verbose else body[:DIFF_PREVIEW_MAX_LINES]
    dropped = len(body) - len(shown)

    text = Text()
    text.append(f"Updated {shown_path} (+{added} −{removed})", style="bold mito.accent")
    for line in shown:
        text.append("\n")
        text.append(CONTINUATION_INDENT)
        if line.startswith("+") and not line.startswith("+++ "):
            text.append(line, style="mito.diff.add")
        elif line.startswith("-") and not line.startswith("--- "):
            text.append(line, style="mito.diff.remove")
        elif line.startswith("@@"):
            text.append(line, style="mito.muted")
        else:
            text.append(line)
    if dropped:
        text.append("\n")
        text.append(CONTINUATION_INDENT)
        text.append(f"… +{dropped} lines", style="mito.muted")
    return text


def _write_result(args: Mapping[str, Any], result: str, *, verbose: bool) -> Text:
    """Render a ``Write`` result as its line and character counts.

    Args:
        args: ``Write`` arguments, including the file ``content``.
        result: Raw tool output, used only when the content is absent.
        verbose: When the content is absent, show the full output instead of a
            preview.

    Returns:
        ``"Wrote N lines (M chars)"``, or a preview when no content was supplied.
    """
    if "content" not in args:
        return _preview_text(result, verbose=verbose)
    content = _string(args.get("content"))
    return Text(f"Wrote {len(content.splitlines())} lines ({len(content)} chars)")


def _read_result(result: str) -> Text:
    """Render a ``Read`` result as a line count.

    Args:
        result: Raw tool output, one numbered line per source line.

    Returns:
        ``"N lines"``.
    """
    return Text(f"{len(result.splitlines())} lines")


def _preview_text(result: str, *, verbose: bool) -> Text:
    """Render a generic tool result as an indented, truncated preview.

    Args:
        result: Raw tool output.
        verbose: Keep every line and character instead of truncating.

    Returns:
        Plain preview text with four-space-indented continuation lines.
    """
    body = result.rstrip("\n")
    if not body.strip():
        return Text()
    return _indented_text(body if verbose else truncate(body))


def format_result(
    name: str,
    args: Mapping[str, Any],
    result: str,
    *,
    verbose: bool,
    root: Path,
) -> Text:
    """Render a tool result preview, without the ``⎿`` gutter.

    ``Edit`` summaries come from the call arguments because its tool output is
    only a confirmation string; ``Write`` and ``Read`` collapse to counts;
    ``Bash`` and every other tool show their output. An ``Error:`` result is
    preserved verbatim and styled as a failure so the user can act on it.

    Args:
        name: Registered tool name.
        args: Arguments supplied by the model for the call.
        result: Raw tool output.
        verbose: Disable truncation so the whole output is shown.
        root: Workspace root used to relativise paths.

    Returns:
        Styled text whose plain form is the preview the caller prints after the
        result gutter. Continuation lines start with four spaces.
    """
    raw = "" if result is None else str(result)
    stripped = raw.strip()

    if stripped.lower().startswith(ERROR_PREFIX):
        body = stripped if verbose else truncate(stripped)
        return _indented_text(body, style="mito.error")

    if name == "Edit":
        return _format_edit_result(args, root, verbose=verbose)
    if name == "Write":
        return _write_result(args, raw, verbose=verbose)
    if name == "Read":
        return _read_result(raw)
    return _preview_text(raw, verbose=verbose)


__all__ = [
    "CONTINUATION_INDENT",
    "DIFF_PREVIEW_MAX_LINES",
    "ERROR_PREFIX",
    "MCP_TOOL_PREFIX",
    "PREVIEW_MAX_CHARS",
    "PREVIEW_MAX_LINES",
    "display_path",
    "estimate_tokens",
    "format_elapsed",
    "format_result",
    "format_tokens",
    "format_tool_call",
    "inline_markdown",
    "param_for",
    "truncate",
]
