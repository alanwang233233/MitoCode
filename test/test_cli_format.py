"""Tests for the CLI theme and the pure formatting helpers.

Every assertion goes through ``Text.plain`` so the results do not depend on a
terminal, on colour support, or on Rich's style resolution.
"""

import os
import sys
from pathlib import Path

from rich.text import Text
from rich.theme import Theme

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.format import (
    display_path,
    estimate_tokens,
    format_elapsed,
    format_result,
    format_tokens,
    format_tool_call,
    inline_markdown,
    param_for,
    truncate,
)
from mitocode.cli.theme import MITOCODE_THEME, THEME_STYLE_NAMES


def _plain(renderable) -> str:
    """Return the unstyled text of a formatter result.

    Args:
        renderable: A ``Text`` instance or a plain string.

    Returns:
        The string a user would read, without styles.
    """
    return renderable.plain if isinstance(renderable, Text) else str(renderable)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def test_theme_is_a_rich_theme_with_semantic_names():
    assert isinstance(MITOCODE_THEME, Theme)
    expected = {
        "mito.accent",
        "mito.success",
        "mito.error",
        "mito.warn",
        "mito.muted",
        "mito.code",
        "mito.diff.add",
        "mito.diff.remove",
    }
    assert expected == set(THEME_STYLE_NAMES)
    for name in expected:
        assert name in MITOCODE_THEME.styles


# ---------------------------------------------------------------------------
# display_path
# ---------------------------------------------------------------------------


def test_display_path_inside_root_is_relative(tmp_path):
    target = tmp_path / "src" / "a.py"
    assert display_path(str(target), tmp_path) == "src/a.py"


def test_display_path_root_itself_is_a_dot(tmp_path):
    assert display_path(str(tmp_path), tmp_path) == "."


def test_display_path_outside_root_is_unchanged(tmp_path):
    assert display_path("/etc/hosts", tmp_path) == "/etc/hosts"


def test_display_path_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert display_path("~/src/a.py", tmp_path) == "src/a.py"


def test_display_path_keeps_relative_paths(tmp_path):
    assert display_path("src/a.py", tmp_path) == "src/a.py"


def test_display_path_empty_is_empty(tmp_path):
    assert display_path("", tmp_path) == ""


# ---------------------------------------------------------------------------
# format_tool_call / param_for
# ---------------------------------------------------------------------------


def test_tool_call_line_for_read_and_mcp(tmp_path):
    assert (
        _plain(
            format_tool_call(
                "Read", {"file_path": str(tmp_path / "a.py")}, "file_path", tmp_path
            )
        )
        == "Read  a.py"
    )
    assert _plain(format_tool_call("mcp__gh__create", {}, "", tmp_path)) == "gh/create"


def test_tool_call_glob_appends_search_directory(tmp_path):
    plain = _plain(
        format_tool_call(
            "Glob",
            {"pattern": "*.py", "path": str(tmp_path / "src")},
            "pattern",
            tmp_path,
        )
    )
    assert plain == "Glob  *.py in src"


def test_tool_call_bash_keeps_command_verbatim(tmp_path):
    command = f"{tmp_path}/run.sh --flag"
    plain = _plain(format_tool_call("Bash", {"command": command}, "command", tmp_path))
    assert plain == f"Bash  {command}"


def test_tool_call_agent_uses_first_description_line(tmp_path):
    args = {"description": "Fix the parser\nand more detail"}
    assert _plain(format_tool_call("Agent", args, "description", tmp_path)) == (
        "Agent  Fix the parser"
    )


def test_tool_call_named_parameters(tmp_path):
    assert (
        _plain(
            format_tool_call("QueryAgent", {"agent_id": "abc"}, "agent_id", tmp_path)
        )
        == "QueryAgent  abc"
    )
    assert (
        _plain(format_tool_call("Background", {"call": "Read: a.py"}, "call", tmp_path))
        == "Background  Read: a.py"
    )
    assert (
        _plain(
            format_tool_call("BackgroundResult", {"handle": "h1"}, "handle", tmp_path)
        )
        == "BackgroundResult  h1"
    )


def test_tool_call_falls_back_to_first_string_argument(tmp_path):
    plain = _plain(
        format_tool_call(
            "ListAgent", {"count": 2, "query": "main"}, "missing_param", tmp_path
        )
    )
    assert plain == "ListAgent  main"


def test_tool_call_without_parameter_has_no_trailing_space(tmp_path):
    assert _plain(format_tool_call("ListAgent", {}, "", tmp_path)) == "ListAgent"


def test_mcp_tool_call_shows_argument(tmp_path):
    plain = _plain(format_tool_call("mcp__srv__tool", {"query": "x"}, "", tmp_path))
    assert plain == "srv/tool  x"


def test_param_for_returns_plain_string(tmp_path):
    args = {"file_path": str(tmp_path / "a.py")}
    assert param_for("Write", args, "file_path", tmp_path) == "a.py"
    assert param_for("ListAgent", {}, "", tmp_path) == ""


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------


def test_edit_result_shows_counts_and_diff(tmp_path):
    text = format_result(
        "Edit",
        {"file_path": "a.py", "old_string": "x = 1\n", "new_string": "x = 2\n"},
        "Updated",
        verbose=False,
        root=tmp_path,
    )
    plain = _plain(text)
    assert "Updated a.py" in plain and "+1" in plain and "−1" in plain
    assert "-x = 1" in plain and "+x = 2" in plain


def test_edit_result_indents_continuation_lines(tmp_path):
    plain = _plain(
        format_result(
            "Edit",
            {
                "file_path": "a.py",
                "old_string": "one\ntwo\nthree",
                "new_string": "one\nTWO\nthree",
            },
            "",
            verbose=False,
            root=tmp_path,
        )
    )
    lines = plain.splitlines()
    assert lines[0].startswith("Updated a.py")
    for line in lines[1:]:
        assert line.startswith("    ")


def test_edit_result_truncates_diff_by_default_and_not_when_verbose(tmp_path):
    old = "\n".join(f"old {i}" for i in range(30))
    new = "\n".join(f"new {i}" for i in range(30))
    args = {"file_path": "a.py", "old_string": old, "new_string": new}

    short = _plain(format_result("Edit", args, "", verbose=False, root=tmp_path))
    full = _plain(format_result("Edit", args, "", verbose=True, root=tmp_path))

    assert short.splitlines()[-1].startswith("    … +")
    assert short.splitlines()[-1].endswith(" lines")
    assert "… +" not in full
    assert len(short.splitlines()) < len(full.splitlines())


def test_write_result_reports_lines_and_chars(tmp_path):
    plain = _plain(
        format_result(
            "Write",
            {"file_path": "a.py", "content": "one\ntwo\n"},
            "",
            verbose=False,
            root=tmp_path,
        )
    )
    assert plain == "Wrote 2 lines (8 chars)"


def test_read_result_reports_line_count(tmp_path):
    plain = _plain(
        format_result(
            "Read",
            {"file_path": "a.py"},
            "1\tone\n2\ttwo\n3\tthree",
            verbose=False,
            root=tmp_path,
        )
    )
    assert plain == "3 lines"


def test_bash_result_is_output_preview(tmp_path):
    plain = _plain(
        format_result(
            "Bash",
            {"command": "pytest"},
            "214 passed in 8.1s",
            verbose=False,
            root=tmp_path,
        )
    )
    assert plain == "214 passed in 8.1s"


def test_error_result_is_preserved_and_styled(tmp_path):
    message = "Error: read failed: a.py does not exist."
    text = format_result(
        "Read", {"file_path": "a.py"}, message, verbose=False, root=tmp_path
    )
    assert _plain(text) == message
    assert str(text.style) == "mito.error"


def test_generic_result_is_truncated_and_indented(tmp_path):
    result = "\n".join(f"match {i}" for i in range(20))
    plain = _plain(
        format_result("Grep", {"pattern": "x"}, result, verbose=False, root=tmp_path)
    )
    lines = plain.splitlines()
    assert lines[0] == "match 0"
    assert plain.endswith("… +12 lines")
    for line in lines[1:-1]:
        assert line.startswith("    ")

    full = _plain(
        format_result("Grep", {"pattern": "x"}, result, verbose=True, root=tmp_path)
    )
    assert "… +" not in full
    assert full.splitlines()[-1] == "    match 19"


def test_result_never_contains_the_result_marker(tmp_path):
    for name, args, result in [
        ("Read", {}, "1\tone"),
        ("Bash", {"command": "ls"}, "out"),
        ("Grep", {"pattern": "x"}, "hit"),
    ]:
        assert "⎿" not in _plain(
            format_result(name, args, result, verbose=False, root=tmp_path)
        )


def test_empty_result_has_no_trailing_whitespace(tmp_path):
    plain = _plain(
        format_result("Bash", {"command": "true"}, "", verbose=False, root=tmp_path)
    )
    assert plain == plain.strip()


def test_write_result_without_content_falls_back_to_output(tmp_path):
    args = {"file_path": "a.py"}
    assert (
        _plain(
            format_result("Write", args, "Wrote 4 chars", verbose=False, root=tmp_path)
        )
        == "Wrote 4 chars"
    )
    assert _plain(format_result("Write", args, "", verbose=True, root=tmp_path)) == ""


def test_write_result_for_empty_content(tmp_path):
    plain = _plain(
        format_result(
            "Write",
            {"file_path": "a.py", "content": ""},
            "",
            verbose=False,
            root=tmp_path,
        )
    )
    assert plain == "Wrote 0 lines (0 chars)"


def test_param_for_unknown_tool_relative_suffix_path(tmp_path):
    args = {"image_path": str(tmp_path / "img.png")}
    assert param_for("CustomTool", args, "", tmp_path) == "img.png"


def test_mcp_name_without_tool_part_stays_intact(tmp_path):
    assert _plain(format_tool_call("mcp__srv", {}, "", tmp_path)) == "mcp__srv"
    assert _plain(format_tool_call("mcp__srv__", {}, "", tmp_path)) == "mcp__srv__"


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


def test_truncate_keeps_marker():
    text = "\n".join(str(i) for i in range(20))
    out = truncate(text)
    assert out.endswith("… +12 lines")


def test_truncate_short_text_is_unchanged():
    assert truncate("a\nb") == "a\nb"


def test_truncate_exactly_max_lines_is_unchanged():
    text = "\n".join(str(i) for i in range(8))
    assert truncate(text) == text


def test_truncate_respects_character_budget():
    out = truncate("x" * 5000, max_chars=100)
    assert out.startswith("x" * 100)
    assert out.endswith("…")


# ---------------------------------------------------------------------------
# inline_markdown
# ---------------------------------------------------------------------------


def test_inline_markdown_never_raises_on_brackets():
    assert "[" in _plain(inline_markdown("see [x]("))
    assert _plain(inline_markdown("**a** and `b`")) == "a and b"


def test_inline_markdown_strips_markers():
    assert _plain(inline_markdown("*italic*")) == "italic"
    assert _plain(inline_markdown("[docs](https://example.com)")) == "docs"
    assert _plain(inline_markdown("**b** *i* `c` [l](u)")) == "b i c l"


def test_inline_markdown_keeps_code_contents_literal():
    assert _plain(inline_markdown("`a**b**`")) == "a**b**"


def test_inline_markdown_keeps_unmatched_markers_literal():
    for raw in ["plain [bracket]", "3 * 4 * 5", "**", "*", "`", "[", "***", "[](x)"]:
        assert _plain(inline_markdown(raw)) == raw


def test_inline_markdown_never_raises_on_malformed_input():
    samples = [
        "[x](",
        "**a*",
        "*a**",
        "`a",
        "a [b](c d)",
        "[[[]]]((()))",
        "**`bold code`**",
        "*`code italic`*",
    ]
    for raw in samples:
        Text.assemble(inline_markdown(raw))


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def test_format_elapsed():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(12) == "12s"
    assert format_elapsed(12.9) == "12s"
    assert format_elapsed(62) == "1m 02s"
    assert format_elapsed(605) == "10m 05s"
    assert format_elapsed(3661) == "1h 01m"
    assert format_elapsed(-5) == "0s"


def test_estimate_tokens():
    assert estimate_tokens(0) == 0
    assert estimate_tokens(1) == 1
    assert estimate_tokens(3) == 1
    assert estimate_tokens(4) == 1
    assert estimate_tokens(4000) == 1000
    assert estimate_tokens(-10) == 0


def test_format_tokens():
    assert format_tokens(0) == "~0"
    assert format_tokens(950) == "~950"
    assert format_tokens(1000) == "~1k"
    assert format_tokens(1800) == "~1.8k"
    assert format_tokens(12500) == "~12.5k"
