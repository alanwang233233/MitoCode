"""Tests for the line-buffered streaming renderer.

The console writes into a ``StringIO`` with colour disabled, so every assertion
checks exactly the text a user would read. Style decisions are covered by the
theme tests in ``test_cli_format.py``.
"""

import os
import sys
from io import StringIO

from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.stream import StreamingRenderer
from mitocode.cli.theme import MITOCODE_THEME


def _renderer() -> tuple[StreamingRenderer, StringIO]:
    """Build a renderer that writes to an in-memory console.

    Returns:
        The renderer and the buffer holding everything it has printed.
    """
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=100,
        force_terminal=False,
        theme=MITOCODE_THEME,
    )
    return StreamingRenderer(console), buffer


def test_streams_prose_by_line_and_flushes_partial():
    stream, buffer = _renderer()
    stream.feed("Hello **world**\nsecond")
    assert buffer.getvalue() == "Hello world\n"
    stream.flush()
    assert buffer.getvalue().endswith("second\n")


def test_complete_lines_print_before_turn_ends():
    stream, buffer = _renderer()
    stream.feed("one\ntwo\n")
    assert buffer.getvalue() == "one\ntwo\n"


def test_flush_without_partial_line_writes_nothing():
    stream, buffer = _renderer()
    stream.feed("done\n")
    assert buffer.getvalue() == "done\n"
    stream.flush()
    assert buffer.getvalue() == "done\n"


def test_delta_split_across_calls_still_emits_one_line():
    stream, buffer = _renderer()
    stream.feed("Hel")
    stream.feed("lo\n")
    assert buffer.getvalue() == "Hello\n"


def test_code_fence_is_not_markdown():
    stream, buffer = _renderer()
    stream.feed("```python\nx = **1**\n```\n")
    text = buffer.getvalue()
    assert "x = **1**" in text


def test_fence_language_is_shown_once():
    stream, buffer = _renderer()
    stream.feed("```python\nx = 1\n")
    assert buffer.getvalue() == "python\nx = 1\n"


def test_plain_fence_prints_no_language_marker():
    stream, buffer = _renderer()
    stream.feed("```\nx = 1\n")
    assert buffer.getvalue() == "x = 1\n"


def test_inline_markdown_is_applied_outside_fence():
    stream, buffer = _renderer()
    stream.feed("- Read `format.py` and *check* [docs](https://example.com)\n")
    assert buffer.getvalue() == "- Read format.py and check docs\n"


def test_closing_fence_restores_inline_markdown():
    stream, buffer = _renderer()
    stream.feed("```py\n**raw**\n```\n**bold**\n")
    assert buffer.getvalue().endswith("bold\n")
    assert "**raw**" in buffer.getvalue()


def test_repeated_blank_lines_print_once():
    stream, buffer = _renderer()
    stream.feed("one\n\n\n\ntwo\n")
    assert buffer.getvalue() == "one\n\ntwo\n"


def test_heading_markers_are_hidden():
    stream, buffer = _renderer()
    stream.feed("# Heading **now**\n")
    assert buffer.getvalue() == "Heading now\n"


def test_flush_prints_partial_code_line_without_markdown():
    stream, buffer = _renderer()
    stream.feed("```\nx = *1*")
    stream.flush()
    assert buffer.getvalue() == "x = *1*\n"
