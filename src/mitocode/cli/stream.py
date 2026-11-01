"""Line-buffered renderer for assistant text streamed into the CLI.

The model arrives in arbitrary pieces, so a renderer cannot treat each delta as
a complete markdown document. :class:`StreamingRenderer` instead buffers until
it has full lines, prints each line once, and switches to a raw code mode inside
```` ``` ```` fences. ``flush()`` drains a trailing partial line at the end of a
turn, which keeps the live transcript lossless without ever re-rendering text
that was already shown.

Nothing here reads global state or touches the terminal directly: the renderer
only calls the :class:`rich.console.Console` it is given, which makes it usable
both by the live UI and by tests writing to a ``StringIO``.
"""

from __future__ import annotations

import re
from typing import Optional

from rich.console import Console
from rich.text import Text

from mitocode.cli.format import inline_markdown

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
"""Markdown ATX heading: up to six ``#`` markers followed by a space."""

_LIST_PATTERN = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$")
"""Bullet or ordered-list marker followed by the item body."""

_QUOTE_PATTERN = re.compile(r"^(\s*>\s?)(.*)$")
"""Blockquote marker followed by the quoted body."""

_FENCE_PREFIX = "```"
"""Opening and closing marker for a fenced code block."""


class StreamingRenderer:
    """Render streamed assistant text one complete line at a time.

    A single instance lives for one turn (or one session); callers pass each
    stream delta to :meth:`feed` and call :meth:`flush` when the turn ends. Text
    outside code fences goes through
    :func:`mitocode.cli.format.inline_markdown`; text inside a fence is emitted
    verbatim with the ``mito.code`` style so source code is never reformatted.

    Attributes:
        _console: Console every line is printed to.
        _buffer: Text received since the last newline.
        _fence: Language of the open code fence, or ``None`` outside a fence.
        _last_blank: Whether the previous emitted line was blank, used to
            collapse runs of empty lines.
    """

    def __init__(self, console: Console) -> None:
        """Initialise an empty renderer.

        Args:
            console: Console that receives the rendered lines.
        """
        self._console = console
        self._buffer = ""
        self._fence: Optional[str] = None
        self._last_blank = False

    def feed(self, delta: str) -> None:
        """Consume a chunk of streamed text.

        Only complete newline-terminated lines are rendered; the fragment after
        the final newline stays buffered until more text arrives or the caller
        invokes :meth:`flush`.

        Args:
            delta: A piece of assistant text as delivered by the model stream.
        """
        if not delta:
            return

        self._buffer += delta
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._emit_line(line)

    def flush(self) -> None:
        """Render and clear any buffered partial line.

        Called at the end of a turn. When the buffer is empty this is a no-op,
        so flushing repeatedly cannot print stray blank lines.
        """
        if not self._buffer:
            return
        line, self._buffer = self._buffer, ""
        self._emit_line(line)

    def _emit_line(self, line: str) -> None:
        """Render one line, honouring code-fence and markdown state.

        Args:
            line: A line without its trailing newline.
        """
        line = line.rstrip("\r")

        if line.startswith(_FENCE_PREFIX):
            self._toggle_fence(line)
            return

        if self._fence is not None:
            self._print_code(line)
            return

        if not line.strip():
            self._print_blank()
            return

        self._last_blank = False
        self._print(self._style_prose(line))

    def _toggle_fence(self, line: str) -> None:
        """Open or close a fenced code block.

        An opening fence prints its language marker in the muted style; an empty
        info string prints nothing. A closing fence prints nothing either.

        Args:
            line: The fence line, including its leading backticks.
        """
        if self._fence is not None:
            self._fence = None
            return

        language = line[len(_FENCE_PREFIX) :].strip()
        self._fence = language
        self._last_blank = False
        if language:
            self._print(Text(language, style="mito.muted"))

    def _style_prose(self, line: str) -> Text:
        """Apply line-level markdown rules to a non-code line.

        Headings lose their ``#`` markers and become accent-bold, list markers
        become accent, and blockquotes become muted. The remainder always goes
        through the inline formatter, so ``**bold**`` and friends keep working
        inside headings and list items.

        Args:
            line: A non-empty line outside a code fence.

        Returns:
            Styled text with line-level markers removed.
        """
        heading = _HEADING_PATTERN.match(line)
        if heading:
            result = inline_markdown(heading.group(2))
            result.style = "mito.accent bold"
            return result

        item = _LIST_PATTERN.match(line)
        if item:
            result = Text(item.group(1), style="mito.accent")
            result.append(inline_markdown(item.group(2)))
            return result

        quote = _QUOTE_PATTERN.match(line)
        if quote:
            result = Text(quote.group(1), style="mito.muted")
            result.append(inline_markdown(quote.group(2)))
            result.style = "mito.muted"
            return result

        return inline_markdown(line)

    def _print_code(self, line: str) -> None:
        """Print one line of a fenced code block verbatim.

        Args:
            line: The raw source line.
        """
        if not line.strip():
            self._print_blank()
            return
        self._last_blank = False
        self._print(Text(line, style="mito.code"))

    def _print_blank(self) -> None:
        """Print a blank line unless the previous emitted line was also blank."""
        if self._last_blank:
            return
        self._last_blank = True
        self._print(Text(""))

    def _print(self, renderable: Text) -> None:
        """Write one renderable to the console as a single line.

        Args:
            renderable: Text to print; a trailing newline is always added.
        """
        self._console.print(
            renderable,
            end="\n",
            highlight=False,
            soft_wrap=False,
        )


__all__ = ["StreamingRenderer"]
