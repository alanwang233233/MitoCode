"""Read-only ``/demo`` playback of a saved or in-memory session.

The demo rebuilds the picture from a session's message blocks with synthetic
pacing: it never calls the orchestrator, never writes into the session and
never touches the CLI's counters.  Pacing is controlled by
:class:`DemoControls` (speed, pause, quit), keys come from a
:class:`KeySource` (:class:`TerminalKeySource` live, :class:`FakeKeySource` in
tests) and :class:`DemoPlayer` drives the loop.
"""

from __future__ import annotations

import asyncio
import os
import select
import sys
import termios
from pathlib import Path
from typing import Any, List, Optional, Protocol

from loguru import logger as loguru_logger
from rich.console import Console
from rich.text import Text

# noinspection PyProtectedMember
from mitocode.core.agent.agent_session import (
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from mitocode.cli.format import format_result, format_tool_call

logger = loguru_logger.bind(name=__name__)

DEMO_CHAR_CHUNK = 16
"""Characters of assistant text printed per playback tick."""

DEMO_TICK = 0.05
"""Base seconds between ticks; the real delay is ``DEMO_TICK / speed``."""

DEMO_SPEEDS = (0.5, 1, 2, 4, 8, 16)
"""Selectable playback speed multipliers, slowest first."""

DEMO_DEFAULT_SPEED_INDEX = 3
"""Index into :data:`DEMO_SPEEDS` the player starts at (4x)."""

_KEY_MAP = {
    "\x1b[D": "left",
    "\x1b[C": "right",
    " ": "space",
    "q": "q",
    "\x1b": "esc",
    "\x03": "ctrl+c",
}
"""Raw terminal input translated into normalized demo key names."""


def _map_key(raw: str) -> Optional[str]:
    """Translate one raw terminal read into a normalized key name.

    Args:
        raw: Characters read from stdin in one read, e.g. ``"\\x1b[D"``.

    Returns:
        A key name understood by :meth:`DemoControls.handle_key`, or ``None``
        when the input is not one of the demo's keys.
    """
    return _KEY_MAP.get(raw)


def format_speed(speed: float) -> str:
    """Render a speed multiplier without a trailing ``.0``.

    Args:
        speed: One of the :data:`DEMO_SPEEDS` values.

    Returns:
        ``"4"`` for ``4.0`` and ``"0.5"`` for ``0.5``.
    """
    return f"{speed:g}"


class DemoControls:
    """Pure playback state machine: speed, pause and quit.

    The class holds no I/O: the demo loop feeds it raw key names and prints
    whatever :meth:`change_note` returns, which makes every transition directly
    unit-testable.  Speed changes and pause toggles clamp at the
    :data:`DEMO_SPEEDS` boundaries and announce themselves through a pending
    note instead of printing directly.
    """

    def __init__(self) -> None:
        """Start at the default speed, playing and not quitting."""
        self.speed_index: int = DEMO_DEFAULT_SPEED_INDEX
        self.paused: bool = False
        self.quit_requested: bool = False
        self._pending_note: str = ""

    @property
    def speed(self) -> float:
        """Return the current speed multiplier."""
        return DEMO_SPEEDS[self.speed_index]

    def handle_key(self, key: Optional[str]) -> None:
        """Apply one key to the playback state.

        ``left``/``right`` shift the speed within :data:`DEMO_SPEEDS` (no-op at
        the boundaries), ``space`` toggles pause, and ``q``/``esc``/``ctrl+c``
        request quitting.  Anything else -- including ``None`` for "no key this
        tick" -- is ignored.

        Args:
            key: Normalized key name, or ``None`` when nothing was pressed.
        """
        if key == "left":
            self._shift_speed(-1)
        elif key == "right":
            self._shift_speed(1)
        elif key == "space":
            self.paused = not self.paused
            self._pending_note = (
                "Paused · space to resume" if self.paused else "Resumed."
            )
        elif key in ("q", "esc", "ctrl+c"):
            self.quit_requested = True

    def _shift_speed(self, delta: int) -> None:
        """Move the speed index by *delta*, clamped to the table.

        Args:
            delta: ``-1`` to slow down, ``1`` to speed up.
        """
        target = self.speed_index + delta
        if target < 0 or target >= len(DEMO_SPEEDS):
            return
        self.speed_index = target
        self._pending_note = f"Speed {format_speed(DEMO_SPEEDS[target])}x"

    def change_note(self) -> str:
        """Return and clear the note produced by the last key, if any.

        Returns:
            A note such as ``"Speed 2x"`` or ``"Paused · space to resume"``,
            or ``""`` when the last key changed nothing worth announcing.
        """
        note, self._pending_note = self._pending_note, ""
        return note


class KeySource(Protocol):
    """Blocking one-key reader the demo loop polls each tick."""

    def poll(self, timeout: float) -> Optional[str]:
        """Wait up to *timeout* seconds for one key press.

        Args:
            timeout: Maximum seconds to block waiting for input.

        Returns:
            A normalized key name, or ``None`` when nothing was pressed.
        """


class TerminalKeySource:
    """Read arrow keys, space, ``q``, Esc and Ctrl+C in raw terminal mode.

    Uses :mod:`termios` (``ICANON``/``ECHO``/``ISIG`` off) plus :func:`select`
    with the poll timeout. When stdin is not a TTY the source degrades to a
    source that never yields a key. The original terminal attributes are
    restored on :meth:`close`; a failed restore is logged, never raised, and
    ``close`` is idempotent.
    """

    def __init__(self) -> None:
        """Switch stdin into raw mode when possible, else degrade silently."""
        self._original = None
        self._closed = False
        try:
            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
        except (termios.error, ValueError, OSError, AttributeError):
            return
        raw = list(attrs)
        raw[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        try:
            termios.tcsetattr(fd, termios.TCSANOW, raw)
        except termios.error:
            return
        self._original = attrs

    def poll(self, timeout: float) -> Optional[str]:
        """Wait for one raw read from stdin and map it to a key name.

        Args:
            timeout: Maximum seconds :func:`select.select` blocks.

        Returns:
            The mapped key name, or ``None`` when nothing arrived in time.
        """
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        try:
            raw = os.read(sys.stdin.fileno(), 32).decode("utf-8", "replace")
        except (OSError, ValueError):
            return None
        return _map_key(raw)

    def close(self) -> None:
        """Restore the original terminal attributes exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._original is None:
            return
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._original)
        except (termios.error, ValueError, OSError):
            logger.warning("Failed to restore terminal attributes after demo")
        self._original = None


class FakeKeySource:
    """Scripted :class:`KeySource` that replays a preset key sequence.

    Poll timeouts are recorded in :attr:`timeouts` so tests can observe the
    pacing the player asked for. Once the sequence is exhausted, ``poll``
    returns ``None`` forever.
    """

    def __init__(self, keys: List[str]) -> None:
        """Store the scripted sequence.

        Args:
            keys: Key names to hand out in order.
        """
        self._keys = list(keys)
        self.timeouts: List[float] = []

    def poll(self, timeout: float) -> Optional[str]:
        """Pop the next scripted key, or ``None`` once the sequence is gone.

        Args:
            timeout: Ignored for input, but recorded in :attr:`timeouts`.

        Returns:
            The next key name, or ``None`` when the sequence is exhausted.
        """
        self.timeouts.append(timeout)
        if self._keys:
            return self._keys.pop(0)
        return None


class DemoTextRenderer:
    """Print assistant text in fixed-size chunks with plain rendering.

    Deliberately not the live ``StreamingRenderer``: that only emits complete
    newline-terminated lines, which would make a whole segment pop out at once
    and defeat chunked pacing. Chunks are printed with ``end=""`` so the next
    chunk continues on the same line; :meth:`finish` writes the single closing
    newline when the text did not already end with one.
    """

    def __init__(self, console: Console) -> None:
        """Bind the renderer to one console.

        Args:
            console: Rich console the chunks are written to.
        """
        self._console = console
        self._needs_newline = False

    def print_chunk(self, chunk: str) -> None:
        """Print one chunk of assistant text inline.

        Args:
            chunk: The next few characters of the message, possibly containing
                newlines. Empty chunks are ignored.
        """
        if not chunk:
            return
        self._console.print(
            Text(chunk), end="", markup=False, soft_wrap=True, highlight=False
        )
        self._needs_newline = not chunk.endswith("\n")

    def finish(self) -> None:
        """Close the last assistant message with one newline, if needed."""
        if self._needs_newline:
            self._console.print()
            self._needs_newline = False


def render_session_history(
    ui: Any,
    session: Any,
    *,
    title: str,
    tool_manager: Any = None,
    root: Optional[Path] = None,
) -> None:
    """Print a restored session's history to the terminal at once.

    A session restored at startup is live in memory but invisible: the fresh
    terminal shows only the banner, so the user has no idea what the
    conversation was about.  This renders the same picture :class:`DemoPlayer`
    would draw -- users, assistant answers with their reasoning, tool calls
    and results -- but instantly, with no key handling and no pacing.

    Read-only by the same rules as the demo: ``system`` blocks and unfinished
    tool calls are skipped, nothing calls the orchestrator, nothing writes
    into the session and no counter is touched.

    Args:
        ui: Terminal surface; its ``console`` receives the transcript and
            ``note`` prints the header.
        session: Object exposing ``messages``; only read, never written.
        title: Session title shown in the header note.
        tool_manager: Optional registry consulted for ``display_param``.
        root: Workspace root used to relativise paths; defaults to ``.``.
    """
    blocks = list(getattr(session, "messages", None) or [])
    ui.note(f'Restored · "{title}" · {len(blocks)} message blocks')
    root = Path(root) if root is not None else Path(".")
    renderer = DemoTextRenderer(ui.console)
    pending_calls: dict[str, tuple[str, dict]] = {}
    verbose = bool(getattr(ui, "verbose", False))
    display_param = _display_param_of(tool_manager)
    for block in blocks:
        if isinstance(block, _PlainTextMessageBlock):
            if block.role == "system":
                continue
            if block.role == "user":
                renderer.finish()
                ui.console.print(Text.assemble(("❯ ", "mito.accent"), block.message))
                continue
            if block.reasoning_content:
                renderer.finish()
                ui.console.print(
                    Text(block.reasoning_content, style="dim"),
                    soft_wrap=True,
                )
            renderer.print_chunk(block.message)
            renderer.finish()
        elif isinstance(block, _ToolCallMessageBlock):
            if not block.finished:
                continue
            call = format_tool_call(
                block.name,
                block.parameter,
                display_param(block.name),
                root,
            )
            ui.console.print(Text.assemble(("● ", "mito.accent"), call))
            pending_calls[block.id] = (block.name, dict(block.parameter))
        elif isinstance(block, _ToolResultMessageBlock):
            name, args = pending_calls.pop(block.tool_call_id, ("", {}))
            preview = format_result(
                name,
                args,
                block.content,
                verbose=verbose,
                root=root,
            )
            ui.console.print(Text.assemble(("  ⎿ ", "mito.muted"), preview))
    renderer.finish()


def _display_param_of(tool_manager: Any):
    """Build the display-parameter lookup used by the transcript renderer.

    Args:
        tool_manager: Optional registry exposing ``get_spec``.

    Returns:
        A callable mapping a tool name to its advertised ``display_param``,
        empty when the manager is missing or the tool unknown.
    """
    if tool_manager is None:
        return lambda name: ""

    def _lookup(name: str) -> str:
        """Read one tool's display parameter from the registry.

        Args:
            name: Registered tool name.

        Returns:
            The advertised ``display_param`` or ``""``.
        """
        spec = tool_manager.get_spec(name)
        if spec is None:
            return ""
        return str(getattr(spec, "display_param", "") or "")

    return _lookup


class DemoPlayer:
    """Replay one restored session into the terminal, read-only.

    The player renders the session's message blocks with synthetic pacing: it
    never calls the orchestrator, never writes into the session, and never
    touches the CLI's counters. Text is played through :class:`DemoTextRenderer`
    at ``DEMO_CHAR_CHUNK`` characters per tick; every tick polls the
    :class:`KeySource` and feeds presses to :class:`DemoControls`, so speed,
    pause and quit stay live while the demo runs.
    """

    def __init__(
        self,
        ui: Any,
        session: Any,
        *,
        title: str,
        key_source: Any,
        tool_manager: Any = None,
        root: Optional[Path] = None,
        controls: Optional[DemoControls] = None,
        sleeper: Any = None,
    ) -> None:
        """Wire the player to its collaborators.

        Args:
            ui: Terminal surface; its ``console`` receives the replay.
            session: Object exposing ``messages``; only read, never written.
            title: Session title shown in the header note.
            key_source: Pollable reader for playback keys.
            tool_manager: Optional registry consulted for ``display_param``.
            root: Workspace root used to relativise paths; defaults to ``.``.
            controls: Playback state machine; defaults to a fresh one.
            sleeper: Async delay function; defaults to :func:`asyncio.sleep`.
                Tests inject a no-op to remove real waiting.
        """
        self._ui = ui
        self._session = session
        self._title = title
        self._key_source = key_source
        self._tool_manager = tool_manager
        self._root = Path(root) if root is not None else Path(".")
        self._controls = controls if controls is not None else DemoControls()
        self._sleep = sleeper if sleeper is not None else asyncio.sleep
        self._renderer = DemoTextRenderer(ui.console)
        self._pending_calls: dict[str, tuple[str, dict]] = {}
        self._verbose = bool(getattr(ui, "verbose", False))

    async def run(self) -> None:
        """Play the session history and finish with a closing note.

        Blocks are rendered in order; ``system`` blocks and unfinished tool
        calls are skipped. On return the key source is closed exactly once so
        the terminal leaves raw mode, and ``Demo finished.`` is printed.
        """
        self._ui.note(self._header())
        try:
            for block in list(self._session.messages):
                if self._controls.quit_requested:
                    break
                await self._render_block(block)
        finally:
            self._renderer.finish()
            close = getattr(self._key_source, "close", None)
            if close is not None:
                close()
        self._ui.note("Demo finished.")

    def _header(self) -> str:
        """Build the muted header note shown before playback starts.

        Returns:
            ``Demo · "<title>" · <speed>x · ←/→ speed · space pause · q quit``.
        """
        return (
            f'Demo · "{self._title}" · '
            f"{format_speed(self._controls.speed)}x · "
            "←/→ speed · space pause · q quit"
        )

    def _display_param(self, name: str) -> str:
        """Look up the display parameter advertised for one tool.

        Args:
            name: Registered tool name.

        Returns:
            The registry's ``display_param``, or ``""`` when unknown.
        """
        if self._tool_manager is None:
            return ""
        spec = self._tool_manager.get_spec(name)
        if spec is None:
            return ""
        return str(getattr(spec, "display_param", "") or "")

    async def _render_block(self, block: Any) -> None:
        """Render one message block according to its kind.

        Args:
            block: One entry of ``session.messages``.
        """
        if isinstance(block, _PlainTextMessageBlock):
            if block.role == "system":
                return
            if block.role == "user":
                self._ui.console.print(
                    Text.assemble(("❯ ", "mito.accent"), block.message)
                )
                return
            if block.reasoning_content:
                self._ui.console.print(
                    Text(block.reasoning_content, style="dim"),
                    soft_wrap=True,
                )
            await self._play_text(block.message)
            self._renderer.finish()
        elif isinstance(block, _ToolCallMessageBlock):
            if not block.finished:
                return
            call = format_tool_call(
                block.name,
                block.parameter,
                self._display_param(block.name),
                self._root,
            )
            self._ui.console.print(Text.assemble(("● ", "mito.accent"), call))
            self._pending_calls[block.id] = (block.name, dict(block.parameter))
        elif isinstance(block, _ToolResultMessageBlock):
            name, args = self._pending_calls.pop(block.tool_call_id, ("", {}))
            preview = format_result(
                name,
                args,
                block.content,
                verbose=self._verbose,
                root=self._root,
            )
            self._ui.console.print(Text.assemble(("  ⎿ ", "mito.muted"), preview))

    async def _play_text(self, message: str) -> None:
        """Print assistant text in paced chunks, honouring keys and pause.

        Each tick polls the key source for ``DEMO_TICK / speed`` seconds, feeds
        any press to the controls and announces the resulting note. Pausing
        keeps polling so the user can change speed, resume or quit.

        Args:
            message: The assistant text to play.
        """
        position = 0
        while position < len(message):
            if self._controls.quit_requested:
                return
            await self._poll_key()
            if self._controls.quit_requested:
                return
            if self._controls.paused:
                await self._sleep(DEMO_TICK / self._controls.speed)
                continue
            self._renderer.print_chunk(message[position : position + DEMO_CHAR_CHUNK])
            position += DEMO_CHAR_CHUNK
            await self._sleep(DEMO_TICK / self._controls.speed)

    async def _poll_key(self) -> None:
        """Poll one key in a thread and apply it to the controls."""
        key = await asyncio.to_thread(
            self._key_source.poll, DEMO_TICK / self._controls.speed
        )
        if key is None:
            return
        self._controls.handle_key(key)
        note = self._controls.change_note()
        if note:
            self._ui.note(note)


__all__ = [
    "DEMO_CHAR_CHUNK",
    "DEMO_DEFAULT_SPEED_INDEX",
    "DEMO_SPEEDS",
    "DEMO_TICK",
    "DemoControls",
    "DemoPlayer",
    "DemoTextRenderer",
    "FakeKeySource",
    "KeySource",
    "TerminalKeySource",
    "format_speed",
    "render_session_history",
]
