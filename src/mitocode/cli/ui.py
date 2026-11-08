"""Interactive shell surface for the MitoCode command line interface.

:class:`CliUI` owns every piece of terminal output that is not pure formatting:
the startup banner, tool call and result lines, the transient thinking/tool
spinner, notes and warnings, the end-of-turn summary, and the prompt input.

Two rules shape the design. First, all output goes through an injected
:class:`rich.console.Console`, so tests read a ``StringIO`` instead of a real
terminal. Second, input goes through an injected prompt session; ``ask`` and
:meth:`CliUI.read_prompt` only toggle :attr:`CliUI.prompt_active` and let
``EOFError``/``KeyboardInterrupt`` reach the caller, because the REPL decides
whether either means "exit".

Spinners are :class:`rich.live.Live` renderables marked transient, so they leave
no residue once stopped. Rich suppresses live output on a non-interactive
console, which is why tests only exercise their lifecycle and never assert on
animation frames.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from mitocode.cli.format import format_elapsed, format_tokens
from mitocode.cli.stream import StreamingRenderer

PROMPT_MESSAGE = "❯ "
"""Message shown for the main input prompt."""

BANNER_HINT = "  /help for commands · \\ + Enter for newline · Ctrl+C to interrupt"
"""Dim hint printed under the banner."""

SLASH_COMMANDS: tuple[str, ...] = (
    "help",
    "status",
    "model",
    "effort",
    "clear",
    "save",
    "sessions",
    "demo",
    "agents",
    "tools",
    "mcp",
    "permissions",
    "verbose",
    "cost",
    "log",
    "exit",
    "quit",
)
"""Slash commands offered by the prompt completer."""

_LABEL_WIDTH = 10
"""Width of the muted label column inside the banner."""

_SUCCESS_STATES = frozenset({"connected", "ready", "running", "ok"})
"""MCP states rendered with the success colour."""

_ERROR_STATES = frozenset({"failed", "error", "disconnected", "closed"})
"""MCP states rendered with the error colour."""

_LIVE_STATES = frozenset({"thinking", "reasoning", "tool"})
"""Kinds of transient spinner :class:`CliUI` can show."""

_REASONING_BUFFER_LIMIT = 2000
"""Maximum characters of reasoning text kept for the live preview."""

_REASONING_PREVIEW_LINES = 6
"""Maximum number of reasoning lines shown under the Reasoning header."""

_REASONING_LINE_WIDTH = 120
"""Maximum characters of one reasoning preview line before truncation."""

_WORKSPACE_LIMIT = 64
"""Maximum characters of workspace path shown in the banner and status line."""


@dataclass
class BannerInfo:
    """Facts rendered by :meth:`CliUI.show_banner`.

    Attributes:
        version: MitoCode version string, shown with a ``v`` prefix.
        model: Model the provider will use for this session.
        workspace: Workspace root the agent operates in.
        tool_count: Number of registered built-in tools.
        mcp: MCP server name to state, rendered as coloured dots.
    """

    version: str
    model: str
    workspace: Path
    tool_count: int
    mcp: Dict[str, str] = field(default_factory=dict)


def _short_home(path: Path) -> str:
    """Render a path with the home directory abbreviated to ``~``.

    Args:
        path: Absolute or relative path to display.

    Returns:
        The path unchanged when it is not under the home directory, otherwise
        the ``~``-prefixed form.
    """
    try:
        expanded = Path(path).expanduser()
        return "~/" + str(expanded.relative_to(Path.home()))
    except (ValueError, RuntimeError, OSError):
        return str(path)


def _workspace_label(path: Path, *, limit: int = _WORKSPACE_LIMIT) -> str:
    """Render a workspace path for one-line display.

    Deep project paths are shortened in the middle rather than the tail, because
    the final directory name identifies the project while the leading
    ``/private/var/...`` prefix does not.

    Args:
        path: Workspace root.
        limit: Maximum length of the returned label.

    Returns:
        A ``~``-abbreviated path, middle-ellipsized when longer than *limit*
        characters.
    """
    shorthand = _short_home(path)
    if len(shorthand) <= limit:
        return shorthand
    budget = limit - 1
    head_length = budget // 3
    tail_length = budget - head_length
    return f"{shorthand[:head_length]}…{shorthand[-tail_length:]}"


def _mcp_style(state: str) -> str:
    """Pick the style for the MCP status dot.

    Args:
        state: Server state as reported by the MCP manager or config loader.

    Returns:
        ``mito.success`` for connected servers, ``mito.error`` for failures and
        ``mito.muted`` for anything else.
    """
    normalized = state.strip().lower()
    if normalized in _SUCCESS_STATES:
        return "mito.success"
    if normalized in _ERROR_STATES:
        return "mito.error"
    return "mito.muted"


def _label_row(label: str, value: str) -> Text:
    """Build one ``label  value`` banner row.

    Args:
        label: Muted left-hand label.
        value: Value text, rendered without an extra style.

    Returns:
        Text whose plain form starts with the padded label. Rows never wrap:
        a long value is ellipsized so the panel keeps its one-line-per-field
        shape.
    """
    row = Text.assemble((f"{label:<{_LABEL_WIDTH}}", "mito.muted"), value)
    row.no_wrap = True
    row.overflow = "ellipsis"
    return row


def _mcp_row(servers: Dict[str, str]) -> Text:
    """Build the banner's MCP row.

    Args:
        servers: Server name to state mapping.

    Returns:
        A muted ``mcp`` label followed by ``● name state`` entries separated by
        ``·``, or ``none`` when no server is configured.
    """
    row = Text(f"{'mcp':<{_LABEL_WIDTH}}", style="mito.muted")
    row.no_wrap = True
    row.overflow = "ellipsis"
    if not servers:
        row.append("none", style="mito.muted")
        return row
    for index, (name, state) in enumerate(servers.items()):
        if index:
            row.append(" · ", style="mito.muted")
        row.append("● ", style=_mcp_style(state))
        row.append(name)
        if state:
            row.append(f" {state}", style="mito.muted")
    return row


def _consume_task_result(task: "asyncio.Task[None]") -> None:
    """Retrieve a finished ticker task's result so asyncio does not warn.

    Args:
        task: A ticker task created by :meth:`CliUI._start_ticker`.
    """
    if task.cancelled():
        return
    task.exception()


class CliUI:  # pylint: disable=too-many-instance-attributes
    """Render the interactive CLI and own the prompt session.

    One instance lives for the whole process. The event pump and the REPL share
    it, which is why every visible state change funnels through these methods
    rather than through raw ``console.print`` calls elsewhere.

    Attributes:
        console: Console every renderable is written to.
        verbose: When true, callers show full tool output instead of previews.
        stream: Renderer for streamed assistant text.
        prompt_active: Whether a prompt is currently asking for input; approval
            requests read this to refuse stealing the input line.
        agent_count: Number of live agents shown in the prompt toolbar.
    """

    def __init__(
        self,
        console: Console,
        *,
        history_path: Path,
        verbose: bool = False,
        prompt_session: Optional[Any] = None,
    ) -> None:
        """Initialise the UI.

        Args:
            console: Console that receives all output.
            history_path: File that receives prompt history, always inside the
                workspace; there is no home-directory fallback.
            verbose: Show complete tool output instead of previews.
            prompt_session: Pre-built prompt session; tests inject a fake. When
                ``None``, :meth:`build_prompt_session` is called on first use.
        """
        self.console = console
        self.verbose = verbose
        self.stream = StreamingRenderer(console)
        self.prompt_active = False
        self.agent_count = 1
        self._prompt_session = prompt_session
        self._history_path = history_path
        self._banner_info: Optional[BannerInfo] = None
        self._live: Optional[Live] = None
        self._live_kind: Optional[str] = None
        self._live_started = 0.0
        self._paused = False
        self._ticker: Optional["asyncio.Task[None]"] = None
        self._reasoning_text = ""

    # ------------------------------------------------------------------
    # Banner, messages and turn summary
    # ------------------------------------------------------------------

    def show_banner(self, info: BannerInfo) -> None:
        """Print the startup banner and its one-line hint.

        Args:
            info: Version, model, workspace, tool and MCP facts to display.
        """
        self._banner_info = info

        workspace = _workspace_label(info.workspace, limit=_WORKSPACE_LIMIT)

        lines: List[Text] = [
            Text.assemble(
                ("◆ MitoCode", "bold mito.accent"),
                (f"  v{info.version}", "mito.muted"),
            ),
            Text(""),
            _label_row("model", info.model),
            _label_row("workspace", workspace),
            _label_row("tools", f"{info.tool_count} built-in · {len(info.mcp)} MCP"),
            _mcp_row(info.mcp),
        ]
        self.console.print(
            Panel(
                Group(*lines),
                box=box.ROUNDED,
                border_style="mito.muted",
                expand=False,
                padding=(0, 2),
            )
        )
        self.console.print(Text(BANNER_HINT, style="mito.muted"))

    def note(self, text: str) -> None:
        """Print a muted informational line.

        Args:
            text: Message such as ``Interrupted.`` or ``Denied by user``.
        """
        self.console.print(Text(f"  {text}", style="mito.muted"))

    def warn(self, text: str) -> None:
        """Print a recoverable warning.

        Args:
            text: Message shown in the warning colour.
        """
        self.console.print(Text(f"  {text}", style="mito.warn"))

    def error(self, text: str) -> None:
        """Print an error line.

        Args:
            text: Message shown in the error colour.
        """
        self.console.print(Text(f"  {text}", style="mito.error"))

    def finish_turn(self, seconds: float, tokens: int, tool_calls: int) -> None:
        """Flush streamed text and print the end-of-turn summary.

        Args:
            seconds: Wall-clock duration of the turn.
            tokens: Approximate tokens used in the turn.
            tool_calls: Number of main-agent tool calls in the turn.
        """
        self.end_reasoning()
        self.stream.flush()
        self.stop_thinking()
        self._stop_live()
        self.console.print()
        self.console.print(
            Text.assemble(
                ("✻ ", "mito.accent"),
                (
                    f"Done in {format_elapsed(seconds)} · {format_tokens(tokens)} tokens"
                    f" · {tool_calls} tool calls",
                    "mito.muted",
                ),
            )
        )

    def clear_screen(self) -> None:
        """Clear the terminal and reprint one line of context."""
        self.console.clear()
        self.console.print(self._status_line(), soft_wrap=True)

    def _status_line(self) -> Text:
        """Build the single line reprinted after a screen clear.

        Returns:
            Muted ``model · workspace`` text, or a plain product line before the
            banner has been shown.
        """
        info = self._banner_info
        if info is None:
            return Text("  MitoCode", style="mito.muted")
        return Text(
            f"  {info.model} · {_workspace_label(info.workspace)}",
            style="mito.muted",
            no_wrap=True,
            overflow="ellipsis",
        )

    # ------------------------------------------------------------------
    # Tool lines and spinners
    # ------------------------------------------------------------------

    def start_thinking(self) -> None:
        """Show the transient ``✻ Thinking…`` spinner."""
        if self._live_kind == "thinking" and self._live is not None:
            return
        self._begin_live("thinking")

    def stop_thinking(self) -> None:
        """Stop the thinking spinner if it is the active one."""
        if self._live_kind == "thinking":
            self._stop_live()

    def feed_reasoning(self, delta: str) -> None:
        """Stream one reasoning delta into the transient reasoning region.

        The first delta switches the thinking spinner to a ``"reasoning"``
        live region. The segment clock is inherited from the thinking
        spinner when one was running, so ``Thought for`` counts from the
        moment the turn started waiting, not from the first token. Empty
        deltas are ignored entirely: they neither open nor update the
        region.

        Args:
            delta: One piece of model reasoning text.
        """
        if not delta:
            return
        if self._live_kind != "reasoning":
            inherited = self._live_started if self._live_kind == "thinking" else None
            self._begin_live("reasoning")
            if inherited is not None:
                self._live_started = inherited
        self._reasoning_text = (self._reasoning_text + delta)[-_REASONING_BUFFER_LIMIT:]
        if self._live is not None:
            self._live.update(self._render_live("reasoning", self._elapsed()))

    def end_reasoning(self) -> None:
        """Collapse the reasoning region, printing one duration summary.

        Idempotent: calls while no reasoning region is active do nothing.
        The summary line is printed only when the segment actually received
        content, so an aborted or empty reasoning stream leaves no residue.
        The live region is transient, so stopping it clears the preview and
        only the summary remains on screen.
        """
        if self._live_kind != "reasoning":
            return
        elapsed = self._elapsed()
        had_content = bool(self._reasoning_text.strip())
        self._reasoning_text = ""
        self._stop_live()
        if had_content:
            self.console.print(
                Text.assemble(
                    ("✻ ", "mito.accent"),
                    (f"Thought for {elapsed}s", "mito.muted"),
                )
            )

    def begin_tool(self, call: Text, prefix: str = "") -> None:
        """Flush assistant text, print a tool header and start its spinner.

        A main-agent call (no *prefix*) owns the transient region: any
        spinner is replaced by the tool spinner. A background call keeps the
        main agent's thinking or reasoning region alive — its header and
        result lines print above it — and only shows its own spinner when
        the region is idle or already showing a tool spinner, so background
        work can never break up a main-agent thinking segment or its clock.

        Args:
            call: Header produced by
                :func:`mitocode.cli.format.format_tool_call`.
            prefix: Optional muted prefix such as ``"↳ sub-agent "`` used for
                background agents.
        """
        self.stream.flush()
        if prefix:
            if self._live_kind not in (None, "tool"):
                self.console.print(
                    Text.assemble(
                        ("● ", "mito.accent"),
                        (prefix, "mito.muted"),
                        call,
                    )
                )
                return
            self._stop_live()
        else:
            self.stop_thinking()
            self._stop_live()
        self.console.print(
            Text.assemble(
                ("● ", "mito.accent"),
                (prefix, "mito.muted"),
                call,
            )
        )
        self._begin_live("tool")

    def finish_tool(self, result: Text, prefix: str = "") -> None:
        """Stop the tool spinner and print the result line.

        Args:
            result: Preview produced by
                :func:`mitocode.cli.format.format_result`; its own styles mark
                failures, so this method never recolours it.
            prefix: Optional muted prefix used for background agents.
        """
        if self._live_kind == "tool":
            self._stop_live()
        self.console.print(
            Text.assemble(
                ("  ⎿ ", "mito.muted"),
                (prefix, "mito.muted"),
                result,
            )
        )

    def pause_live(self) -> None:
        """Stop any transient spinner before a permission panel is printed."""
        if self._live is None and self._live_kind is None:
            return
        self._paused = True
        self._stop_live(clear_kind=False)

    def resume_live(self) -> None:
        """Restart the spinner that :meth:`pause_live` stopped."""
        if not self._paused:
            return
        self._paused = False
        if self._live_kind in _LIVE_STATES and self._live is None:
            self._start_live()

    def _begin_live(self, kind: str) -> None:
        """Switch to a new transient spinner.

        Args:
            kind: One of :data:`_LIVE_STATES`.
        """
        self._stop_live()
        self._paused = False
        self._live_kind = kind
        self._live_started = time.monotonic()
        self._start_live()

    def _start_live(self) -> None:
        """Create and start the live renderable for the current spinner kind."""
        kind = self._live_kind
        if kind is None or self._live is not None:
            return
        live = Live(
            self._render_live(kind, 0),
            console=self.console,
            transient=True,
            refresh_per_second=8,
            vertical_overflow="visible",
        )
        live.start()
        self._live = live
        self._start_ticker()

    def _stop_live(self, clear_kind: bool = True) -> None:
        """Cancel the ticker and stop the live display.

        Args:
            clear_kind: Also forget which spinner was active. ``False`` is used
                by :meth:`pause_live`, which must remember what to resume.
        """
        self._stop_ticker()
        live, self._live = self._live, None
        if live is not None:
            live.stop()
        if clear_kind:
            self._live_kind = None

    def _elapsed(self) -> int:
        """Return whole seconds since the current spinner started.

        Returns:
            Non-negative elapsed seconds.
        """
        return max(0, int(time.monotonic() - self._live_started))

    def _render_live(self, kind: str, elapsed: int) -> Any:
        """Build the renderable for one spinner refresh.

        Args:
            kind: ``"thinking"``, ``"reasoning"`` or ``"tool"``.
            elapsed: Whole seconds since the spinner started.

        Returns:
            A ``Text`` for thinking, a header plus trailing reasoning preview
            lines for reasoning, or a one-row grid with a dots spinner for a
            running tool.
        """
        if kind == "thinking":
            return Text.assemble(
                ("✻ ", "mito.accent"),
                (f"Thinking… ({elapsed}s)", "mito.muted"),
            )
        if kind == "reasoning":
            header = Text.assemble(
                ("✻ ", "mito.accent"),
                (f"Reasoning… ({elapsed}s)", "mito.muted"),
            )
            lines = [
                line.strip()
                for line in self._reasoning_text.splitlines()
                if line.strip()
            ]
            preview = [
                Text(f"  {line[:_REASONING_LINE_WIDTH]}", style="mito.muted")
                for line in lines[-_REASONING_PREVIEW_LINES:]
            ]
            if not preview:
                return header
            return Group(header, *preview)
        grid = Table.grid(padding=0)
        grid.add_row(
            Text("  ⎿ ", style="mito.muted"),
            Spinner("dots", style="mito.accent"),
            Text(f" Running… ({elapsed}s)", style="mito.muted"),
        )
        return grid

    def _start_ticker(self) -> None:
        """Schedule the coroutine that refreshes elapsed seconds every second."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called outside the event loop (a sync test or a one-shot script):
            # the spinner simply stays on its initial frame.
            return
        task = loop.create_task(self._tick())
        task.add_done_callback(_consume_task_result)
        self._ticker = task

    def _stop_ticker(self) -> None:
        """Cancel the elapsed-seconds ticker if one is running."""
        task, self._ticker = self._ticker, None
        if task is not None and not task.done():
            task.cancel()

    async def _tick(self) -> None:
        """Refresh the active spinner's elapsed counter once per second."""
        while True:
            await asyncio.sleep(1.0)
            live = self._live
            kind = self._live_kind
            if live is None or kind is None:
                return
            live.update(self._render_live(kind, self._elapsed()))

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    async def ask(self, message: str) -> str:
        """Ask for one line of input through the prompt session.

        ``EOFError`` and ``KeyboardInterrupt`` are deliberately not caught: the
        REPL owns exit and cancellation semantics.

        Args:
            message: Prompt message shown to the user.

        Returns:
            The submitted text, exactly as ``prompt_async`` returned it.
        """
        session = self._prompt()
        self.prompt_active = True
        try:
            answer = await session.prompt_async(message)
        finally:
            self.prompt_active = False
        return answer

    async def read_prompt(self) -> str:
        """Read the main REPL prompt.

        Returns:
            The submitted text, exactly as ``prompt_async`` returned it.

        Raises:
            EOFError: When the user presses Ctrl+D.
            KeyboardInterrupt: When the user presses Ctrl+C.
        """
        return await self.ask(PROMPT_MESSAGE)

    def _prompt(self) -> Any:
        """Return the prompt session, building the real one on first use.

        Returns:
            The injected fake or the lazily built ``PromptSession``.
        """
        if self._prompt_session is None:
            self._prompt_session = self.build_prompt_session()
        return self._prompt_session

    def build_prompt_session(self, history_path: Optional[Path] = None) -> Any:
        """Build the real ``PromptSession`` used for input.

        The history directory is created with ``0o700`` and the file itself
        with ``0o600`` because prompt history can contain file paths and pasted
        content. A directory or file that cannot be created or tightened
        degrades to a session that keeps history in memory rather than failing
        startup.

        Args:
            history_path: History file override; defaults to the instance's
                configured path.

        Returns:
            A configured :class:`prompt_toolkit.PromptSession`.
        """
        path = (
            Path(history_path) if history_path is not None else Path(self._history_path)
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.parent.chmod(0o700)
            path.touch(mode=0o600, exist_ok=True)
            path.chmod(0o600)
        except OSError:
            pass
        return PromptSession(
            history=FileHistory(str(path)),
            completer=WordCompleter(
                [f"/{name}" for name in SLASH_COMMANDS], ignore_case=True
            ),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._prompt_key_bindings(),
            bottom_toolbar=self._toolbar_text,
        )

    def _prompt_key_bindings(self) -> KeyBindings:
        """Build the custom prompt key bindings.

        ``Enter`` submits, unless the buffer ends with a backslash (which is
        removed while a newline is inserted). ``Alt+Enter`` and ``Ctrl+J``
        always insert a newline, and ``Ctrl+L`` clears the screen.

        Returns:
            Key bindings layered over prompt_toolkit's defaults.
        """
        bindings = KeyBindings()

        @bindings.add("enter")
        def _submit(event: Any) -> None:
            """Submit the buffer, or continue the line after a backslash.

            Args:
                event: prompt_toolkit key press event.
            """
            buffer = event.current_buffer
            if buffer.text.endswith("\\"):
                buffer.delete_before_cursor(1)
                buffer.insert_text("\n")
                return
            buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        @bindings.add("c-j")
        def _newline(event: Any) -> None:
            """Insert a literal newline.

            Args:
                event: prompt_toolkit key press event.
            """
            event.current_buffer.insert_text("\n")

        @bindings.add("c-l")
        def _clear(event: Any) -> None:
            """Clear the screen and redraw the prompt.

            Args:
                event: prompt_toolkit key press event.
            """
            self.clear_screen()
            event.app.invalidate()

        return bindings

    def _toolbar_text(self) -> str:
        """Render the prompt's bottom toolbar.

        Returns:
            ``model · workspace · N agents · /help`` built from the last banner.
        """
        info = self._banner_info
        model = info.model if info is not None else "model"
        workspace = (
            _workspace_label(info.workspace)
            if info is not None
            else _workspace_label(Path.cwd())
        )
        agent_word = "agent" if self.agent_count == 1 else "agents"
        return f"{model} · {workspace} · {self.agent_count} {agent_word} · /help"


__all__ = [
    "BANNER_HINT",
    "CliUI",
    "PROMPT_MESSAGE",
    "SLASH_COMMANDS",
    "BannerInfo",
]
