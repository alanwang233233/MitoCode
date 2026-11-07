"""Slash-command registry and handlers for the interactive CLI.

Every command is an ``async`` handler registered under a bare name (``help``,
not ``/help``) with a usage string and a one-line description.  ``dispatch`` is
the only entry point: a line that does not start with ``/`` is a prompt, not a
command, and returns ``None``; an unknown command prints one red line and
returns an ordinary :class:`CommandResult` so the REPL keeps running.

Two rules keep commands from taking down the REPL.  Handlers that address an
agent by id verify the id and warn instead of indexing a missing session, and
every handler that can fail for a recoverable reason (a bad model number, an
applied prompt answer) reports through
:class:`~mitocode.cli.ui.CliUI` rather than raising.  A command's failures are
never a reason to exit: only ``/exit`` and ``/quit`` return
``exit_requested=True``.

The handlers are deliberately shallow over the objects in
:class:`~mitocode.cli.state.CliState`; the orchestrator, provider, MCP manager
and approver are duck-typed, so the tests drive them with fakes and no command
needs a terminal or a network connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from rich.table import Table
from rich.text import Text

from loguru import logger as loguru_logger

from mitocode.core.agent.agent_state import AgentState

from mitocode.cli.config import KNOWN_REASONING_EFFORTS
from mitocode.cli.demo import DemoPlayer, TerminalKeySource
from mitocode.cli.sessions import (
    SessionLoadError,
    SessionStore,
    select_session,
)
from mitocode.cli.state import CommandContext

if TYPE_CHECKING:
    from mitocode.cli.state import CliState

CommandHandler = Callable[
    [CommandContext, List[str]], Awaitable[Optional["CommandResult"]]
]
"""Signature every command handler implements."""

MAX_AGENT_MESSAGE_CHARS = 2000
"""Longest last-assistant message ``/agents <id>`` prints before truncating."""

UNKNOWN_COMMAND_TEMPLATE = "Unknown command /{name} · /help"
"""Line printed for a slash command that is not registered."""

COMMAND_FAILED_TEMPLATE = "Command failed: {error}"
"""Line printed when a registered handler raises unexpectedly."""


@dataclass(frozen=True)
class Command:
    """One registered slash command.

    Attributes:
        name: Bare command name without the leading slash.
        usage: Usage string starting with ``/``, shown by ``/help``.
        description: One-line human description.
        handler: Coroutine called as ``await handler(ctx, args)``.
    """

    name: str
    usage: str
    description: str
    handler: CommandHandler


@dataclass
class CommandResult:
    """What ``dispatch`` reports back to the REPL.

    Attributes:
        exit_requested: ``True`` only for ``/exit`` and ``/quit``.
    """

    exit_requested: bool = False


COMMANDS: Dict[str, Command] = {}
"""Every registered command, keyed by bare name."""


def command(name: str, usage: str, description: str):  # type: ignore[no-untyped-def]
    """Register the decorated coroutine as a slash command.

    Args:
        name: Bare command name, e.g. ``"help"``.
        usage: Usage string, e.g. ``"/help"``.
        description: One-line description rendered in the help table.

    Returns:
        A decorator that stores the handler and returns it unchanged.
    """

    def decorator(handler: CommandHandler) -> CommandHandler:
        """Store *handler* in :data:`COMMANDS`.

        Args:
            handler: Coroutine implementing the command.

        Returns:
            The same coroutine, so the module keeps a named function.
        """
        COMMANDS[name] = Command(name, usage, description, handler)
        return handler

    return decorator


async def dispatch(line: str, ctx: CommandContext) -> Optional[CommandResult]:
    """Run the slash command in *line*, or report that it is not one.

    Args:
        line: Raw line submitted at the prompt.
        ctx: State, UI and approver available to the handler.

    Returns:
        ``None`` when the line is ordinary prompt text; otherwise a
        :class:`CommandResult`.  Unknown commands return a default result
        (no exit) after printing one error line.  A registered handler that
        raises is logged and reported the same way, so no command failure can
        escape the REPL.
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped[1:].split()
    if not parts:
        ctx.ui.error(UNKNOWN_COMMAND_TEMPLATE.format(name=""))
        return CommandResult()

    name = parts[0].lower()
    entry = COMMANDS.get(name)
    if entry is None:
        ctx.ui.error(UNKNOWN_COMMAND_TEMPLATE.format(name=name))
        return CommandResult()

    try:
        result = await entry.handler(ctx, parts[1:])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Deliberately ``Exception``: KeyboardInterrupt and asyncio's
        # CancelledError are shutdown signals, not command failures.
        loguru_logger.exception("Slash command /{} failed", name)
        ctx.ui.error(COMMAND_FAILED_TEMPLATE.format(error=exc))
        return CommandResult()
    return result if result is not None else CommandResult()


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _model_name(state: "CliState") -> str:
    """Return the provider's current default model.

    Args:
        state: Current CLI state.

    Returns:
        The model name, or ``"unknown"`` when the provider reports none.
    """
    return str(getattr(state.provider, "default_model", "") or "unknown")


def _workspace_path(state: "CliState") -> str:
    """Return the workspace root for display.

    Args:
        state: Current CLI state.

    Returns:
        The root path, or ``"."`` when the workspace object has none.
    """
    return str(getattr(state.workspace, "path", "") or ".")


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    """Read one field from a tree-snapshot entry.

    Args:
        entry: Dictionary or attribute-bearing snapshot entry.
        key: Field name.
        default: Value returned when the field is absent.

    Returns:
        The field value, or *default*.
    """
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _mcp_summary(state: "CliState") -> str:
    """Build the one-line MCP summary shown by ``/status``.

    Args:
        state: Current CLI state.

    Returns:
        ``disabled``, ``no servers``, or a comma-separated list of server
        names and states.
    """
    manager = state.mcp_manager
    if manager is None:
        return "disabled" + (f" · {state.mcp_error}" if state.mcp_error else "")
    try:
        statuses = manager.status()
    except Exception:  # pylint: disable=broad-exception-caught
        return "unavailable"
    if not statuses:
        return "no servers"
    rendered = []
    for name, status in statuses.items():
        state_name = (
            status.get("state", "")
            if isinstance(status, dict)
            else getattr(status, "state", "")
        )
        rendered.append(f"{name} {state_name}".rstrip())
    return ", ".join(rendered)


def _snapshot_ids(state: "CliState") -> List[Any]:
    """Return the agent ids from the orchestrator's tree snapshot.

    Args:
        state: Current CLI state.

    Returns:
        The ids in snapshot order; an empty list when the orchestrator raises
        or the snapshot has no ids.
    """
    try:
        snapshot = state.orchestrator.get_tree_snapshot()
    except Exception:  # pylint: disable=broad-exception-caught
        return []
    return [
        entry_id
        for entry in snapshot
        if (entry_id := _entry_value(entry, "id")) is not None
    ]


def _tree_lines(snapshot: List[Any]) -> List[str]:
    """Render a flat tree snapshot with indentation.

    Args:
        snapshot: Entries carrying ``id``, ``name``, ``state``,
            ``parent_id`` and optionally ``sub_agent_ids``.

    Returns:
        One display line per agent, indented by its depth in the tree.
    """
    by_id = {
        entry_id: entry
        for entry in snapshot
        if (entry_id := _entry_value(entry, "id")) is not None
    }

    def depth(entry: Any) -> int:
        """Return how many ancestors an entry has.

        Args:
            entry: Snapshot entry to measure.

        Returns:
            Non-negative nesting depth, capped by a visited set so a corrupted
            snapshot cannot loop.
        """
        seen = {_entry_value(entry, "id")}
        level = 0
        parent_id = _entry_value(entry, "parent_id")
        while parent_id is not None and parent_id in by_id and parent_id not in seen:
            seen.add(parent_id)
            level += 1
            parent_id = _entry_value(by_id[parent_id], "parent_id")
        return level

    lines = []
    for entry in sorted(by_id.values(), key=lambda item: _entry_value(item, "id", 0)):
        entry_id = _entry_value(entry, "id")
        name = _entry_value(entry, "name", "") or "(unnamed)"
        entry_state = _entry_value(entry, "state", "") or "unknown"
        parent_id = _entry_value(entry, "parent_id")
        children = _entry_value(entry, "sub_agent_ids", []) or []
        line = f"{'  ' * depth(entry)}#{entry_id} {name} [{entry_state}]"
        if parent_id is not None:
            line += f" parent=#{parent_id}"
        if children:
            line += " children=" + ",".join(f"#{child}" for child in children)
        lines.append(line)
    return lines


def _last_assistant_text(session: Any) -> Optional[str]:
    """Return the last assistant text in a session's history.

    Args:
        session: Session whose ``messages`` are inspected.

    Returns:
        The most recent non-empty assistant message, truncated to
        :data:`MAX_AGENT_MESSAGE_CHARS`, or ``None`` when there is none.
    """
    for block in reversed(getattr(session, "messages", []) or []):
        role = (
            block.get("role")
            if isinstance(block, dict)
            else getattr(block, "role", None)
        )
        message = (
            block.get("message")
            if isinstance(block, dict)
            else getattr(block, "message", None)
        )
        if role == "assistant" and isinstance(message, str) and message:
            if len(message) > MAX_AGENT_MESSAGE_CHARS:
                return message[:MAX_AGENT_MESSAGE_CHARS] + "…"
            return message
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@command("help", "/help", "Show commands and keyboard shortcuts.")
async def _cmd_help(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render the command table and the prompt's shortcuts.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Command", style="mito.accent", no_wrap=True)
    table.add_column("Description")
    for entry in COMMANDS.values():
        # Cell values are wrapped in Text: Rich parses plain str cells as
        # console markup, which eats literal "[name]" in a usage string.
        table.add_row(Text(entry.usage), Text(entry.description))
    ctx.ui.console.print(table)
    ctx.ui.console.print(Text("Shortcuts", style="bold mito.accent"))
    ctx.ui.console.print(
        Text(
            "  Enter submit · \\ + Enter newline · Ctrl+L clear · "
            "Ctrl+C interrupt · Ctrl+D exit",
            style="mito.muted",
        )
    )
    return CommandResult()


@command("status", "/status", "Show the session, model, workspace and MCP state.")
async def _cmd_status(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Print one fact per row for the current session.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    session = state.orchestrator.get_session(state.main_id)
    name = ""
    for entry in state.orchestrator.get_tree_snapshot():
        if _entry_value(entry, "id") == state.main_id:
            name = str(_entry_value(entry, "name", "") or "")
            break
    if not name and session is not None:
        name = str(getattr(session, "name", "") or "")
    if not name:
        name = "Main"

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Field", style="mito.muted", no_wrap=True)
    table.add_column("Value")
    table.add_row("session", Text(f"#{state.main_id} {name}"))
    manager = state.session_manager
    if manager is not None:
        table.add_row("title", Text(manager.current_title))
    table.add_row("model", Text(_model_name(state)))
    table.add_row(
        "effort",
        Text(
            str(
                getattr(state.provider, "reasoning_effort", None)
                or "default (not sent)"
            )
        ),
    )
    table.add_row("workspace", Text(_workspace_path(state)))
    table.add_row("messages", Text(str(len(getattr(session, "messages", []) or []))))
    table.add_row("tokens", Text(f"~{state.estimate_tokens()}"))
    table.add_row("tools", Text(str(len(state.tool_manager.human_docs()))))
    table.add_row("agents", Text(str(len(state.orchestrator.get_tree_snapshot()))))
    table.add_row("mcp", Text(_mcp_summary(state)))
    ctx.ui.console.print(table)
    return CommandResult()


@command("model", "/model [name]", "List models or switch the default model.")
async def _cmd_model(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Switch ``provider.default_model`` from an argument or a prompt answer.

    A numeric answer is an index into ``provider.model_list``; any other
    non-empty answer is a model name.  A name that the endpoint did not
    advertise is used anyway, with a warning, and appended to
    ``provider.model_list`` (creating it when the provider reports none) so the
    next turn's ``chat()`` accepts it instead of raising.

    Args:
        ctx: Command context.
        args: Optional model name.

    Returns:
        A default result.
    """
    state = ctx.state
    provider = state.provider
    models = [str(model) for model in (getattr(provider, "model_list", None) or [])]

    if args:
        answer = args[0]
    else:
        if not models:
            ctx.ui.warn(
                "No models reported by the provider; pass a model name instead."
            )
            return CommandResult()
        for index, model in enumerate(models, start=1):
            marker = " (current)" if model == _model_name(state) else ""
            ctx.ui.console.print(Text(f"  {index}. {model}{marker}"))
        try:
            answer = await ctx.ui.ask("  Model number or name: ")
        except (EOFError, KeyboardInterrupt):
            ctx.ui.note("Model unchanged.")
            return CommandResult()
        answer = answer.strip()
        if not answer:
            ctx.ui.note("Model unchanged.")
            return CommandResult()

    if answer.isdigit():
        index = int(answer) - 1
        if not 0 <= index < len(models):
            ctx.ui.warn(f"Model number {answer} is out of range; model unchanged.")
            return CommandResult()
        chosen = models[index]
    else:
        chosen = answer

    if chosen not in models:
        ctx.ui.warn(f"'{chosen}' is not in the provider's model list; using it anyway.")
        registered = list(getattr(provider, "model_list", None) or [])
        if chosen not in registered:
            registered.append(chosen)
            provider.model_list = registered
    provider.default_model = chosen
    ctx.ui.note(f"Model set to {chosen}.")
    return CommandResult()


@command("effort", "/effort [tier]", "Show or set the reasoning effort tier.")
async def _cmd_effort(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Show or set ``provider.reasoning_effort`` for this session.

    A tier outside :data:`KNOWN_REASONING_EFFORTS` is warned about but still
    accepted, mirroring the startup warning: endpoints expose private tiers,
    and a typo only surfaces at request time. ``default`` is a reserved word
    that clears the setting so requests omit the parameter again.

    Args:
        ctx: Command context.
        args: Optional tier name.

    Returns:
        A default result.
    """
    provider = ctx.state.provider
    if not args:
        current = getattr(provider, "reasoning_effort", None) or "default (not sent)"
        ctx.ui.console.print(Text(f"  effort: {current}", style="mito.muted"))
        ctx.ui.console.print(
            Text(
                "  known tiers: "
                + " ".join(sorted(KNOWN_REASONING_EFFORTS))
                + " · /effort default to clear",
                style="mito.muted",
            )
        )
        return CommandResult()
    value = args[0].strip().lower()
    if value == "default":
        provider.reasoning_effort = None
        ctx.ui.note("Reasoning effort cleared; requests will not carry it.")
        return CommandResult()
    if value not in KNOWN_REASONING_EFFORTS:
        ctx.ui.warn(
            f"'{value}' is not a known tier "
            f"({', '.join(sorted(KNOWN_REASONING_EFFORTS))}); "
            "passing it through anyway."
        )
    provider.reasoning_effort = value
    ctx.ui.note(f"Reasoning effort set to {value}.")
    return CommandResult()


@command("clear", "/clear", "Destroy the main agent and start a new session.")
async def _cmd_clear(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Replace the main session and reset this process's counters.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    manager = state.session_manager
    if manager is not None:
        await manager.save_current(force=True)
    await state.orchestrator.destroy_agent(state.main_id)
    state.main_id = await state.orchestrator.create_agent(
        name="Main", description="Interactive session"
    )
    state.turn_count = 0
    state.tool_call_count = 0
    state.total_chars = 0
    state.agents = state.orchestrator.get_tree_snapshot()
    if manager is not None:
        await manager.start_new_session(state.main_session())
    ctx.ui.note(f"New session #{state.main_id}.")
    return CommandResult()


async def _confirm_force_stop(ctx: CommandContext) -> bool:
    """Ask whether a running session may be force-stopped for a switch.

    Busy means an orchestrator task, a RUNNING main session, or a live
    background tool task.  Refusal is reported as ``Switch cancelled.``

    Args:
        ctx: Command context carrying state and UI.

    Returns:
        ``True`` when the session is idle or the user accepted the forced
        stop; ``False`` when the user refused.
    """
    state = ctx.state
    busy = False

    has_active_tasks = getattr(state.orchestrator, "has_active_tasks", None)
    if callable(has_active_tasks) and has_active_tasks():
        busy = True

    if not busy:
        try:
            session = state.main_session()
        except RuntimeError:
            session = None
        if (
            session is not None
            and getattr(session, "state", None) is AgentState.RUNNING
        ):
            busy = True

    if not busy:
        background = getattr(state.tool_context, "background", None)
        has_background = getattr(background, "has_active_tasks", None)
        if callable(has_background) and has_background():
            busy = True

    if not busy:
        return True
    answer = await ctx.ui.ask(
        "Current session is still running. Force stop and switch? [y/N] "
    )
    if answer.strip().lower() in ("y", "yes"):
        return True
    ctx.ui.note("Switch cancelled.")
    return False


def _session_store(ctx: CommandContext) -> SessionStore:
    """Build the session store for the command's workspace.

    Args:
        ctx: Command context carrying the state's workspace.

    Returns:
        A store rooted at the workspace's ``.mitocode/sessions`` directory.
    """
    return SessionStore(Path(_workspace_path(ctx.state)))


@command("save", "/save", "Save the current session to disk now.")
async def _cmd_save(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Force-save the current session tree to disk.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    manager = ctx.state.session_manager
    if manager is None:
        ctx.ui.warn("Session saving is not available in this run.")
        return CommandResult()
    path = await manager.save_current(force=True, report_failure=True)
    if path is None:
        ctx.ui.warn("No session to save.")
        return CommandResult()
    ctx.ui.note(f"Saved session {manager.current_session_id} to {path}.")
    return CommandResult()


@command("sessions", "/sessions", "List saved sessions and switch to one.")
async def _cmd_sessions(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Switch the live tree to another saved session or a brand-new one.

    The current session is saved first; a busy session asks for
    confirmation before its tasks are force-stopped.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    manager = state.session_manager
    if manager is None:
        ctx.ui.warn("Session switching is not available in this run.")
        return CommandResult()

    await manager.save_current()
    if not await _confirm_force_stop(ctx):
        return CommandResult()

    store = _session_store(ctx)
    summaries, _skipped = store.list_sessions()
    choice = await select_session(state.ui, summaries, allow_new=True)
    if choice == manager.current_session_id:
        ctx.ui.note("Already current.")
        return CommandResult()
    if choice == "new":
        await manager.switch_new()
        ctx.ui.note(f'Switched to "{manager.current_title}".')
        return CommandResult()

    try:
        record = store.load(choice)
    except SessionLoadError as exc:
        ctx.ui.warn(f"Session could not be loaded; staying on the current one. ({exc})")
        return CommandResult()
    if await manager.switch_to(record):
        ctx.ui.note(f'Switched to "{manager.current_title}".')
        return CommandResult()
    ctx.ui.warn("Session could not be restored; staying on the current one.")
    return CommandResult()


@command("demo", "/demo", "Replay a saved session as a read-only demo.")
async def _cmd_demo(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Replay one saved session with synthetic pacing, read-only.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    state = ctx.state
    manager = state.session_manager
    if manager is None:
        ctx.ui.warn("Demo playback is not available in this run.")
        return CommandResult()

    store = _session_store(ctx)
    summaries, _skipped = store.list_sessions()
    if not summaries:
        ctx.ui.warn("No saved sessions to replay.")
        return CommandResult()

    if not await _confirm_force_stop(ctx):
        return CommandResult()
    await manager.save_current()

    choice = await select_session(state.ui, summaries, allow_new=False)
    if choice == manager.current_session_id:
        session = state.main_session()
    else:
        try:
            record = store.load(choice)
        except SessionLoadError as exc:
            ctx.ui.warn(
                f"Session could not be loaded; staying on the current one. ({exc})"
            )
            return CommandResult()
        if not await manager.switch_to(record):
            ctx.ui.warn("Session could not be restored; staying on the current one.")
            return CommandResult()
        session = state.main_session()

    pump = getattr(state, "pump", None)
    if pump is not None:
        await pump.wait_idle()
    player = DemoPlayer(
        state.ui,
        session,
        title=manager.current_title,
        key_source=TerminalKeySource(),
        tool_manager=state.tool_manager,
        root=Path(_workspace_path(state)),
    )
    await player.run()
    return CommandResult()


@command("agents", "/agents [id]", "Show the agent tree or one agent's last answer.")
async def _cmd_agents(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render the tree, or the last assistant message for one id.

    Args:
        ctx: Command context.
        args: Optional agent id.

    Returns:
        A default result.
    """
    state = ctx.state
    snapshot = state.orchestrator.get_tree_snapshot()

    if not args:
        lines = _tree_lines(snapshot)
        if not lines:
            ctx.ui.note("No agents.")
            return CommandResult()
        for line in lines:
            ctx.ui.console.print(Text(f"  {line}"))
        return CommandResult()

    raw_id = args[0]
    try:
        agent_id = int(raw_id)
    except ValueError:
        ctx.ui.warn(f"Invalid agent id '{raw_id}'.")
        return CommandResult()

    session = state.orchestrator.get_session(agent_id)
    if session is None:
        ctx.ui.warn(f"No agent #{agent_id}.")
        return CommandResult()

    message = _last_assistant_text(session)
    if message is None:
        ctx.ui.note(f"Agent #{agent_id} has no assistant message yet.")
        return CommandResult()
    ctx.ui.console.print(Text(f"  #{agent_id} {message}"))
    return CommandResult()


@command("tools", "/tools", "List the registered tools.")
async def _cmd_tools(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render tool names and their human descriptions.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    docs = ctx.state.tool_manager.human_docs()
    if not docs:
        ctx.ui.note("No tools registered.")
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Tool", style="mito.accent", no_wrap=True)
    table.add_column("Description")
    for name, description in docs.items():
        table.add_row(Text(name), Text(description or "—"))
    ctx.ui.console.print(table)
    return CommandResult()


@command(
    "skills",
    "/skills [refresh]",
    "List installed skills, or re-scan their directories first with refresh.",
)
async def _cmd_skills(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render the skills catalogue, or re-scan the roots and render it.

    Args:
        ctx: Command context.
        args: Empty for the last discovery's catalogue, otherwise ``refresh``
            to re-scan first.

    Returns:
        A default result.
    """
    state = ctx.state
    manager = state.skills_manager
    if manager is None:
        ctx.ui.note("Skills are not available in this run.")
        return CommandResult()

    if args and args[0].lower() == "refresh":
        if len(args) != 1:
            ctx.ui.warn("Usage: /skills [refresh]")
            return CommandResult()
        try:
            await manager.discover()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            ctx.ui.error(f"Skills refresh failed: {exc}")
            return CommandResult()
    elif args:
        ctx.ui.warn("Usage: /skills [refresh]")
        return CommandResult()

    skills = manager.list_skills()
    issues = manager.issues()
    if issues:
        ctx.ui.warn(
            f"{len(issues)} skill entries were skipped during discovery; "
            "details are in the log."
        )
    if not skills:
        ctx.ui.note(
            "No skills installed. Add a skill as a directory holding a "
            "SKILL.md under .mitocode/skills in this workspace, then run "
            "/skills refresh."
        )
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Skill", style="mito.accent", no_wrap=True)
    table.add_column("Source")
    table.add_column("Description")
    for skill in skills:
        table.add_row(
            Text(skill.name), Text(skill.source or "—"), Text(skill.description)
        )
    ctx.ui.console.print(table)
    return CommandResult()


@command("mcp", "/mcp [refresh [name]]", "Show MCP servers or refresh their tools.")
async def _cmd_mcp(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Render MCP status, or reconnect servers and refresh session tools.

    Args:
        ctx: Command context.
        args: Empty for status, otherwise ``refresh`` and an optional server.

    Returns:
        A default result.
    """
    if args and args[0].lower() == "refresh":
        return await _refresh_mcp(ctx, args[1] if len(args) > 1 else None)
    if args:
        ctx.ui.warn("Usage: /mcp [refresh [name]]")
        return CommandResult()
    return _render_mcp_status(ctx)


async def _refresh_mcp(ctx: CommandContext, server: Optional[str]) -> CommandResult:
    """Reconnect MCP servers and re-snapshot every session's tools.

    Args:
        ctx: Command context.
        server: Server to refresh, or ``None`` for all configured servers.

    Returns:
        A default result.
    """
    state = ctx.state
    manager = state.mcp_manager
    if manager is None:
        ctx.ui.warn("No MCP servers configured.")
        return CommandResult()

    try:
        await manager.refresh(server)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        ctx.ui.error(f"MCP refresh failed: {exc}")
        return CommandResult()

    for agent_id in _snapshot_ids(state):
        session = state.orchestrator.get_session(agent_id)
        refresh = getattr(session, "refresh_tools", None)
        if callable(refresh):
            refresh()
    ctx.ui.note("MCP tools refreshed.")
    return CommandResult()


def _render_mcp_status(ctx: CommandContext) -> CommandResult:
    """Print the server status table, or explain why MCP is unavailable.

    Args:
        ctx: Command context.

    Returns:
        A default result.
    """
    state = ctx.state
    manager = state.mcp_manager
    if manager is None:
        if state.mcp_error:
            ctx.ui.warn(f"MCP unavailable: {state.mcp_error}")
        else:
            ctx.ui.note("No MCP servers configured.")
        return CommandResult()

    statuses = manager.status()
    if not statuses:
        ctx.ui.note("No MCP servers configured.")
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Server", style="mito.accent", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Tools", justify="right", no_wrap=True)
    table.add_column("Error")
    for name, status in statuses.items():
        entry_state = (
            status.get("state", "")
            if isinstance(status, dict)
            else getattr(status, "state", "")
        )
        error = (
            status.get("error", "")
            if isinstance(status, dict)
            else getattr(status, "error", "")
        )
        tool_count = (
            status.get("tool_count", 0)
            if isinstance(status, dict)
            else getattr(status, "tool_count", 0)
        )
        table.add_row(
            Text(name),
            Text(str(entry_state)),
            Text(str(tool_count)),
            Text(str(error or "")),
        )
    ctx.ui.console.print(table)
    return CommandResult()


@command(
    "permissions",
    "/permissions [reset]",
    "Show the approval policy and cached always-allow requests.",
)
async def _cmd_permissions(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Show the policy/cache, or clear the cache on ``reset``.

    Args:
        ctx: Command context.
        args: Optional ``reset``.

    Returns:
        A default result.
    """
    approver = ctx.approver or ctx.state.approver
    if approver is None:
        ctx.ui.note("No interactive approver configured.")
        return CommandResult()

    if args and args[0].lower() == "reset":
        approver.reset_cache()
        ctx.ui.note("Permission cache cleared.")
        return CommandResult()
    if args:
        ctx.ui.warn("Usage: /permissions [reset]")
        return CommandResult()

    policy = getattr(approver, "policy", "ask")
    policy_text = getattr(policy, "value", str(policy))
    ctx.ui.console.print(Text(f"  Policy: {policy_text}"))
    cached = getattr(approver, "cached_requests", None)
    raw_entries: Any = cached() if callable(cached) else []
    entries = sorted(raw_entries)
    if not entries:
        ctx.ui.note("No always-allow requests cached.")
        return CommandResult()

    table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Action", style="mito.accent", no_wrap=True)
    table.add_column("Detail")
    for entry in entries:
        if isinstance(entry, tuple) and len(entry) == 2:
            table.add_row(Text(str(entry[0])), Text(str(entry[1])))
        else:
            table.add_row(Text(str(entry)), "")
    ctx.ui.console.print(table)
    return CommandResult()


@command("verbose", "/verbose", "Toggle full tool output.")
async def _cmd_verbose(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Toggle verbose output on the state and the UI together.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    enabled = not ctx.state.verbose
    ctx.state.verbose = enabled
    ctx.ui.verbose = enabled
    ctx.ui.note(f"Verbose output {'on' if enabled else 'off'}.")
    return CommandResult()


@command("cost", "/cost", "Show approximate tokens used by this session.")
async def _cmd_cost(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Print the character/4 token estimate for the main session.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    ctx.ui.note(f"Approximate tokens: ~{ctx.state.estimate_tokens()}")
    return CommandResult()


@command("log", "/log", "Show the log file path.")
async def _cmd_log(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Print the file the CLI's logs are written to.

    Args:
        ctx: Command context.
        args: Unused extra arguments.

    Returns:
        A default result.
    """
    del args
    ctx.ui.note(str(ctx.state.log_path))
    return CommandResult()


@command("exit", "/exit", "Exit MitoCode.")
async def _cmd_exit(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Ask the REPL to shut down.

    Args:
        ctx: Unused command context.
        args: Unused extra arguments.

    Returns:
        A result with ``exit_requested=True``.
    """
    del ctx, args
    return CommandResult(exit_requested=True)


@command("quit", "/quit", "Exit MitoCode (same as /exit).")
async def _cmd_quit(ctx: CommandContext, args: List[str]) -> CommandResult:
    """Ask the REPL to shut down, aliasing ``/exit``.

    Args:
        ctx: Unused command context.
        args: Unused extra arguments.

    Returns:
        A result with ``exit_requested=True``.
    """
    del ctx, args
    return CommandResult(exit_requested=True)


__all__ = [
    "COMMANDS",
    "Command",
    "CommandHandler",
    "CommandResult",
    "MAX_AGENT_MESSAGE_CHARS",
    "UNKNOWN_COMMAND_TEMPLATE",
    "command",
    "dispatch",
]
