"""Application entry point for the MitoCode interactive CLI.

:func:`main` is the only place that decides the process exit code.  It parses
arguments, answers ``--version``/``--help`` without a terminal, refuses a non
interactive stdin/stdout, and then hands one run to
:func:`_amain` inside :func:`asyncio.run`.  :func:`run` is the console-script
target and only wraps :func:`main` in ``SystemExit``.

The startup order is a constraint, not a preference:

1. arguments, version/help and the terminal check (no logging yet, so a
   non-interactive invocation never touches the log directory);
2. workspace confirmation (when ``.mitocode/config.json`` is missing) and
   logging takeover, in that order: a declined workspace must not create the
   log directory, and every log lives inside the workspace;
3. provider setup (interactive when the credential file is missing), then the
   console's UI, approver, saved-session selection (``SessionStore`` +
   :func:`~mitocode.cli.sessions.select_session`, which only needs the
   workspace) and :func:`bootstrap`, which resolves the workspace, registers
   tools, starts MCP and restores or creates the main session;
4. state, session manager, autosave task, event pump task and REPL, in that
   order, because the pump must be consuming events before the first turn can
   produce any;
5. ``SIGINT`` is routed to :meth:`Repl.handle_sigint` when the platform supports
   it, then the REPL runs.

Teardown lives in ``finally`` and is equally ordered: the autosave task is
stopped first, then the session manager writes its final forced save, then the
pump task is cancelled and awaited, because it renders into the UI and MCP
shutdown may log; ``mcp.stop()`` runs under :func:`asyncio.shield` so a second
``Ctrl+C`` cannot leave a subprocess behind.  MCP shutdown failures are logged
and never change the exit code.  The ``try`` starts as soon as
:func:`bootstrap` returns, so a banner, state, pump, REPL or task-creation
failure still stops MCP; bootstrap itself rolls back a manager it had already
started.

Errors surface once: configuration and application errors print a red panel and
return ``1``, a ``KeyboardInterrupt`` during startup returns ``130``, and an
unexpected exception is logged with its traceback before the same panel and
``1``.  Nothing here prints a resolved credential.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence

from loguru import logger as loguru_logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from mitocode import MITOCODE_CONFIG_PATH
from mitocode.cli.approval import ApprovalPolicy, CliApprover
from mitocode.cli.args import CliArgs, parse_args, render_help, version_string
from mitocode.cli.bootstrap import Bootstrapped, bootstrap, ensure_workspace
from mitocode.cli.config import (
    ask_provider_settings,
    secret_path,
    write_secret_file,
)
from mitocode.cli.demo import render_session_history
from mitocode.cli.errors import MitoCodeCliConfigError, MitoCodeCliError
from mitocode.cli.logging_setup import configure_logging
from mitocode.cli.pump import EventPump
from mitocode.cli.repl import Repl
from mitocode.cli.sessions import (
    SessionAutosaver,
    SessionLoadError,
    SessionManager,
    SessionStore,
    select_session,
)
from mitocode.cli.state import CliState
from mitocode.cli.theme import MITOCODE_THEME
from mitocode.cli.ui import BannerInfo, CliUI
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError
from mitocode.core.mcp.mcp_manager import McpManager
from mitocode.core.tool.tool_manager import MCP_TOOL_PREFIX

NON_INTERACTIVE_EXIT_CODE = 2
"""Exit code for a run without a terminal; the CLI has no pipe mode."""

INTERRUPTED_EXIT_CODE = 130
"""Exit code for ``Ctrl+C`` during startup, the shell's ``SIGINT`` convention."""

FAILURE_EXIT_CODE = 1
"""Exit code for configuration, application and unexpected failures."""


def _is_interactive() -> bool:
    """Return whether both standard streams are terminals.

    The CLI deliberately has no non-interactive mode, so this is checked before
    logging is configured and before anything is written to the screen.

    Returns:
        ``True`` only when ``sys.stdin`` and ``sys.stdout`` are both TTYs.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def _display_version() -> str:
    """Return the bare version number the banner prefixes with ``v``.

    Returns:
        The package version from :func:`version_string`, without its
        ``"mitocode "`` prefix.
    """
    return version_string().split(" ", 1)[-1]


def _error_console() -> Console:
    """Build a console for startup failures, which belong on stderr.

    Returns:
        A themed console writing to :data:`sys.stderr`.
    """
    return Console(file=sys.stderr, theme=MITOCODE_THEME)


def _report_error(message: str, *, title: str = "Error") -> None:
    """Print one red panel describing a startup failure.

    Args:
        message: Readable explanation; callers guarantee it holds no secret.
        title: Panel heading, ``Error`` by default.
    """
    _error_console().print(
        Panel(
            Text(message),
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            expand=False,
        )
    )


def _install_sigint_handler(loop: Any, repl: Repl) -> bool:
    """Route ``SIGINT`` to the REPL when the event loop supports it.

    ``asyncio.run`` turns ``Ctrl+C`` into ``KeyboardInterrupt`` on some
    platforms; installing the handler as well lets a running turn be cancelled
    cooperatively instead of killing the process.

    Args:
        loop: The running event loop.
        repl: Receiver of the signal callback.

    Returns:
        ``True`` when the handler was installed, ``False`` when the platform or
        the loop implementation cannot install one.
    """
    add_handler = getattr(loop, "add_signal_handler", None)
    if add_handler is None:
        return False
    try:
        add_handler(signal.SIGINT, repl.handle_sigint)
    except (NotImplementedError, RuntimeError, ValueError):
        loguru_logger.debug("SIGINT handler is not supported on this platform")
        return False
    return True


def _banner_info(boot: Bootstrapped) -> BannerInfo:
    """Assemble the facts the startup banner renders.

    Built-in and MCP tool counts are separated by the ``mcp__`` name prefix the
    registry uses, so the banner's two numbers cannot double count.

    Args:
        boot: Everything :func:`bootstrap` started.

    Returns:
        The banner's display facts.
    """
    tool_names = list(boot.tool_manager.tools_map())
    builtin_count = sum(
        1 for name in tool_names if not name.startswith(MCP_TOOL_PREFIX)
    )
    mcp_states = {}
    if boot.mcp_manager is not None:
        mcp_states = {
            name: status.state for name, status in boot.mcp_manager.status().items()
        }
    return BannerInfo(
        version=_display_version(),
        model=boot.provider_config.model,
        workspace=boot.resolution.workspace.path,
        tool_count=builtin_count,
        mcp=mcp_states,
    )


async def _stop_mcp(manager: Optional[McpManager]) -> None:
    """Stop the MCP manager without letting its failure end the run.

    The stop coroutine is shielded because it releases subprocesses: a
    cancellation racing with shutdown must not leave them orphaned.  Errors are
    logged only; the process has already finished its user-visible work.

    Args:
        manager: The live manager, or ``None`` when MCP is disabled.
    """
    if manager is None:
        return
    try:
        await asyncio.shield(manager.stop())
    except Exception:  # pylint: disable=broad-exception-caught
        loguru_logger.exception("MCP shutdown failed")


async def _amain(args: CliArgs, log_path: Any, console: Console) -> int:
    """Start the components in the design's order and run the REPL.

    Args:
        args: Parsed command line arguments.
        log_path: Log file reported by :func:`configure_logging`; stored on the
            state so ``/log`` can show it and printed on exit under ``--debug``.
        console: Themed console the UI renders through.

    Returns:
        The REPL's exit code, ``0`` for every clean exit path.

    Raises:
        MitoCodeCliError: Configuration is invalid.
        MitoCodeBaseError: A workspace, provider or MCP error aborted startup.
        KeyboardInterrupt: The user interrupted startup.
    """
    ui = CliUI(
        console,
        verbose=args.verbose,
        history_path=Path(args.workspace) / MITOCODE_CONFIG_PATH / "history",
    )
    approver = CliApprover(ui, ApprovalPolicy(args.permission_mode))

    # Saved-session selection runs before bootstrap: it only needs the
    # workspace to exist, and the choice decides what bootstrap restores.
    store = SessionStore(Path(args.workspace))
    summaries, skipped = store.list_sessions()
    if skipped > 0:
        ui.warn(f"{skipped} session file(s) could not be read and were skipped.")

    restore: Optional[dict] = None
    # Zero or one saved session needs no interactive choice: the panel (and
    # its prompting, hence EOF/interrupt paths) only exists from two on.
    if len(summaries) <= 1:
        choice = summaries[0].session_id if summaries else "new"
    else:
        try:
            choice = await select_session(ui, summaries)
        except EOFError:
            return FAILURE_EXIT_CODE
    if choice != "new":
        try:
            restore = store.load(choice)
        except SessionLoadError as exc:
            ui.warn(f"Saved session could not be loaded; starting a new one. ({exc})")
            restore = None

    boot = await bootstrap(args, ui, approver, restore=restore)

    pump_task: Optional[asyncio.Task[None]] = None
    state: Optional[CliState] = None
    manager: Optional[SessionManager] = None
    autosaver: Optional[SessionAutosaver] = None
    try:
        ui.show_banner(_banner_info(boot))

        state = CliState(
            args=args,
            workspace=boot.resolution.workspace,
            tool_manager=boot.tool_manager,
            provider=boot.provider,
            orchestrator=boot.orchestrator,
            mcp_manager=boot.mcp_manager,
            tool_context=boot.tool_context,
            skills_manager=boot.skills_manager,
            ui=ui,
            approver=approver,
            log_path=log_path,
            main_id=boot.main_session.agent_id,
            mcp_error=boot.mcp_error,
            verbose=args.verbose,
        )
        manager = SessionManager(
            store,
            boot.orchestrator,
            ui,
            state=state,
            provider=boot.provider,
            workspace=str(Path(args.workspace).resolve()),
            # A tool that read the credential file would otherwise put the API
            # key into a session record, which is read back and echoed at
            # startup.  The same literal the orchestrator scrubs from error
            # messages is scrubbed from what is written to disk.
            redact_secrets=(boot.provider_config.api_key,),
        )
        state.session_manager = manager
        if boot.restore_succeeded and restore is not None:
            manager.attach_record(restore)
            # A restored session is live in memory but invisible on the fresh
            # terminal; echoing the transcript is what makes the continuation
            # legible instead of looking like an empty conversation.
            render_session_history(
                ui,
                boot.main_session,
                title=manager.current_title,
                tool_manager=boot.tool_manager,
                root=Path(args.workspace),
            )
            if manager.current_title == "New Session":
                manager.maybe_generate_title()
        else:
            await manager.start_new_session(boot.main_session)
        autosaver = SessionAutosaver(manager)
        autosaver.start()

        pump = EventPump(boot.orchestrator.event_queue, ui, state)
        state.pump = pump
        pump_task = asyncio.create_task(pump.run())
        repl = Repl(state, ui, pump, approver)
        _install_sigint_handler(asyncio.get_running_loop(), repl)

        return await repl.run()
    finally:
        # The teardown steps that stop the autosaver and write the final save
        # are allowed to fail, but they must never take the cleanup below with
        # them: a cancelled shutdown (a second Ctrl+C) would otherwise leave
        # the pump running and the MCP subprocess alive.  The cleanup is in an
        # inner ``finally`` so it runs whatever happens above it.
        try:
            if autosaver is not None:
                await autosaver.stop()
            if manager is not None:
                await manager.shutdown()
        except BaseException:  # pylint: disable=broad-exception-caught
            loguru_logger.exception("Session teardown step failed")
        finally:
            if pump_task is not None:
                pump_task.cancel()
                await asyncio.gather(pump_task, return_exceptions=True)
            await _stop_mcp(boot.mcp_manager)
            if state is not None:
                ui.note(
                    f"Session ended · {state.turn_count} turns "
                    f"· ~{state.estimate_tokens()} tokens"
                )
            if args.debug:
                ui.note(f"Log file: {log_path}")


def _confirm_workspace(console: Console, workspace: Path) -> bool:
    """Ask whether a missing workspace configuration should be created.

    Args:
        console: Console the panel and the question are printed to.
        workspace: Workspace root the CLI was pointed at.

    Returns:
        ``True`` when the user agrees to create the workspace.
    """
    config_dir = Path(workspace) / MITOCODE_CONFIG_PATH
    console.print(
        Panel(
            Text.assemble(
                ("This directory has no MitoCode workspace yet.\n\n", ""),
                ("Workspace state will live in ", "mito.muted"),
                (str(config_dir), "mito.accent"),
                (".", "mito.muted"),
            ),
            title="Workspace setup",
            border_style="mito.accent",
            expand=False,
        )
    )
    return Confirm.ask("Create the workspace here?", console=console, default=True)


def _ensure_provider_settings(workspace: Path, console: Console) -> None:
    """Create the provider credential file when one is missing.

    Only a missing path triggers the prompts. An existing but unusable entry is
    left untouched: overwriting it could destroy a credential the user can
    still repair by hand, and the resolver reports the specific problem.

    Args:
        workspace: Workspace root that receives the credential file.
        console: Console used for the setup panel and prompts.

    Raises:
        EOFError: The user cancelled the prompts with ``Ctrl+D``.
        MitoCodeCliConfigError: The file cannot be written.
    """
    path = secret_path(workspace)
    if path.exists() or path.is_symlink():
        return
    values = ask_provider_settings(console)
    written = write_secret_file(workspace, values)
    loguru_logger.debug("Provider configuration written to {}", written)


def _start(argv: Sequence[str] | None) -> int:
    """Parse arguments and run one session.

    The workspace check happens before logging is configured, so declining it
    leaves nothing on disk. Provider setup runs after logging because the
    credential file is inside the same workspace as the log; cancelling it
    therefore leaves only the workspace's own ``.mitocode/logs`` behind. There
    is no home-directory fallback.

    Args:
        argv: Argument list without the program name, or ``None`` for
            :data:`sys.argv`.

    Returns:
        ``0`` for version, help or a clean session, ``1`` when the user
        declines the workspace or cancels provider setup, or ``2`` without a
        terminal.

    Raises:
        MitoCodeCliConfigError: The workspace cannot be created, the log
            directory cannot be created, or the provider file cannot be
            written.
    """
    args = parse_args(argv)

    if args.show_version:
        Console(theme=MITOCODE_THEME).print(version_string())
        return 0
    if args.show_help:
        render_help(Console(theme=MITOCODE_THEME))
        return 0

    if not _is_interactive():
        _report_error(
            "mitocode needs an interactive terminal: stdin and stdout "
            "must both be a TTY.",
            title="No interactive terminal",
        )
        return NON_INTERACTIVE_EXIT_CODE

    console = Console(theme=MITOCODE_THEME)
    workspace_path = Path(args.workspace)
    if not ensure_workspace(
        workspace_path, confirm=lambda path: _confirm_workspace(console, path)
    ):
        console.print(Text("  Workspace was not created; exiting.", style="mito.muted"))
        return FAILURE_EXIT_CODE

    log_dir = workspace_path / MITOCODE_CONFIG_PATH / "logs"
    try:
        log_path = configure_logging(debug=args.debug, log_dir=log_dir)
    except OSError as exc:
        raise MitoCodeCliConfigError(
            f"Cannot create the log directory at '{log_dir}': "
            f"{exc.strerror or 'I/O error'}."
        ) from exc

    try:
        _ensure_provider_settings(workspace_path, console)
    except EOFError:
        console.print(
            Text("  Provider setup was cancelled; exiting.", style="mito.muted")
        )
        return FAILURE_EXIT_CODE

    return asyncio.run(_amain(args, log_path, console))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Argument list without the program name; ``None`` reads
            :data:`sys.argv`.

    Returns:
        ``0`` on a clean session, ``1`` for a configuration/application
        failure, ``2`` without an interactive terminal and ``130`` when
        startup is interrupted.
    """
    try:
        return _start(argv)
    except KeyboardInterrupt:
        return INTERRUPTED_EXIT_CODE
    except (MitoCodeCliError, MitoCodeBaseError) as exc:
        loguru_logger.debug("Startup failed: {}", type(exc).__name__)
        _report_error(str(exc))
        return FAILURE_EXIT_CODE
    except Exception as exc:  # pylint: disable=broad-exception-caught
        loguru_logger.exception("Unexpected CLI failure")
        _report_error(str(exc), title="Unexpected error")
        return FAILURE_EXIT_CODE


def run() -> NoReturn:
    """Run the CLI as the ``mitocode`` console script.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = ["main", "run"]
