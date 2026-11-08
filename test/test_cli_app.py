"""Tests for the CLI application entry point: orchestration, signals and exit.

Every test replaces the real terminal, provider and REPL with fakes: the console
is captured by ``capsys``, :func:`mitocode.cli.app._is_interactive` is
monkeypatched instead of the real streams, and ``bootstrap``/``EventPump``/
``Repl``/``configure_logging`` are replaced at the application module.  No test
opens a terminal, raises a real signal or reaches the network.
"""

import asyncio
import json
import os
import signal
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli import app as app_module
from mitocode.cli.args import CliArgs
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.cli.sessions import SessionStore
from mitocode.cli.theme import MITOCODE_THEME
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace

# The startup-consent/setup gates are bypassed for every test in this module;
# the two tests that exercise the real functions resolve the original
# implementation through this alias before the autouse fixture patches it.
_REAL_ENSURE_PROVIDER_SETTINGS = app_module._ensure_provider_settings


def _args(**overrides):
    """Build parsed arguments, defaulting the rest to :class:`CliArgs` values.

    Args:
        **overrides: Fields to replace on the default :class:`CliArgs`.

    Returns:
        The parsed arguments.
    """
    return CliArgs(**overrides)


@pytest.fixture(autouse=True)
def _bypass_workspace_and_provider_setup(monkeypatch):
    """Skip the interactive consent/setup gates for every startup test.

    The interactive prompts are covered in their own modules; here they would
    otherwise call ``Confirm.ask`` and read the terminal.
    """
    monkeypatch.setattr(
        app_module,
        "ensure_workspace",
        lambda workspace_path, *, confirm: True,
    )
    monkeypatch.setattr(
        app_module,
        "_ensure_provider_settings",
        lambda workspace, console: None,
    )


def _boot(
    tmp_path, *, stop=None, mcp_status=None
):  # pylint: disable=too-many-arguments
    """Build a ``Bootstrapped``-shaped fake for the application.

    Args:
        tmp_path: Directory used as the workspace root.
        stop: Async MCP ``stop`` implementation; defaults to a no-op.
        mcp_status: Mapping returned by the fake manager's ``status()``.

    Returns:
        A namespace carrying every attribute :func:`app.main` consumes.
    """

    async def _noop():
        """Do nothing."""

    statuses = {
        name: (
            value
            if hasattr(value, "state")
            else SimpleNamespace(state=value, error="", tool_count=0)
        )
        for name, value in (mcp_status or {}).items()
    }
    workspace = Workspace(tmp_path, "app-test", 1)
    manager = (
        None
        if mcp_status is None and stop is None
        else SimpleNamespace(
            status=lambda: statuses, stop=stop if stop is not None else _noop
        )
    )
    return SimpleNamespace(
        args=_args(workspace=tmp_path),
        resolution=SimpleNamespace(workspace=workspace, migrated_from=None),
        provider_config=SimpleNamespace(
            model="model-x",
            base_url=None,
            timeout=60.0,
            source="flags",
            api_key="sk-test-key",
        ),
        tool_manager=ToolManager(),
        tool_context=SimpleNamespace(),
        provider=SimpleNamespace(default_model="model-x", model_list=["model-x"]),
        orchestrator=SimpleNamespace(
            event_queue=asyncio.Queue(), get_session=lambda agent_id: None
        ),
        main_session=SimpleNamespace(agent_id=0),
        mcp_manager=manager,
        skills_manager=None,
        mcp_error="",
    )


def _session_record(
    session_id, *, title="Saved one", updated_at="2026-11-07T10:00:00Z"
):
    """Build a minimal valid session record for startup-selection tests.

    Args:
        session_id: Disk id the record pretends to be stored under.
        title: Record title.
        updated_at: ISO timestamp controlling the panel sort order.

    Returns:
        A schema-shaped record with one empty main agent.
    """
    return {
        "schema_version": 1,
        "session_id": session_id,
        "title": title,
        "created_at": "2026-11-07T09:00:00Z",
        "updated_at": updated_at,
        "workspace": "/tmp/w",
        "model": "model-x",
        "main_agent_id": 0,
        "agents": [
            {
                "agent_id": 0,
                "name": "Main",
                "description": "Interactive session",
                "parent_id": None,
                "system_prompt": "",
                "messages": [],
            }
        ],
    }


def _write_record(tmp_path, record):
    """Write a session record into *tmp_path*'s sessions directory.

    Args:
        tmp_path: Workspace root.
        record: The schema document to store.

    Returns:
        The written path.
    """
    sessions_dir = tmp_path / ".mitocode" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{record['session_id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _serializable_boot(tmp_path, **overrides):
    """Build a fake boot whose orchestrator can be serialized.

    The real :class:`~mitocode.cli.sessions.SessionManager` reads the tree
    through ``get_tree_snapshot``/``get_session``; this boot provides both so
    a test can exercise the real manager and store end to end.

    Args:
        tmp_path: Directory used as the workspace root.
        **overrides: Passed through to :func:`_boot`.

    Returns:
        The boot namespace with a serializable orchestrator.
    """
    boot = _boot(tmp_path, **overrides)
    session = SimpleNamespace(agent_id=0, system_prompt="", messages=[])
    boot.orchestrator = SimpleNamespace(
        event_queue=asyncio.Queue(),
        get_tree_snapshot=lambda: [
            {
                "id": 0,
                "name": "Main",
                "description": "Interactive session",
                "parent_id": None,
                "sub_agent": [],
            }
        ],
        get_session=lambda agent_id: session if agent_id == 0 else None,
    )
    return boot


def _patch_store(monkeypatch, tmp_path):
    """Point the app's session store at *tmp_path*.

    ``main([])`` parses a default workspace, so without this patch the
    startup selection would read and write the repository's own
    ``.mitocode/sessions`` directory.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Workspace root the store is rooted at.
    """
    real_store = SessionStore(tmp_path)
    monkeypatch.setattr(app_module, "SessionStore", lambda workspace: real_store)


def _patch_session_fakes(monkeypatch, boot, timeline):
    """Replace SessionManager/SessionAutosaver with recording fakes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        boot: Fake boot; the manager instance is attached as
            ``boot.session_manager`` for assertions.
        timeline: Event list the fakes append their lifecycle to.
    """

    class FakeManager:
        """Recording stand-in for :class:`SessionManager`."""

        def __init__(
            self,
            store,
            orchestrator,
            ui,
            *,
            state,
            provider,
            workspace,
            clock=None,
            redact_secrets=(),
        ):
            timeline.append("manager-created")
            boot.redacted_secrets = tuple(redact_secrets)
            boot.session_manager = self
            self.store = store
            self.state = state
            self.attached = None
            self.started_new = None
            self.title_calls = 0
            self.shutdown_calls = 0

        @property
        def current_session_id(self):
            return "fake-session-id"

        @property
        def current_title(self):
            return "New Session"

        async def save_current(self, *, force=False, report_failure=False):
            return None

        async def start_new_session(self, main_session):
            self.started_new = main_session
            timeline.append("start-new")

        def attach_record(self, record):
            self.attached = record
            timeline.append("attach_record")

        def maybe_generate_title(self):
            self.title_calls += 1

        async def shutdown(self):
            self.shutdown_calls += 1
            timeline.append("manager-shutdown")

    class FakeAutosaver:
        """Recording stand-in for :class:`SessionAutosaver`."""

        def __init__(self, manager, interval=30.0):
            timeline.append("autosaver-created")
            self.manager = manager

        def start(self):
            timeline.append("autosaver-start")

        async def stop(self):
            timeline.append("autosaver-stop")

    monkeypatch.setattr(app_module, "SessionManager", FakeManager)
    monkeypatch.setattr(app_module, "SessionAutosaver", FakeAutosaver)


def _install_common_fakes(
    monkeypatch, tmp_path, boot, events=None, *, session_fakes=True
):
    """Replace the application's collaborators with recording fakes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Directory used for the log file and MCP recording.
        boot: Fake returned by :func:`bootstrap`.
        events: List that records lifecycle order; a new one is created when
            ``None``.
        session_fakes: Whether to patch ``SessionManager``/``SessionAutosaver``
            with recording fakes; ``False`` leaves the real ones in place.

    Returns:
        The event list in recording order.
    """
    timeline = [] if events is None else events

    async def fake_bootstrap(args, ui, approver, restore=None):
        timeline.append("bootstrap")
        boot.received_restore = restore
        boot.restore_succeeded = restore is not None
        return boot

    class FakePump:
        """Pump that records its own cancellation."""

        def __init__(self, queue, ui, state):
            timeline.append("pump-created")

        async def run(self):
            timeline.append("pump-started")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                timeline.append("pump-cancelled")
                raise

    class FakeRepl:
        """REPL that records its state and returns immediately."""

        def __init__(self, state, ui, pump, approver):
            timeline.append("repl-created")
            self.state = state
            self.ui = ui
            self.pump = pump
            self.approver = approver

        async def run(self):
            timeline.append("repl-started")
            # Yield once so the already-created pump task actually starts
            # before the fake REPL returns.
            await asyncio.sleep(0)
            return 0

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(app_module, "EventPump", FakePump)
    monkeypatch.setattr(app_module, "Repl", FakeRepl)
    _patch_store(monkeypatch, tmp_path)
    if session_fakes:
        _patch_session_fakes(monkeypatch, boot, timeline)
    # Signal installation is covered on its own; keeping it out of these tests
    # leaves pytest's own SIGINT handling untouched.
    monkeypatch.setattr(app_module, "_install_sigint_handler", lambda loop, repl: False)
    return timeline


def test_version_does_not_need_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(app_module, "_is_interactive", lambda: False)
    assert app_module.main(["--version"]) == 0
    assert "mitocode" in capsys.readouterr().out.lower()


def test_help_does_not_need_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(app_module, "_is_interactive", lambda: False)
    assert app_module.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "usage:" in output.lower()
    assert "--workspace" in output


def test_non_tty_is_rejected_before_logging(monkeypatch, capsys):
    logged = []
    monkeypatch.setattr(app_module, "_is_interactive", lambda: False)
    monkeypatch.setattr(app_module, "configure_logging", lambda **kw: logged.append(kw))
    assert app_module.main([]) == 2
    assert logged == []
    assert "interactive terminal" in capsys.readouterr().err.lower()


def test_main_delegates_to_bootstrap_and_repl(monkeypatch, tmp_path):
    stopped = []

    async def fake_stop():
        stopped.append("stop")

    timeline = _install_common_fakes(
        monkeypatch, tmp_path, _boot(tmp_path, stop=fake_stop)
    )

    assert app_module.main([]) == 0
    assert stopped == ["stop"]
    # The session manager and autosaver are wired right after bootstrap, the
    # pump task is created before the REPL runs, and teardown stops the
    # autosaver, then the manager, then the pump, then MCP.
    assert timeline == [
        "bootstrap",
        "manager-created",
        "start-new",
        "autosaver-created",
        "autosaver-start",
        "pump-created",
        "repl-created",
        "repl-started",
        "pump-started",
        "autosaver-stop",
        "manager-shutdown",
        "pump-cancelled",
    ]


def test_state_is_built_from_bootstrapped_components(monkeypatch, tmp_path):
    boot = _boot(tmp_path, mcp_status={})
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot)
    captured = {}

    class CapturingRepl:
        """REPL that records the state it received."""

        def __init__(self, state, ui, pump, approver):
            captured["state"] = state
            captured["ui"] = ui
            captured["pump"] = pump
            captured["approver"] = approver

        async def run(self):
            return 0

    monkeypatch.setattr(app_module, "Repl", CapturingRepl)

    assert app_module.main(["--verbose"]) == 0
    state = captured["state"]
    assert isinstance(state, app_module.CliState)
    assert state.args.verbose is True
    assert state.main_id == 0
    assert state.log_path == tmp_path / "cli.log"
    assert state.mcp_manager is boot.mcp_manager
    assert captured["approver"].policy.value == "ask"
    assert "bootstrap" in timeline


def test_debug_prints_the_log_path_on_exit(monkeypatch, tmp_path, capsys):
    _install_common_fakes(monkeypatch, tmp_path, _boot(tmp_path))

    assert app_module.main(["--debug"]) == 0
    output = capsys.readouterr().out
    assert "Log file" in output
    # Rich may wrap a long path across lines; join them before comparing.
    assert str(tmp_path / "cli.log") in output.replace("\n", "")


def test_cancelling_the_pump_precedes_stopping_mcp(monkeypatch, tmp_path):
    events = []

    async def fake_stop():
        events.append("mcp-stopped")

    _install_common_fakes(
        monkeypatch, tmp_path, _boot(tmp_path, stop=fake_stop), events=events
    )

    assert app_module.main([]) == 0
    assert events.index("pump-cancelled") < events.index("mcp-stopped")


def test_a_failed_restore_is_reported_to_the_user(monkeypatch, tmp_path, capsys):
    """恢复失败必须告知用户，否则用户以为选中的会话已经恢复."""
    record = _session_record("20261107T100000Z-aaaaaaaa")
    _write_record(tmp_path, record)

    boot = _boot(tmp_path)
    boot.restore_error = "parent chain cycles"
    _install_common_fakes(monkeypatch, tmp_path, boot)

    async def fake_bootstrap(args, ui, approver, restore=None):
        # The restore was attempted and failed.
        boot.restore_succeeded = False
        return boot

    monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)

    assert app_module.main([]) == 0

    output = capsys.readouterr().out
    # The panel wraps, so the text is checked without its line breaks.
    flattened = " ".join(output.lower().split())
    assert "could not be restored" in flattened
    assert "parent chain cycles" in flattened


def test_ctrl_d_at_the_workspace_prompt_reports_it(monkeypatch, tmp_path, capsys):
    """首次工作区确认按 Ctrl+D 必须给出可读的信息，而不是空的 Unexpected error."""

    def _ensure(workspace_path, *, confirm):
        # The real implementation asks the user; Ctrl+D ends that prompt.
        return confirm(workspace_path)

    def _cancel(console, path):
        raise EOFError()

    monkeypatch.setattr(app_module, "ensure_workspace", _ensure)
    monkeypatch.setattr(app_module, "_confirm_workspace", _cancel)
    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )

    assert app_module.main([]) == 1

    output = capsys.readouterr().out
    assert "cancelled" in output.lower(), (
        "the user must be told the setup was cancelled; an empty panel tells "
        "them nothing"
    )
    assert "Unexpected error" not in output


def test_unreadable_session_directory_reports_and_exits(monkeypatch, tmp_path, capsys):
    """会话目录无法读取时必须提示用户并退出，而不是当成"没有存档"."""
    from mitocode.cli.sessions import SessionLoadError

    def _boom(self):
        raise SessionLoadError(f"the session directory '{tmp_path}' is unreadable")

    monkeypatch.setattr(app_module.SessionStore, "list_sessions", _boom, raising=False)
    _install_common_fakes(monkeypatch, tmp_path, _boot(tmp_path))

    assert app_module.main([]) == 1

    output = capsys.readouterr().out
    assert "unreadable" in output


def test_the_manager_receives_the_api_key_to_scrub_from_records(monkeypatch, tmp_path):
    """会话记录可能包含工具读到的凭据，manager 必须知道要脱敏的字面量."""
    boot = _boot(tmp_path)
    _install_common_fakes(monkeypatch, tmp_path, boot)

    app_module.main([])

    assert boot.redacted_secrets == ("sk-test-key",), (
        "the manager must be told which literal to scrub before writing a " "record"
    )


def test_mcp_stop_failure_is_not_fatal(monkeypatch, tmp_path):
    attempts = []

    async def failing_stop():
        attempts.append("stop")
        raise RuntimeError("transport already closed")

    _install_common_fakes(monkeypatch, tmp_path, _boot(tmp_path, stop=failing_stop))

    assert app_module.main([]) == 0
    assert attempts == ["stop"]


def test_manager_shutdown_failure_still_cancels_pump_and_stops_mcp(
    monkeypatch, tmp_path
):
    """关机时 manager.shutdown 抛异常，pump 与 MCP 仍必须被清理."""
    events = []

    async def fake_stop():
        events.append("mcp-stopped")

    boot = _boot(tmp_path, stop=fake_stop)
    _install_common_fakes(monkeypatch, tmp_path, boot, events=events)

    class ExplodingManager:
        """Manager whose shutdown fails the way a second Ctrl+C would."""

        def __init__(self, *args, **kwargs):
            self.shutdown_calls = 0

        @property
        def current_session_id(self):
            return "fake-session-id"

        @property
        def current_title(self):
            return "New Session"

        async def save_current(self, *, force=False, report_failure=False):
            return None

        async def start_new_session(self, main_session):
            return None

        def attach_record(self, record):
            return None

        def maybe_generate_title(self):
            return None

        async def shutdown(self):
            self.shutdown_calls += 1
            raise RuntimeError("shutdown blew up")

    monkeypatch.setattr(app_module, "SessionManager", ExplodingManager)

    assert app_module.main([]) == 0
    # The MCP server must be stopped even though the shutdown step failed:
    # otherwise a subprocess is left behind.
    assert "mcp-stopped" in events
    assert "pump-cancelled" in events


def test_autosaver_failure_still_cancels_pump_and_stops_mcp(monkeypatch, tmp_path):
    """关机时 autosaver.stop 抛异常，pump 与 MCP 仍必须被清理."""
    events = []

    async def fake_stop():
        events.append("mcp-stopped")

    boot = _boot(tmp_path, stop=fake_stop)
    _install_common_fakes(monkeypatch, tmp_path, boot, events=events)

    class ExplodingAutosaver:
        """Autosaver whose stop fails."""

        def __init__(self, manager, interval=30.0):
            self.manager = manager

        def start(self):
            return None

        async def stop(self):
            raise RuntimeError("autosaver blew up")

    monkeypatch.setattr(app_module, "SessionAutosaver", ExplodingAutosaver)

    assert app_module.main([]) == 0
    assert "mcp-stopped" in events
    assert "pump-cancelled" in events


def test_shutdown_cancellation_still_cancels_pump_and_stops_mcp(monkeypatch, tmp_path):
    """第二次 Ctrl+C 造成的 CancelledError 也不得跳过 pump 与 MCP 的清理."""
    events = []

    async def fake_stop():
        events.append("mcp-stopped")

    boot = _boot(tmp_path, stop=fake_stop)
    _install_common_fakes(monkeypatch, tmp_path, boot, events=events)

    class CancellingManager:
        """Manager whose shutdown is cancelled, as a second Ctrl+C does."""

        def __init__(self, *args, **kwargs):
            pass

        @property
        def current_session_id(self):
            return "fake-session-id"

        @property
        def current_title(self):
            return "New Session"

        async def save_current(self, *, force=False, report_failure=False):
            return None

        async def start_new_session(self, main_session):
            return None

        def attach_record(self, record):
            return None

        def maybe_generate_title(self):
            return None

        async def shutdown(self):
            raise asyncio.CancelledError()

    monkeypatch.setattr(app_module, "SessionManager", CancellingManager)

    assert app_module.main([]) == 0
    assert "mcp-stopped" in events
    assert "pump-cancelled" in events


def test_missing_mcp_manager_is_skipped(monkeypatch, tmp_path):
    boot = _boot(tmp_path)
    boot.mcp_manager = None
    _install_common_fakes(monkeypatch, tmp_path, boot)

    assert app_module.main([]) == 0


def test_banner_has_model_workspace_and_mcp_state(monkeypatch, tmp_path, capsys):
    boot = _boot(tmp_path, mcp_status={"fs": "connected"})
    _install_common_fakes(monkeypatch, tmp_path, boot)

    assert app_module.main([]) == 0
    output = capsys.readouterr().out
    assert "MitoCode" in output
    assert "model-x" in output
    assert "app-test" not in output  # the workspace label is the path, not the name
    assert "connected" in output


def test_config_error_prints_a_panel_and_returns_1(monkeypatch, tmp_path, capsys):
    async def failing_bootstrap(args, ui, approver, restore=None):
        raise MitoCodeCliConfigError("provider rejected the request")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", failing_bootstrap)
    _patch_store(monkeypatch, tmp_path)

    assert app_module.main([]) == 1
    error = capsys.readouterr().err
    assert "provider rejected the request" in error


def test_base_error_prints_a_panel_and_returns_1(monkeypatch, tmp_path, capsys):
    async def failing_bootstrap(args, ui, approver, restore=None):
        raise MitoCodeBaseError("workspace is too old")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", failing_bootstrap)
    _patch_store(monkeypatch, tmp_path)

    assert app_module.main([]) == 1
    assert "workspace is too old" in capsys.readouterr().err


def test_keyboard_interrupt_returns_130(monkeypatch, tmp_path):
    async def interrupted_bootstrap(args, ui, approver, restore=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", interrupted_bootstrap)
    _patch_store(monkeypatch, tmp_path)

    assert app_module.main([]) == 130


def test_unexpected_exception_is_reported_and_returns_1(monkeypatch, tmp_path, capsys):
    async def exploding_bootstrap(args, ui, approver, restore=None):
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", exploding_bootstrap)
    _patch_store(monkeypatch, tmp_path)

    assert app_module.main([]) == 1
    assert "unexpected boom" in capsys.readouterr().err


def test_sigint_handler_is_installed_when_supported():
    class FakeLoop:
        """Loop recording signal handler registrations."""

        def __init__(self):
            self.calls = []

        def add_signal_handler(self, sig, callback):
            self.calls.append((sig, callback))

    repl = SimpleNamespace(handle_sigint=lambda: None)
    loop = FakeLoop()

    assert app_module._install_sigint_handler(loop, repl) is True
    assert loop.calls == [(signal.SIGINT, repl.handle_sigint)]


def test_sigint_handler_is_skipped_without_support():
    repl = SimpleNamespace(handle_sigint=lambda: None)

    class UnsupportedLoop:
        """Loop whose platform cannot install signal handlers."""

        def add_signal_handler(self, sig, callback):
            raise NotImplementedError

    assert app_module._install_sigint_handler(SimpleNamespace(), repl) is False
    assert app_module._install_sigint_handler(UnsupportedLoop(), repl) is False


def test_is_interactive_requires_both_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))
    assert app_module._is_interactive() is False

    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    assert app_module._is_interactive() is True


def test_banner_failure_still_stops_mcp(monkeypatch, tmp_path, capsys):
    """A post-bootstrap failure must release the MCP servers it started."""
    stopped = []
    timeline = []

    async def fake_stop():
        stopped.append("stop")

    boot = _boot(tmp_path, stop=fake_stop)

    class ExplodingBannerUI:
        """UI whose banner rendering fails after bootstrap succeeded."""

        def __init__(self, console, *, history_path=None, verbose=False):
            self.console = console
            self.verbose = verbose
            self.history_path = history_path

        def show_banner(self, info):
            raise RuntimeError("banner failed")

    async def fake_bootstrap(args, ui, approver, restore=None):
        return boot

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(app_module, "CliUI", ExplodingBannerUI)
    _patch_store(monkeypatch, tmp_path)
    _patch_session_fakes(monkeypatch, boot, timeline)

    assert app_module.main([]) == 1
    assert stopped == ["stop"]
    assert "banner failed" in capsys.readouterr().err


class _RecordingPumpTask:
    """Awaitable stand-in for the pump task that records cancellation."""

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __await__(self):
        async def _finished():
            return None

        return _finished().__await__()


def test_repl_constructor_failure_cancels_pump_and_stops_mcp(monkeypatch, tmp_path):
    """A failure while wiring the REPL leaves no task and no live MCP."""
    stopped = []
    created = []
    timeline = []

    async def fake_stop():
        stopped.append("stop")

    boot = _boot(tmp_path, stop=fake_stop)

    class FakePump:
        """Pump that is never started because the REPL cannot be built."""

        def __init__(self, *args):
            pass

        async def run(self):
            await asyncio.Event().wait()

    class ExplodingRepl:
        """REPL whose constructor fails after the pump task exists."""

        def __init__(self, *args):
            raise RuntimeError("repl wiring failed")

    def fake_create_task(coro):
        coro.close()
        task = _RecordingPumpTask()
        created.append(task)
        return task

    async def fake_bootstrap(args, ui, approver, restore=None):
        boot.received_restore = restore
        boot.restore_succeeded = restore is not None
        return boot

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        app_module, "configure_logging", lambda **kw: tmp_path / "cli.log"
    )
    monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(app_module, "EventPump", FakePump)
    monkeypatch.setattr(app_module, "Repl", ExplodingRepl)
    monkeypatch.setattr(app_module.asyncio, "create_task", fake_create_task)
    _patch_store(monkeypatch, tmp_path)
    _patch_session_fakes(monkeypatch, boot, timeline)

    assert app_module.main([]) == 1
    assert stopped == ["stop"]
    assert len(created) == 1
    assert created[0].cancelled


def test_first_run_prompts_and_writes_provider_file(monkeypatch, tmp_path):
    """A missing provider file is collected through the setup prompts."""

    recorded = {}

    def fake_ask(console):
        recorded.update(values={"api_key": "k", "model": "m"})
        return recorded["values"]

    def fake_write(workspace, values):
        recorded["workspace"] = workspace
        recorded["written"] = values
        return tmp_path / ".mitocode" / "secret" / "provider.json"

    monkeypatch.setattr(app_module, "ask_provider_settings", fake_ask)
    monkeypatch.setattr(app_module, "write_secret_file", fake_write)
    monkeypatch.setattr(
        app_module, "_ensure_provider_settings", _REAL_ENSURE_PROVIDER_SETTINGS
    )

    app_module._ensure_provider_settings(
        tmp_path, Console(file=StringIO(), force_terminal=False)
    )

    assert recorded["workspace"] == tmp_path
    assert recorded["written"] == {"api_key": "k", "model": "m"}


def test_existing_provider_file_is_left_untouched(monkeypatch, tmp_path):
    secret = tmp_path / ".mitocode" / "secret" / "provider.json"
    secret.parent.mkdir(parents=True)
    secret.write_text('{"api_key":"k","model":"m"}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        app_module, "ask_provider_settings", lambda console: calls.append(console)
    )
    monkeypatch.setattr(
        app_module, "write_secret_file", lambda workspace, values: calls.append(values)
    )
    monkeypatch.setattr(
        app_module, "_ensure_provider_settings", _REAL_ENSURE_PROVIDER_SETTINGS
    )

    app_module._ensure_provider_settings(
        tmp_path, Console(file=StringIO(), force_terminal=False)
    )

    assert calls == []


def test_non_file_provider_path_is_left_untouched(monkeypatch, tmp_path):
    secret = tmp_path / ".mitocode" / "secret" / "provider.json"
    secret.parent.mkdir(parents=True)
    secret.mkdir()
    calls = []
    monkeypatch.setattr(
        app_module, "ask_provider_settings", lambda console: calls.append(console)
    )
    monkeypatch.setattr(
        app_module, "write_secret_file", lambda workspace, values: calls.append(values)
    )
    monkeypatch.setattr(
        app_module, "_ensure_provider_settings", _REAL_ENSURE_PROVIDER_SETTINGS
    )

    app_module._ensure_provider_settings(
        tmp_path, Console(file=StringIO(), force_terminal=False)
    )

    assert calls == []
    assert secret.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs extra privileges")
def test_dangling_provider_symlink_is_left_untouched(monkeypatch, tmp_path):
    secret = tmp_path / ".mitocode" / "secret" / "provider.json"
    secret.parent.mkdir(parents=True)
    target = tmp_path / "missing-provider.json"
    secret.symlink_to(target)
    calls = []
    monkeypatch.setattr(
        app_module, "ask_provider_settings", lambda console: calls.append(console)
    )
    monkeypatch.setattr(
        app_module, "write_secret_file", lambda workspace, values: calls.append(values)
    )
    monkeypatch.setattr(
        app_module, "_ensure_provider_settings", _REAL_ENSURE_PROVIDER_SETTINGS
    )

    app_module._ensure_provider_settings(
        tmp_path, Console(file=StringIO(), force_terminal=False)
    )

    assert calls == []
    assert secret.is_symlink()
    assert not target.exists()


def test_malformed_provider_file_is_left_untouched(monkeypatch, tmp_path):
    secret = tmp_path / ".mitocode" / "secret" / "provider.json"
    secret.parent.mkdir(parents=True)
    secret.write_text('{"api_key": "unfinished', encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        app_module, "ask_provider_settings", lambda console: calls.append(console)
    )
    monkeypatch.setattr(
        app_module, "write_secret_file", lambda workspace, values: calls.append(values)
    )
    monkeypatch.setattr(
        app_module, "_ensure_provider_settings", _REAL_ENSURE_PROVIDER_SETTINGS
    )

    app_module._ensure_provider_settings(
        tmp_path, Console(file=StringIO(), force_terminal=False)
    )

    assert calls == []
    assert secret.read_text(encoding="utf-8") == '{"api_key": "unfinished'


def test_confirm_workspace_renders_panel_and_asks(monkeypatch, tmp_path):
    buffer = StringIO()
    console = Console(
        file=buffer, width=400, force_terminal=False, theme=MITOCODE_THEME
    )
    workspace = tmp_path / "here"
    monkeypatch.setattr(
        app_module.Confirm,
        "ask",
        lambda prompt, *, console, default: default,
    )

    assert app_module._confirm_workspace(console, workspace) is True
    output = buffer.getvalue()
    assert "Workspace setup" in output
    assert str(workspace) in output


def test_workspace_prompt_declined_returns_1(monkeypatch, capsys):
    logged = []

    def fake_logging(**kwargs):
        logged.append(kwargs)
        return Path("unused")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(app_module, "ensure_workspace", lambda path, *, confirm: False)
    monkeypatch.setattr(app_module, "configure_logging", fake_logging)

    assert app_module.main([]) == 1
    assert logged == []
    assert "Workspace was not created" in capsys.readouterr().out


def test_provider_setup_cancelled_returns_1(monkeypatch, capsys):
    timeline = []

    def fake_ensure(workspace, console):
        timeline.append("provider")
        raise EOFError()

    def fake_logging(**kwargs):
        timeline.append("logging")
        return Path("cli.log")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(app_module, "ensure_workspace", lambda path, *, confirm: True)
    monkeypatch.setattr(app_module, "_ensure_provider_settings", fake_ensure)
    monkeypatch.setattr(app_module, "configure_logging", fake_logging)

    assert app_module.main([]) == 1
    # Logging must be configured before the credential prompts, so a cancelled
    # setup still leaves the failure in the workspace log.
    assert timeline == ["logging", "provider"]
    assert "Provider setup was cancelled" in capsys.readouterr().out


def test_provider_setup_interrupt_returns_130(monkeypatch, capsys):
    def fake_ensure(workspace, console):
        raise KeyboardInterrupt()

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(app_module, "ensure_workspace", lambda path, *, confirm: True)
    monkeypatch.setattr(app_module, "_ensure_provider_settings", fake_ensure)
    monkeypatch.setattr(app_module, "configure_logging", lambda **kw: Path("cli.log"))

    assert app_module.main([]) == 130
    assert "Unexpected error" not in capsys.readouterr().err


def test_provider_setup_error_returns_1(monkeypatch, capsys):
    def fake_ensure(workspace, console):
        raise MitoCodeCliConfigError("cannot write provider file")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(app_module, "ensure_workspace", lambda path, *, confirm: True)
    monkeypatch.setattr(app_module, "_ensure_provider_settings", fake_ensure)
    monkeypatch.setattr(app_module, "configure_logging", lambda **kw: Path("cli.log"))

    assert app_module.main([]) == 1
    assert "cannot write provider file" in capsys.readouterr().err


def test_log_directory_failure_returns_1(monkeypatch, capsys):
    def fake_logging(**kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(app_module, "_is_interactive", lambda: True)
    monkeypatch.setattr(app_module, "ensure_workspace", lambda path, *, confirm: True)
    monkeypatch.setattr(
        app_module, "_ensure_provider_settings", lambda workspace, console: None
    )
    monkeypatch.setattr(app_module, "configure_logging", fake_logging)

    assert app_module.main([]) == 1
    assert "log directory" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Session selection, autosave and exit-time save (session persistence)
# ---------------------------------------------------------------------------


def test_no_saved_sessions_starts_new_and_writes_the_file(monkeypatch, tmp_path):
    """With an empty sessions directory the run starts fresh and saves it."""
    boot = _serializable_boot(tmp_path)
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot, session_fakes=False)

    assert app_module.main([]) == 0
    assert boot.received_restore is None
    assert "attach_record" not in timeline
    sessions_dir = tmp_path / ".mitocode" / "sessions"
    files = list(sessions_dir.glob("*.json"))
    assert len(files) == 1, "启动后磁盘上必须已有一个会话记录"


def test_single_saved_session_is_restored_without_asking(monkeypatch, tmp_path):
    """Exactly one saved session is restored directly, no panel, no prompt."""
    record = _session_record("20261107T100000Z-11111111")
    _write_record(tmp_path, record)
    boot = _boot(tmp_path)
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot)
    calls = []

    async def fake_select(ui, summaries, *, allow_new=True):
        calls.append((summaries, allow_new))
        return "new"

    monkeypatch.setattr(app_module, "select_session", fake_select)

    assert app_module.main([]) == 0
    assert calls == [], "单会话必须直接恢复而不是进入选择面板"
    assert boot.received_restore == record
    assert boot.restore_succeeded is True
    assert boot.session_manager.attached == record
    assert boot.session_manager.title_calls == 1
    assert "start-new" not in timeline
    assert "attach_record" in timeline


def test_multiple_sessions_show_the_panel_and_restore_the_choice(monkeypatch, tmp_path):
    """Two or more sessions open the panel; the picked id is restored."""
    older = _session_record(
        "20261107T090000Z-22222222",
        title="Older",
        updated_at="2026-11-07T09:00:00Z",
    )
    newer = _session_record("20261107T100000Z-33333333", title="Newer")
    _write_record(tmp_path, older)
    _write_record(tmp_path, newer)
    boot = _boot(tmp_path)
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot)
    seen = {}

    async def fake_select(ui, summaries, *, allow_new=True):
        seen["ids"] = [summary.session_id for summary in summaries]
        seen["allow_new"] = allow_new
        return summaries[0].session_id

    monkeypatch.setattr(app_module, "select_session", fake_select)

    assert app_module.main([]) == 0
    assert seen["ids"] == [newer["session_id"], older["session_id"]]
    assert seen["allow_new"] is True
    assert boot.received_restore == newer
    assert boot.session_manager.attached == newer
    assert "attach_record" in timeline


def test_choosing_new_session_at_the_panel_starts_fresh(monkeypatch, tmp_path):
    """Picking ``n`` keeps no restore and registers a brand-new session."""
    _write_record(tmp_path, _session_record("20261107T100000Z-44444444"))
    _write_record(tmp_path, _session_record("20261107T090000Z-55555555"))
    boot = _boot(tmp_path)
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot)

    async def fake_select(ui, summaries, *, allow_new=True):
        return "n"

    monkeypatch.setattr(app_module, "select_session", fake_select)

    assert app_module.main([]) == 0
    assert boot.received_restore is None
    assert "start-new" in timeline
    assert "attach_record" not in timeline


def test_corrupt_session_file_is_skipped_with_one_warning(
    monkeypatch, tmp_path, capsys
):
    """A broken session file is counted and warned about exactly once."""
    _write_record(tmp_path, _session_record("20261107T100000Z-66666666"))
    (tmp_path / ".mitocode" / "sessions" / "broken.json").write_text(
        "{ not json", encoding="utf-8"
    )
    boot = _boot(tmp_path)
    _install_common_fakes(monkeypatch, tmp_path, boot)

    assert app_module.main([]) == 0
    output = capsys.readouterr().out
    assert "could not be read" in output
    assert boot.received_restore == _session_record("20261107T100000Z-66666666")


def test_load_failure_at_selection_falls_back_to_new(monkeypatch, tmp_path, capsys):
    """An id that fails to load warns and the run continues with a new one."""
    _write_record(tmp_path, _session_record("20261107T100000Z-44444444"))
    _write_record(tmp_path, _session_record("20261107T090000Z-55555555"))
    boot = _boot(tmp_path)
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot)

    async def fake_select(ui, summaries, *, allow_new=True):
        return "20261107T100000Z-deadbeef"

    monkeypatch.setattr(app_module, "select_session", fake_select)

    assert app_module.main([]) == 0
    assert "could not be loaded" in capsys.readouterr().out
    assert boot.received_restore is None
    assert "start-new" in timeline


def test_eof_during_selection_returns_1(monkeypatch, tmp_path):
    """``Ctrl+D`` at the selection prompt exits with the failure code."""
    _write_record(tmp_path, _session_record("20261107T100000Z-44444444"))
    _write_record(tmp_path, _session_record("20261107T090000Z-55555555"))
    boot = _boot(tmp_path)
    _install_common_fakes(monkeypatch, tmp_path, boot)

    async def raising_select(ui, summaries, *, allow_new=True):
        raise EOFError

    monkeypatch.setattr(app_module, "select_session", raising_select)

    assert app_module.main([]) == 1


def test_interrupt_during_selection_returns_130(monkeypatch, tmp_path):
    """``Ctrl+C`` at the selection prompt keeps the interrupt convention."""
    _write_record(tmp_path, _session_record("20261107T100000Z-44444444"))
    _write_record(tmp_path, _session_record("20261107T090000Z-55555555"))
    boot = _boot(tmp_path)
    _install_common_fakes(monkeypatch, tmp_path, boot)

    async def raising_select(ui, summaries, *, allow_new=True):
        raise KeyboardInterrupt

    monkeypatch.setattr(app_module, "select_session", raising_select)

    assert app_module.main([]) == 130


def test_teardown_order_autosaver_manager_pump_mcp(monkeypatch, tmp_path):
    """Teardown stops the autosaver, then the manager, then pump, then MCP."""
    timeline = []

    async def fake_stop():
        timeline.append("mcp-stopped")

    _install_common_fakes(
        monkeypatch, tmp_path, _boot(tmp_path, stop=fake_stop), events=timeline
    )

    assert app_module.main([]) == 0
    order = [entry for entry in timeline if isinstance(entry, str)]
    assert order.index("autosaver-stop") < order.index("manager-shutdown")
    assert order.index("manager-shutdown") < order.index("pump-cancelled")
    assert order.index("pump-cancelled") < order.index("mcp-stopped")
    assert boot_shutdown_happened(timeline)


def boot_shutdown_happened(timeline):
    """Return whether the manager recorded exactly one shutdown.

    Args:
        timeline: The ordered event list.

    Returns:
        ``True`` when ``manager-shutdown`` appears once.
    """
    return timeline.count("manager-shutdown") == 1


def test_restored_session_history_is_echoed(monkeypatch, tmp_path):
    """恢复成功后，先前会话内容被回显到终端一次."""
    record = _session_record("20261107T100000Z-11111111")
    _write_record(tmp_path, record)
    boot = _boot(tmp_path)
    timeline = _install_common_fakes(monkeypatch, tmp_path, boot)
    echo_calls = []

    def fake_echo(ui, session, *, title, tool_manager=None, root=None):
        echo_calls.append((session, title))

    monkeypatch.setattr(app_module, "render_session_history", fake_echo)

    assert app_module.main([]) == 0
    assert boot.restore_succeeded is True
    assert len(echo_calls) == 1
    echo_session, echo_title = echo_calls[0]
    assert echo_session is boot.main_session
    assert echo_title == boot.session_manager.current_title
    assert "attach_record" in timeline


def test_new_session_does_not_echo_history(monkeypatch, tmp_path):
    """新建会话不回显任何历史."""
    boot = _boot(tmp_path)
    _install_common_fakes(monkeypatch, tmp_path, boot)
    echo_calls = []

    def fake_echo(ui, session, *, title, tool_manager=None, root=None):
        echo_calls.append((session, title))

    monkeypatch.setattr(app_module, "render_session_history", fake_echo)

    assert app_module.main([]) == 0
    assert boot.received_restore is None
    assert boot.restore_succeeded is False
    assert echo_calls == []


def _answer_confirm(monkeypatch, answer):
    """Make the startup confirmation return a fixed answer.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        answer: Value ``Confirm.ask`` returns.

    Returns:
        The list of prompts the confirmation was asked with.
    """
    asked = []

    def _ask(prompt, **_kwargs):
        asked.append(prompt)
        return answer

    monkeypatch.setattr(app_module.Confirm, "ask", staticmethod(_ask))
    return asked


def test_the_workspace_is_locked_while_the_cli_runs(monkeypatch, tmp_path):
    """运行期间 workspace 必须被锁住，另一个实例进不来."""
    from mitocode.core.workspace.workspace import WorkspaceLock

    observed = {}
    boot = _boot(tmp_path)

    async def fake_bootstrap(args, ui, approver, restore=None):
        boot.restore_succeeded = restore is not None
        other = WorkspaceLock(tmp_path)
        observed["acquired"] = other.acquire()
        if observed["acquired"]:
            other.release()
        return boot

    _install_common_fakes(monkeypatch, tmp_path, boot)
    monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)

    assert app_module.main(["-w", str(tmp_path)]) == 0
    assert observed["acquired"] is False, (
        "the workspace must be locked for the whole run; otherwise a second "
        "CLI can start on top of this one"
    )


def test_the_workspace_lock_is_released_when_the_cli_exits(monkeypatch, tmp_path):
    """退出后必须释放锁，下一次启动才能拿到."""
    from mitocode.core.workspace.workspace import WorkspaceLock

    _install_common_fakes(monkeypatch, tmp_path, _boot(tmp_path))

    assert app_module.main(["-w", str(tmp_path)]) == 0

    other = WorkspaceLock(tmp_path)
    assert other.acquire() is True, "the lock must be released when the CLI exits"
    other.release()


def test_a_held_workspace_lock_warns_and_names_the_risk(monkeypatch, tmp_path, capsys):
    """锁被占用时必须警告用户，并说明强制进入的风险."""
    from mitocode.core.workspace.workspace import WorkspaceLock

    holder = WorkspaceLock(tmp_path)
    assert holder.acquire() is True
    try:
        _install_common_fakes(monkeypatch, tmp_path, _boot(tmp_path))
        asked = _answer_confirm(monkeypatch, False)

        assert app_module.main(["-w", str(tmp_path)]) == 1

        output = capsys.readouterr().out
        assert asked, "the user must be asked whether to force entry"
        assert "already" in output.lower() or "in use" in output.lower()
        # The risk must be spelled out, not just announced.
        assert "overwrite" in output.lower()
        assert "background" in output.lower()
    finally:
        holder.release()


def test_the_user_can_refuse_to_force_entry(monkeypatch, tmp_path):
    """用户拒绝强制进入时 CLI 不得启动."""
    from mitocode.core.workspace.workspace import WorkspaceLock

    started = []
    boot = _boot(tmp_path)

    async def fake_bootstrap(args, ui, approver, restore=None):
        started.append(True)
        boot.restore_succeeded = restore is not None
        return boot

    holder = WorkspaceLock(tmp_path)
    assert holder.acquire() is True
    try:
        _install_common_fakes(monkeypatch, tmp_path, boot)
        monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)
        _answer_confirm(monkeypatch, False)

        assert app_module.main(["-w", str(tmp_path)]) == 1
        assert not started, "the CLI must not start while another one holds the lock"
    finally:
        holder.release()


def test_forcing_entry_starts_the_cli_despite_the_held_lock(monkeypatch, tmp_path):
    """用户确认风险后可以强制进入."""
    from mitocode.core.workspace.workspace import WorkspaceLock

    started = []
    boot = _boot(tmp_path)

    async def fake_bootstrap(args, ui, approver, restore=None):
        started.append(True)
        boot.restore_succeeded = restore is not None
        return boot

    holder = WorkspaceLock(tmp_path)
    assert holder.acquire() is True
    try:
        _install_common_fakes(monkeypatch, tmp_path, boot)
        monkeypatch.setattr(app_module, "bootstrap", fake_bootstrap)
        _answer_confirm(monkeypatch, True)

        assert app_module.main(["-w", str(tmp_path)]) == 0
        assert started, "the CLI must start when the user forces entry"
    finally:
        holder.release()
