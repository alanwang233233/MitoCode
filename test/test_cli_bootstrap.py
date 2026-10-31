"""Tests for the CLI startup assembly in :mod:`mitocode.cli.bootstrap`.

The provider, the MCP manager and the orchestrator are replaced by recording
fakes, so these tests stay offline and can assert the *order* of the steps that
the design fixes: MCP must start before the first agent session exists, because
``AgentSession`` snapshots the tool registry in its constructor.
"""

import os
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mitocode.cli.bootstrap as bootstrap_module
from mitocode.cli.approval import CliApprover
from mitocode.cli.args import CliArgs
from mitocode.cli.bootstrap import (
    SYSTEM_PROMPT_TEMPLATE,
    Bootstrapped,
    bootstrap,
    build_system_prompt,
)
from mitocode.cli.errors import MitoCodeCliConfigError
from mitocode.cli.ui import CliUI

ORDERS: list[str] = []


class _FakeProvider:
    """Recording stand-in for :class:`ChatProviderOpenAI`.

    ``factory`` records the call, then returns an instance holding the keyword
    arguments so tests can inspect what the bootstrap passed to the provider.
    """

    def __init__(self, **kwargs):
        self.default_model = kwargs.get("default_model", "")
        self.model_list = list(kwargs.get("default_model_list") or [])
        self.settings = kwargs.get("settings", {})
        self.stream = kwargs.get("stream", False)

    @classmethod
    async def factory(cls, **kwargs):
        ORDERS.append("provider.factory")
        return cls(**kwargs)


class _FakeMcp:
    """Recording stand-in for :class:`McpManager`."""

    instances: list = []

    def __init__(self, config, *, tool_manager):
        self.config = config
        self.tool_manager = tool_manager
        _FakeMcp.instances.append(self)

    async def start(self):
        ORDERS.append("mcp.start")

    async def stop(self):
        ORDERS.append("mcp.stop")

    def status(self):
        return {}


class _FakeOrchestrator:
    """Recording stand-in for :class:`AgentOrchestrator`."""

    def __init__(self, **kwargs):
        ORDERS.append("orchestrator")
        self.kw = kwargs

    async def create_session(self, **kwargs):
        ORDERS.append("create_session")
        self.session_kwargs = kwargs
        return SimpleNamespace(agent_id=0, **kwargs)


class _RecordingUI(CliUI):
    """A real :class:`CliUI` that also keeps its warnings and errors."""

    def __init__(self, console: Console) -> None:
        super().__init__(console)
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        super().warn(text)

    def error(self, text: str) -> None:
        self.errors.append(text)
        super().error(text)


def _ui() -> _RecordingUI:
    """Build a recording UI whose console writes into a buffer.

    Returns:
        The UI, with the ``StringIO`` buffer attached as ``stream_buffer``.
    """
    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False)
    ui = _RecordingUI(console)
    ui.stream_buffer = buffer  # type: ignore[attr-defined]
    return ui


def _args(**over) -> CliArgs:
    """Build :class:`CliArgs` with every startup field defaulted.

    Args:
        **over: Fields to override.

    Returns:
        The arguments.
    """
    base = dict(
        workspace=Path("."),
        model=None,
        api_key=None,
        base_url=None,
        timeout=60.0,
        permission_mode="ask",
        mcp_config=None,
        no_mcp=False,
        verbose=False,
        debug=False,
        initial_prompt=None,
        show_help=False,
        show_version=False,
    )
    base.update(over)
    return CliArgs(**base)


def _patch(monkeypatch, *, provider=_FakeProvider, mcp=_FakeMcp):
    """Install the recording fakes on the bootstrap module.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        provider: Replacement for ``ChatProviderOpenAI``.
        mcp: Replacement for ``McpManager``.
    """
    monkeypatch.setattr(bootstrap_module, "ChatProviderOpenAI", provider)
    monkeypatch.setattr(bootstrap_module, "McpManager", mcp)
    monkeypatch.setattr(bootstrap_module, "AgentOrchestrator", _FakeOrchestrator)


def _write_mcp(root: Path, payload: str) -> Path:
    """Write the default MCP configuration file under *root*.

    Args:
        root: Workspace root.
        payload: File contents.

    Returns:
        The path that was written.
    """
    path = root / ".mitocode" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def test_system_prompt_names_the_workspace(tmp_path):
    prompt = build_system_prompt(tmp_path)

    assert prompt.startswith("You are MitoCode")
    assert str(tmp_path.resolve()) in prompt
    assert "{workspace}" not in prompt
    assert SYSTEM_PROMPT_TEMPLATE.count("{workspace}") == 1


@pytest.mark.asyncio
async def test_mcp_starts_before_session_and_model_is_union(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="my-model", api_key="k"),
        ui,
        CliApprover(ui),
    )

    assert ORDERS.index("provider.factory") < ORDERS.index("mcp.start")
    assert ORDERS.index("mcp.start") < ORDERS.index("create_session")
    assert "my-model" in result.provider.model_list
    assert result.orchestrator.kw["system_prompt"].startswith("You are MitoCode")
    assert result.orchestrator.kw["system_prompt"].count(str(tmp_path.resolve())) == 1
    assert result.orchestrator.kw["tool_manager"] is result.tool_manager
    assert result.orchestrator.kw["default_provider"] is result.provider
    assert result.tool_context.orchestrator is result.orchestrator
    assert result.orchestrator.session_kwargs == {
        "name": "Main",
        "description": "Interactive session",
    }
    assert result.mcp_manager is not None
    assert result.mcp_error == ""
    assert result.provider_config.model == "my-model"
    assert result.provider.settings["api_key"] == "k"
    assert result.provider.settings["timeout"] == 60.0
    assert isinstance(result, Bootstrapped)


@pytest.mark.asyncio
async def test_model_absent_from_live_list_is_appended_with_warning(
    tmp_path, monkeypatch
):
    ORDERS.clear()

    class _EndpointProvider(_FakeProvider):
        @classmethod
        async def factory(cls, **kwargs):
            ORDERS.append("provider.factory")
            instance = cls(**kwargs)
            instance.model_list = ["other-model"]
            return instance

    _patch(monkeypatch, provider=_EndpointProvider)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="my-model", api_key="k"),
        ui,
        CliApprover(ui),
    )

    assert result.provider.model_list == ["other-model", "my-model"]
    assert any("not advertised" in warning for warning in ui.warnings)


@pytest.mark.asyncio
async def test_model_present_in_live_list_is_not_duplicated(tmp_path, monkeypatch):
    ORDERS.clear()

    class _ListingProvider(_FakeProvider):
        @classmethod
        async def factory(cls, **kwargs):
            ORDERS.append("provider.factory")
            instance = cls(**kwargs)
            instance.model_list = ["my-model"]
            return instance

    _patch(monkeypatch, provider=_ListingProvider)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="my-model", api_key="k"),
        ui,
        CliApprover(ui),
    )

    assert result.provider.model_list == ["my-model"]
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_broken_default_mcp_is_non_fatal(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, "{}")
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="m", api_key="k"), ui, CliApprover(ui)
    )

    assert result.mcp_manager is None
    assert result.mcp_error
    assert ui.warnings


@pytest.mark.asyncio
async def test_explicit_broken_mcp_is_fatal(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    explicit = tmp_path / "servers.json"
    explicit.write_text("{", encoding="utf-8")
    ui = _ui()

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await bootstrap(
            _args(
                workspace=tmp_path,
                model="m",
                api_key="k",
                mcp_config=explicit,
            ),
            ui,
            CliApprover(ui),
        )

    assert "explicit" in str(excinfo.value)
    assert not any(order == "mcp.start" for order in ORDERS)


@pytest.mark.asyncio
async def test_no_mcp_flag_creates_no_manager(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="m", api_key="k", no_mcp=True),
        ui,
        CliApprover(ui),
    )

    assert result.mcp_manager is None
    assert result.mcp_error == ""
    assert ui.warnings == []


@pytest.mark.asyncio
async def test_missing_default_mcp_creates_no_manager(tmp_path, monkeypatch):
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="m", api_key="k"), ui, CliApprover(ui)
    )

    assert result.mcp_manager is None
    assert result.mcp_error == ""


@pytest.mark.asyncio
async def test_provider_failure_becomes_config_error_with_scrubbed_key(
    tmp_path, monkeypatch
):
    ORDERS.clear()

    class _FailingProvider:
        @classmethod
        async def factory(cls, **kwargs):
            raise RuntimeError(f"refused credential {kwargs['settings']['api_key']}")

    _patch(monkeypatch, provider=_FailingProvider)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    with pytest.raises(MitoCodeCliConfigError) as excinfo:
        await bootstrap(
            _args(workspace=tmp_path, model="m", api_key="super-secret"),
            ui,
            CliApprover(ui),
        )

    message = str(excinfo.value)
    assert "super-secret" not in message
    assert "***" in message
    assert not any(order == "mcp.start" for order in ORDERS)


@pytest.mark.asyncio
async def test_session_failure_after_mcp_start_stops_the_manager(tmp_path, monkeypatch):
    """A failure after ``mcp.start()`` must not orphan the MCP servers.

    ``create_session`` is the last startup step and the first one that can fail
    while a live manager exists; the manager reference is otherwise lost with
    the exception.
    """
    ORDERS.clear()

    class _ExplodingOrchestrator(_FakeOrchestrator):
        """Orchestrator whose first session cannot be created."""

        async def create_session(self, **kwargs):
            ORDERS.append("create_session")
            raise RuntimeError("session creation failed")

    _patch(monkeypatch)
    monkeypatch.setattr(bootstrap_module, "AgentOrchestrator", _ExplodingOrchestrator)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    with pytest.raises(RuntimeError, match="session creation failed"):
        await bootstrap(
            _args(workspace=tmp_path, model="m", api_key="k"),
            ui,
            CliApprover(ui),
        )

    assert ORDERS.count("mcp.stop") == 1
    assert ORDERS.index("mcp.start") < ORDERS.index("mcp.stop")
    assert ORDERS.index("create_session") < ORDERS.index("mcp.stop")
    assert _FakeMcp.instances[-1].config is not None
