"""Tests for the CLI startup assembly in :mod:`mitocode.cli.bootstrap`.

The provider, the MCP manager and the orchestrator are replaced by recording
fakes, so these tests stay offline and can assert the *order* of the steps that
the design fixes: MCP must start before the first agent session exists, because
``AgentSession`` snapshots the tool registry in its constructor.
"""

import asyncio
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
from mitocode.core.tool.tool_manager import ToolManager
from mitocode.core.workspace.workspace import Workspace

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


# ──────────────────────────────────────────────
# _start_mcp: start() 失败时不得遗留 manager
# ──────────────────────────────────────────────


def _workspace(root: Path) -> Workspace:
    """Build a workspace rooted at *root* for direct ``_start_mcp`` calls.

    Args:
        root: Directory the default MCP configuration is read from.

    Returns:
        The workspace object.
    """
    return Workspace(root, "test-workspace", 1)


class _FailingStartMcp:
    """MCP manager stand-in whose ``start()`` raises a scripted exception.

    Attributes:
        failure: Exception instance raised by ``start()``.
    """

    def __init__(self, config, *, tool_manager):
        self.config = config
        self.tool_manager = tool_manager

    async def start(self):
        """Raise the scripted failure after recording the start."""
        ORDERS.append("mcp.start")
        raise _FailingStartMcp.failure

    async def stop(self):
        """Record a stop without touching real subprocesses."""
        ORDERS.append("mcp.stop")


@pytest.mark.asyncio
async def test_start_mcp_cancelled_stops_manager_and_propagates(tmp_path, monkeypatch):
    """start() 被取消 → 先 shielded stop 掉 manager，CancelledError 原样传播."""
    ORDERS.clear()
    _FailingStartMcp.failure = asyncio.CancelledError()
    monkeypatch.setattr(bootstrap_module, "McpManager", _FailingStartMcp)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    with pytest.raises(asyncio.CancelledError):
        await bootstrap_module._start_mcp(
            _args(workspace=tmp_path), _workspace(tmp_path), ToolManager(), ui
        )

    assert ORDERS.count("mcp.stop") == 1, "取消传播前必须 stop 已启动的 manager"
    assert ORDERS.index("mcp.start") < ORDERS.index("mcp.stop")


@pytest.mark.asyncio
async def test_start_mcp_runtime_error_stops_manager_and_propagates(
    tmp_path, monkeypatch
):
    """start() 抛 RuntimeError → 先 stop manager，异常原样传播."""
    ORDERS.clear()
    _FailingStartMcp.failure = RuntimeError("connect exploded")
    monkeypatch.setattr(bootstrap_module, "McpManager", _FailingStartMcp)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    with pytest.raises(RuntimeError, match="connect exploded"):
        await bootstrap_module._start_mcp(
            _args(workspace=tmp_path), _workspace(tmp_path), ToolManager(), ui
        )

    assert ORDERS.count("mcp.stop") == 1, "异常传播前必须 stop 已启动的 manager"
    assert ORDERS.index("mcp.start") < ORDERS.index("mcp.stop")


@pytest.mark.asyncio
async def test_start_mcp_success_path_returns_manager_without_stop(
    tmp_path, monkeypatch
):
    """start() 成功 → 返回 (manager, "") 且不触发任何 stop."""
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, '{"mcpServers":{"s":{"command":"echo"}}}')
    ui = _ui()

    manager, error = await bootstrap_module._start_mcp(
        _args(workspace=tmp_path), _workspace(tmp_path), ToolManager(), ui
    )

    assert isinstance(manager, _FakeMcp), "成功路径应返回启动的 manager 本身"
    assert error == ""
    assert ORDERS == ["mcp.start"], "成功路径不应有 stop 或其它记录"


@pytest.mark.asyncio
async def test_config_error_path_never_constructs_or_stops_a_manager(
    tmp_path, monkeypatch
):
    """默认 MCP 配置损坏 → 仅警告禁用，不构造 manager 也不调用 stop."""
    ORDERS.clear()
    _patch(monkeypatch)
    _write_mcp(tmp_path, "{ not json")
    ui = _ui()
    known = len(_FakeMcp.instances)

    result = await bootstrap(
        _args(workspace=tmp_path, model="m", api_key="k"), ui, CliApprover(ui)
    )

    assert result.mcp_manager is None
    assert result.mcp_error
    assert len(_FakeMcp.instances) == known, "配置错误路径不应构造任何 manager"
    assert "mcp.stop" not in ORDERS, "不存在 manager 时不得调用 stop"


@pytest.mark.asyncio
async def test_explicit_config_error_propagates_without_stopping(tmp_path, monkeypatch):
    """显式 MCP 配置损坏 → 启动失败传播，且不构造、不 stop 任何 manager."""
    ORDERS.clear()
    _patch(monkeypatch)
    explicit = tmp_path / "servers.json"
    explicit.write_text("{", encoding="utf-8")
    ui = _ui()
    known = len(_FakeMcp.instances)

    with pytest.raises(MitoCodeCliConfigError):
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

    assert len(_FakeMcp.instances) == known, "配置错误路径不应构造任何 manager"
    assert "mcp.stop" not in ORDERS, "不存在 manager 时不得调用 stop"


# ──────────────────────────────────────────────
# bootstrap: API key 交给 orchestrator 用于字面抹除
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_passes_provider_api_key_for_redaction(tmp_path, monkeypatch):
    """bootstrap 把解析出的 API key 作为 redact_secrets 传给 orchestrator."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="m", api_key="k"), ui, CliApprover(ui)
    )

    assert result.orchestrator.kw["redact_secrets"] == (
        "k",
    ), "orchestrator 必须拿到字面 key 才能在错误事件与日志中抹除它"


@pytest.mark.asyncio
async def test_unknown_effort_tier_warns_at_startup(tmp_path, monkeypatch):
    """未知 effort 档位启动时警告一次，且原值进入 provider settings."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="my-model", api_key="k", reasoning_effort="ultra"),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["reasoning_effort"] == "ultra"
    assert any("reasoning_effort" in warning for warning in ui.warnings)


@pytest.mark.asyncio
async def test_known_effort_tier_is_silent(tmp_path, monkeypatch):
    """已知档位不警告，settings 正常携带."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="my-model", api_key="k", reasoning_effort="high"),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["reasoning_effort"] == "high"
    assert not any("reasoning_effort" in warning for warning in ui.warnings)


@pytest.mark.asyncio
async def test_unset_effort_passes_none_in_settings(tmp_path, monkeypatch):
    """未配置 effort 时 settings 携带 None，且不警告."""
    ORDERS.clear()
    _patch(monkeypatch)
    (tmp_path / ".mitocode").mkdir()
    ui = _ui()

    result = await bootstrap(
        _args(workspace=tmp_path, model="my-model", api_key="k"),
        ui,
        CliApprover(ui),
    )

    assert result.provider.settings["reasoning_effort"] is None
    assert not any("reasoning_effort" in warning for warning in ui.warnings)
