"""cli.mcp_setup 模块单元测试。

覆盖：mcp.json 读写（缺失/损坏降级）、McpController 生命周期
（start/stop/restart、配置错误不阻塞）、/mcp 命令三分支
（list 状态展示、add stdio/http 写回+重建、remove 移除+重建）、
以及 bootstrap 的调序约束（manager.start 先于 Main session 创建）
与退出时干净断开。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli import bootstrap
from mitocode.cli import mcp_setup
from mitocode.cli.commands import CommandRegistry
from mitocode.cli.mcp_setup import (
    MCP_FILE_RELATIVE,
    McpController,
    load_mcp_file,
    register_mcp_commands,
    save_mcp_file,
)
from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.mcp.mcp_manager import McpServerStatus
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError


class _FakeManager:
    """记录调用的 McpManager 替身."""

    instances = []

    def __init__(self, config, *, tool_manager, client_factory=None):
        self.config = config
        self.tool_manager = tool_manager
        self.calls = []
        self._status = {
            "s1": McpServerStatus(state="connected", error="", tool_count=2)
        }
        _FakeManager.instances.append(self)

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("stop")

    def status(self):
        return self._status


@pytest.fixture(autouse=True)
def _fake_manager(monkeypatch):
    """替换 mcp_setup 命名空间中的 McpManager."""
    _FakeManager.instances = []
    monkeypatch.setattr(mcp_setup, "McpManager", _FakeManager)
    return _FakeManager


def _write_mcp_file(root, servers):
    path = root / MCP_FILE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


# ──────────────────────────────────────────────
# 文件读写
# ──────────────────────────────────────────────


def test_load_returns_none_when_missing(tmp_path):
    assert load_mcp_file(tmp_path) is None


def test_load_roundtrip(tmp_path):
    save_mcp_file(tmp_path, {"mcpServers": {"a": {"command": "x"}}})
    assert load_mcp_file(tmp_path) == {"mcpServers": {"a": {"command": "x"}}}


def test_load_corrupt_file_warns_and_returns_none(tmp_path):
    _write_mcp_file(tmp_path, {})
    (tmp_path / MCP_FILE_RELATIVE).write_text("{bad", encoding="utf-8")
    console = Console(record=True, width=200)

    assert load_mcp_file(tmp_path, console) is None
    assert "Could not load MCP config" in console.export_text()


# ──────────────────────────────────────────────
# McpController 生命周期
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_without_file_is_noop(tmp_path):
    """无 mcp.json → 不构建 manager."""
    controller = McpController(tmp_path, tool_manager=object(), console=Console())

    await controller.start()

    assert controller.manager is None
    assert _FakeManager.instances == []


@pytest.mark.asyncio
async def test_start_builds_manager_and_prints_summary(tmp_path):
    """有配置 → manager 以 (config, tool_manager=) 构建并 start."""
    _write_mcp_file(tmp_path, {"echo": {"command": "echo"}})
    tool_manager = object()
    console = Console(record=True, width=200)
    controller = McpController(tmp_path, tool_manager=tool_manager, console=console)

    await controller.start()

    assert controller.manager is not None
    assert controller.manager.config == {"mcpServers": {"echo": {"command": "echo"}}}
    assert controller.manager.tool_manager is tool_manager
    assert controller.manager.calls == ["start"]
    assert "MCP: 1 connected" in console.export_text()


@pytest.mark.asyncio
async def test_start_swallows_config_error(tmp_path, monkeypatch):
    """配置非法（构造器抛 MitoCode 错误）→ 警告不阻塞，manager 保持 None."""
    _write_mcp_file(tmp_path, {"bad": {"command": "x", "url": "y"}})

    class _RaisingManager:
        def __init__(self, config, *, tool_manager, client_factory=None):
            raise MitoCodeBaseError("mutually exclusive transports")

    monkeypatch.setattr(mcp_setup, "McpManager", _RaisingManager)
    console = Console(record=True, width=200)
    controller = McpController(tmp_path, tool_manager=object(), console=console)

    await controller.start()

    assert controller.manager is None
    assert "Could not load MCP config" in console.export_text()


@pytest.mark.asyncio
async def test_restart_stops_then_starts(tmp_path):
    """restart → stop 旧 manager 再从文件重建."""
    _write_mcp_file(tmp_path, {"echo": {"command": "echo"}})
    controller = McpController(tmp_path, tool_manager=object(), console=Console())
    await controller.start()
    first = controller.manager

    await controller.restart()

    assert first is not controller.manager
    assert first.calls == ["start", "stop"]
    assert controller.manager.calls == ["start"]


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path):
    """重复 stop 安全（manager 置 None）."""
    _write_mcp_file(tmp_path, {"echo": {"command": "echo"}})
    controller = McpController(tmp_path, tool_manager=object(), console=Console())
    await controller.start()

    await controller.stop()
    await controller.stop()

    assert controller.manager is None


# ──────────────────────────────────────────────
# /mcp 命令
# ──────────────────────────────────────────────


class _FakeRepl:
    """/mcp 命令依赖面的 REPL 替身."""

    def __init__(self, tmp_path, controller, answers):
        self.root = tmp_path
        self.console = Console(record=True, width=200)
        self.mcp = controller
        self._answers = list(answers)

    async def input_reader(self, prompt):
        return self._answers.pop(0) if self._answers else ""


def _registry(repl):
    registry = CommandRegistry()
    register_mcp_commands(registry, repl.mcp)
    return registry


@pytest.mark.asyncio
async def test_mcp_list_without_manager_shows_none(tmp_path):
    repl = _FakeRepl(tmp_path, McpController(tmp_path, object(), Console()), [])
    _write_mcp_file(tmp_path, {})

    await _registry(repl).dispatch("/mcp list", repl)

    assert "No MCP servers configured" in repl.console.export_text()


@pytest.mark.asyncio
async def test_mcp_list_shows_server_states(tmp_path):
    controller = McpController(tmp_path, object(), Console())
    controller.manager = _FakeManager({}, tool_manager=object())
    controller.manager._status = {
        "alpha": McpServerStatus(state="connected", error="", tool_count=3),
        "beta": McpServerStatus(state="failed", error="boom", tool_count=0),
    }
    repl = _FakeRepl(tmp_path, controller, [])

    await _registry(repl).dispatch("/mcp", repl)

    output = repl.console.export_text()
    assert "alpha: connected" in output
    assert "beta: failed" in output
    assert "boom" in output


@pytest.mark.asyncio
async def test_mcp_add_stdio_writes_and_restarts(tmp_path):
    controller = McpController(tmp_path, object(), Console())
    repl = _FakeRepl(tmp_path, controller, ["echo", "stdio", "python -m echo_server"])

    await _registry(repl).dispatch("/mcp add", repl)

    stored = json.loads((tmp_path / MCP_FILE_RELATIVE).read_text())
    assert stored == {"mcpServers": {"echo": {"command": "python -m echo_server"}}}
    assert controller.manager is not None  # restart 已重建
    assert "Added MCP server 'echo'" in repl.console.export_text()


@pytest.mark.asyncio
async def test_mcp_add_http_writes_url_entry(tmp_path):
    controller = McpController(tmp_path, object(), Console())
    repl = _FakeRepl(
        tmp_path, controller, ["remote", "http", "https://mcp.example/sse"]
    )

    await _registry(repl).dispatch("/mcp add", repl)

    stored = json.loads((tmp_path / MCP_FILE_RELATIVE).read_text())
    assert stored == {"mcpServers": {"remote": {"url": "https://mcp.example/sse"}}}


@pytest.mark.asyncio
async def test_mcp_add_rejects_invalid_kind(tmp_path):
    controller = McpController(tmp_path, object(), Console())
    repl = _FakeRepl(tmp_path, controller, ["x", "ftp", "y"])

    await _registry(repl).dispatch("/mcp add", repl)

    assert "Kind must be 'stdio' or 'http'" in repl.console.export_text()
    assert not (tmp_path / MCP_FILE_RELATIVE).exists()


@pytest.mark.asyncio
async def test_mcp_remove_deletes_and_restarts(tmp_path):
    _write_mcp_file(tmp_path, {"echo": {"command": "x"}, "other": {"command": "y"}})
    controller = McpController(tmp_path, object(), Console())
    repl = _FakeRepl(tmp_path, controller, ["echo"])

    await _registry(repl).dispatch("/mcp remove", repl)

    stored = json.loads((tmp_path / MCP_FILE_RELATIVE).read_text())
    assert stored == {"mcpServers": {"other": {"command": "y"}}}
    assert "Removed MCP server 'echo'" in repl.console.export_text()


@pytest.mark.asyncio
async def test_mcp_remove_unknown_name(tmp_path):
    _write_mcp_file(tmp_path, {"echo": {"command": "x"}})
    controller = McpController(tmp_path, object(), Console())
    repl = _FakeRepl(tmp_path, controller, ["ghost"])

    await _registry(repl).dispatch("/mcp remove", repl)

    assert "No MCP server named 'ghost'" in repl.console.export_text()


@pytest.mark.asyncio
async def test_mcp_unknown_subcommand(tmp_path):
    controller = McpController(tmp_path, object(), Console())
    repl = _FakeRepl(tmp_path, controller, [])

    await _registry(repl).dispatch("/mcp frobnicate", repl)

    assert "Unknown command: mcp frobnicate" in repl.console.export_text()


# ──────────────────────────────────────────────
# bootstrap 调序与清理
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_starts_mcp_before_main_and_stops_on_exit(
    tmp_path, monkeypatch
):
    """mcp start 先于 Main session 创建；退出时 manager 干净 stop."""
    # 先建合法 workspace（.mitocode/ 已存在时 new_workspace 会拒绝重建）
    await bootstrap._await_or_value(
        bootstrap.Workspace.new_workspace(tmp_path, name="ws")
    )
    monkeypatch.setattr(bootstrap, "ensure_git_repo", lambda root, console: None)
    _write_mcp_file(tmp_path, {"echo": {"command": "x"}})
    order = []

    original_start = _FakeManager.start

    async def recording_start(self):
        order.append("mcp_start")
        await original_start(self)

    monkeypatch.setattr(_FakeManager, "start", recording_start)

    class _OrderingOrchestrator(AgentOrchestrator):
        async def create_agent(self, *args, **kwargs):
            order.append("main")
            return await super().create_agent(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "AgentOrchestrator", _OrderingOrchestrator)
    monkeypatch.setenv("MITOCODE_API_KEY", "key-1")

    async def factory(model, settings):
        from unittest import mock

        from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider

        return mock.MagicMock(spec=_SampleChatProvider)

    async def eof_reader(prompt):
        raise EOFError

    repl = await bootstrap.cli_main(
        root=tmp_path,
        console=Console(record=True, width=200),
        input_reader=eof_reader,
        provider_factory=factory,
    )

    assert order == ["mcp_start", "main"]
    assert repl is not None
    # controller.stop() 在退出时把 manager 置 None（设计如此），
    # 干净断开通过 fake manager 的调用记录验证。
    assert len(_FakeManager.instances) == 1
    assert _FakeManager.instances[0].calls == ["start", "stop"]
