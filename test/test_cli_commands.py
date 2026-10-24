"""cli.commands 模块单元测试。

覆盖注册表查找/别名/去重、dispatch 的认领规则（斜杠行/普通文本/空命令/
大小写）、各 handler 行为（/help 列表、/clear 清空、/exit 请求、
/connect 向导+热切换联动、/model 委托）与未知命令提示。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli import commands as cmds
from mitocode.cli.commands import (
    SlashCommand,
    build_default_registry,
)


class _FakeSession:
    """记录 clear_context 调用并持有 provider 的 session 替身."""

    def __init__(self):
        self.cleared = 0
        self.provider = object()

    def clear_context(self):
        self.cleared += 1


class _FakeRepl:
    """commands handler 依赖面的最小 REPL 替身."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.console = Console(record=True, width=200)
        self.session = _FakeSession()
        self.orchestrator = object()
        self.exit_requested = 0
        self.applied = []
        self.reader_answers = []

    async def input_reader(self, prompt):
        return self.reader_answers.pop(0) if self.reader_answers else ""

    def request_exit(self):
        self.exit_requested += 1

    async def apply_provider_config(self, config):
        self.applied.append(config)
        return True

    @property
    def commands(self):
        return build_default_registry()


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def repl(tmp_path):
    return _FakeRepl(tmp_path)


# ──────────────────────────────────────────────
# 注册表与别名
# ──────────────────────────────────────────────


def test_lookup_finds_primary_and_alias(registry):
    """login 别名与 connect 指向同一命令对象."""
    assert registry.lookup("connect") is registry.lookup("login")


def test_commands_lists_primaries_without_alias_duplicates(registry):
    """commands() 每个主命令只出现一次（别名不计）."""
    names = [command.name for command in registry.commands()]
    assert names == ["help", "clear", "exit", "connect", "model"]


def test_unknown_lookup_returns_none(registry):
    assert registry.lookup("nope") is None


# ──────────────────────────────────────────────
# dispatch 认领规则
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_ignores_plain_text(registry, repl):
    """普通文本 → 不认领（返回 False，交由聊天路径）."""
    assert await registry.dispatch("hello world", repl) is False
    assert await registry.dispatch("  ", repl) is False


@pytest.mark.asyncio
async def test_dispatch_is_case_insensitive(registry, repl):
    """/HELP 大写 → 照常分发."""
    assert await registry.dispatch("/HELP", repl) is True


@pytest.mark.asyncio
async def test_dispatch_bare_slash_is_not_a_command(registry, repl):
    """仅有斜杠 → 不是命令，按普通文本处理."""
    assert await registry.dispatch("/", repl) is False


@pytest.mark.asyncio
async def test_dispatch_unknown_command_reports(registry, repl):
    """未知命令 → 提示并认领（不当作聊天发送）."""
    assert await registry.dispatch("/nope arg", repl) is True
    assert "Unknown command: nope" in repl.console.export_text()


# ──────────────────────────────────────────────
# 各 handler
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_help_lists_all_commands(registry, repl):
    """/help → 输出包含全部主命令与其帮助文案."""
    await registry.dispatch("/help", repl)
    output = repl.console.export_text()
    for name in ("/help", "/clear", "/exit", "/connect", "/model"):
        assert name in output


@pytest.mark.asyncio
async def test_clear_resets_session_context(registry, repl):
    """/clear → session.clear_context 被调且打印确认."""
    await registry.dispatch("/clear", repl)
    assert repl.session.cleared == 1
    assert "Context cleared" in repl.console.export_text()


@pytest.mark.asyncio
async def test_exit_requests_shutdown(registry, repl):
    """/exit → repl.request_exit 被调一次."""
    await registry.dispatch("/exit", repl)
    assert repl.exit_requested == 1


@pytest.mark.asyncio
async def test_connect_runs_wizard_then_applies(registry, repl, monkeypatch):
    """/connect → 向导保存成功后把配置交给 apply_provider_config."""
    sentinel = object()

    async def fake_wizard(root, reader, console):
        return sentinel

    monkeypatch.setattr(cmds, "connect_wizard", fake_wizard)

    await registry.dispatch("/connect", repl)

    assert repl.applied == [sentinel]


@pytest.mark.asyncio
async def test_connect_skips_apply_when_wizard_aborts(registry, repl, monkeypatch):
    """/connect → 向导中止（None）时不应用配置."""

    async def fake_wizard(root, reader, console):
        return None

    monkeypatch.setattr(cmds, "connect_wizard", fake_wizard)

    await registry.dispatch("/connect", repl)

    assert repl.applied == []


@pytest.mark.asyncio
async def test_connect_alias_login_equivalent(registry, repl, monkeypatch):
    """/login → 与 /connect 同一 handler 路径."""
    hit = []

    async def fake_wizard(root, reader, console):
        hit.append("wizard")
        return None

    monkeypatch.setattr(cmds, "connect_wizard", fake_wizard)

    assert await registry.dispatch("/login", repl) is True
    assert hit == ["wizard"]


@pytest.mark.asyncio
async def test_model_delegates_to_switch_model(registry, repl, monkeypatch):
    """/model → 委托 switch_model，传入 session 的 provider."""
    passed = {}

    async def fake_switch(provider, root, reader, console):
        passed["provider"] = provider
        passed["root"] = root

    monkeypatch.setattr(cmds, "switch_model", fake_switch)

    await registry.dispatch("/model", repl)

    assert passed["provider"] is repl.session.provider
    assert passed["root"] == repl.root


# ──────────────────────────────────────────────
# 注册行为细节
# ──────────────────────────────────────────────


def test_later_registration_replaces_same_name():
    """同名后注册覆盖先注册."""
    registry = build_default_registry()

    async def handler(repl, args):
        return None

    registry.register(SlashCommand("help", handler, "cli.cmd.help.help"))
    assert registry.lookup("help").handler is handler
