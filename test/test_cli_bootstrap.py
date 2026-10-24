"""cli.bootstrap 模块单元测试。

用注入的 input_reader（立即 EOF）与 provider_factory 跑完整装配：
git 引导先行、无 provider 仍进 REPL（提示 /connect）、有 provider 则
创建 Main（系统提示词注入 + 11 个内置工具注册）、provider 构建失败
降级不崩溃、损坏 workspace 报错退出、SYSTEM_PROMPT.md 覆盖生效。
"""

import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli import bootstrap
from mitocode.cli.provider_config import DEFAULT_BASE_URL
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.core.exception.mitocode_exception import MitoCodeBaseError

ENV_VARS = (
    "MITOCODE_API_KEY",
    "OPENAI_API_KEY",
    "MITOCODE_BASE_URL",
    "OPENAI_BASE_URL",
    "MITOCODE_MODEL",
    "MITOCODE_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """清空 provider 环境变量并桩掉 git 引导（避免真实 git 调用）."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    git_calls = []
    monkeypatch.setattr(
        bootstrap, "ensure_git_repo", lambda root, console: git_calls.append(root)
    )
    return git_calls


def _eof_reader():
    """立即 EOF 的输入端口."""

    async def _read(prompt):
        raise EOFError

    return _read


def _provider_factory(sentinel=None):
    """返回构建 spec Mock provider 的工厂."""

    async def factory(model, settings):
        provider = mock.MagicMock(spec=_SampleChatProvider)
        provider.default_model = model
        if sentinel is not None:
            sentinel.append((model, settings))
        return provider

    return factory


def _console():
    return Console(record=True, width=200)


# ──────────────────────────────────────────────
# 无 provider：仍进 REPL
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_without_provider_enters_repl_with_hint(tmp_path, _clean_env):
    """无任何配置 → 欢迎语 + /connect 引导，Main 未创建."""
    console = _console()

    repl = await bootstrap.cli_main(
        root=tmp_path,
        console=console,
        input_reader=_eof_reader(),
        provider_factory=_provider_factory(),
    )

    assert repl is not None
    assert _clean_env == [tmp_path]
    assert repl.session is None
    output = console.export_text()
    assert "MitoCode CLI" in output
    assert "No provider configured" in output


# ──────────────────────────────────────────────
# 有 provider：Main 创建 + 注入 + 工具
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_with_provider_creates_main_session(tmp_path, _clean_env, monkeypatch):
    """env 有 key → 工厂收到正确参数，Main 带系统提示词与全部内置工具."""
    monkeypatch.setenv("MITOCODE_API_KEY", "key-1")
    monkeypatch.setenv("MITOCODE_MODEL", "model-x")
    monkeypatch.setenv("MITOCODE_TIMEOUT", "33")
    calls = []
    console = _console()

    repl = await bootstrap.cli_main(
        root=tmp_path,
        console=console,
        input_reader=_eof_reader(),
        provider_factory=_provider_factory(calls),
    )

    assert calls == [
        (
            "model-x",
            {
                "api_key": "key-1",
                "base_url": DEFAULT_BASE_URL,
                "timeout": 33,
            },
        )
    ]
    session = repl.session
    assert session is not None
    assert session.name == "Main"
    assert session.messages[0].role == "system"
    for tool in ("Read", "Write", "Edit", "Glob", "Grep", "Bash", "Agent"):
        assert tool in session.tools_map, f"built-in {tool} missing"
    output = console.export_text()
    assert "No provider configured" not in output


@pytest.mark.asyncio
async def test_system_prompt_override_file_wins(tmp_path, _clean_env, monkeypatch):
    """SYSTEM_PROMPT.md 存在 → 覆盖内置默认提示词."""
    monkeypatch.setenv("MITOCODE_API_KEY", "key-1")
    await bootstrap._await_or_value(
        bootstrap.Workspace.new_workspace(tmp_path, name="ws")
    )
    override = tmp_path / ".mitocode" / "SYSTEM_PROMPT.md"
    override.write_text("custom override", encoding="utf-8")

    repl = await bootstrap.cli_main(
        root=tmp_path,
        console=_console(),
        input_reader=_eof_reader(),
        provider_factory=_provider_factory(),
    )

    assert repl.session.messages[0].message == "custom override"


# ──────────────────────────────────────────────
# 降级路径
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_failure_degrades_to_hint(tmp_path, _clean_env, monkeypatch):
    """provider 构建抛 MitoCodeBaseError → 警告并继续进 REPL."""
    monkeypatch.setenv("MITOCODE_API_KEY", "key-1")

    async def failing_factory(model, settings):
        raise MitoCodeBaseError("bad key")

    console = _console()
    repl = await bootstrap.cli_main(
        root=tmp_path,
        console=console,
        input_reader=_eof_reader(),
        provider_factory=failing_factory,
    )

    assert repl is not None
    assert repl.session is None
    output = console.export_text()
    assert "Failed to launch the CLI: bad key" in output
    assert "No provider configured" in output


@pytest.mark.asyncio
async def test_corrupt_workspace_reports_and_returns_none(tmp_path, _clean_env):
    """workspace 损坏且无法重建 → 本地化报错，返回 None."""
    config_dir = tmp_path / ".mitocode"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{not json", encoding="utf-8")
    console = _console()

    repl = await bootstrap.cli_main(
        root=tmp_path,
        console=console,
        input_reader=_eof_reader(),
        provider_factory=_provider_factory(),
    )

    assert repl is None
    assert "Failed to launch the CLI" in console.export_text()
