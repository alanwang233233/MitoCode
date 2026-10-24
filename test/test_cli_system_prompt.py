"""cli.system_prompt 模块与系统提示词注入链路的单元测试。

覆盖三件事：DEFAULT_SYSTEM_PROMPT 的内容完整性（身份/工具/风格/安全四要素）、
load_system_prompt 的覆盖与回退行为、以及 orchestrator 默认提示词向 session
与子 Agent 的传递（含 set_default_provider 的作用域语义）。
"""

import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.cli.system_prompt import (
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_FILE,
    load_system_prompt,
)
from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider


class _FakeWorkspace:
    """带 path 属性的最小 workspace 替身."""

    def __init__(self, path):
        self.path = str(path)


def _mock_provider() -> mock.MagicMock:
    """构造 spec 限定的 Mock provider（不伪造 turn 属性，避免 deepcopy 分支）."""
    return mock.MagicMock(name="mock_provider", spec=_SampleChatProvider)


# ──────────────────────────────────────────────
# 默认提示词内容
# ──────────────────────────────────────────────


def test_default_prompt_covers_identity_tools_style_safety():
    """内置提示词必须包含身份、工具名、输出风格与安全边界四要素."""
    assert "coding agent" in DEFAULT_SYSTEM_PROMPT
    for tool in ("Read", "Write", "Edit", "Glob", "Grep", "Bash", "Agent"):
        assert tool in DEFAULT_SYSTEM_PROMPT
    assert "concise" in DEFAULT_SYSTEM_PROMPT
    assert "approval" in DEFAULT_SYSTEM_PROMPT


# ──────────────────────────────────────────────
# load_system_prompt
# ──────────────────────────────────────────────


def test_load_returns_default_without_workspace():
    """workspace 为 None → 返回内置默认提示词."""
    assert load_system_prompt(None) is DEFAULT_SYSTEM_PROMPT


def test_load_returns_default_when_override_absent(tmp_path):
    """workspace 存在但没有覆盖文件 → 返回内置默认提示词."""
    assert load_system_prompt(_FakeWorkspace(tmp_path)) is DEFAULT_SYSTEM_PROMPT


def test_load_prefers_override_file(tmp_path):
    """覆盖文件存在 → 整体替换内置提示词."""
    override = tmp_path / ".mitocode" / SYSTEM_PROMPT_FILE
    override.parent.mkdir()
    override.write_text("custom prompt", encoding="utf-8")

    assert load_system_prompt(_FakeWorkspace(tmp_path)) == "custom prompt"


def test_load_falls_back_when_override_unreadable(tmp_path):
    """覆盖文件存在但读取失败 → 回退默认，不阻塞启动."""
    override = tmp_path / ".mitocode" / SYSTEM_PROMPT_FILE
    override.parent.mkdir()
    override.write_text("x", encoding="utf-8")

    with mock.patch("pathlib.Path.read_text", side_effect=OSError("boom")):
        assert load_system_prompt(_FakeWorkspace(tmp_path)) is DEFAULT_SYSTEM_PROMPT


def test_load_returns_default_for_workspace_without_path():
    """workspace 没有 path 属性 → 返回默认提示词."""
    assert load_system_prompt(object()) is DEFAULT_SYSTEM_PROMPT


# ──────────────────────────────────────────────
# orchestrator 默认提示词传递
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_default_prompt_reaches_session():
    """default_system_prompt → 新 session 首条消息为 system 且进入标准上下文."""
    orch = AgentOrchestrator(
        default_provider=_mock_provider(), default_system_prompt="SYS"
    )

    session = await orch.create_session(name="Main")

    assert session.messages[0].role == "system"
    assert session.messages[0].message == "SYS"
    assert session.build_standard_context() == [{"role": "system", "content": "SYS"}]


@pytest.mark.asyncio
async def test_explicit_system_prompt_wins_over_default():
    """create_session 显式传 system_prompt → 覆盖 orchestrator 默认值."""
    orch = AgentOrchestrator(
        default_provider=_mock_provider(), default_system_prompt="SYS"
    )

    session = await orch.create_session(name="Main", system_prompt="EXPLICIT")

    assert session.messages[0].message == "EXPLICIT"


@pytest.mark.asyncio
async def test_sub_agent_inherits_default_prompt():
    """子 Agent（Agent 工具的同一路径）→ 自动继承默认提示词."""
    orch = AgentOrchestrator(
        default_provider=_mock_provider(), default_system_prompt="SYS"
    )
    main_id = await orch.create_agent(name="Main")

    sub_id = await orch.create_agent(name="Sub", parent_id=main_id)

    sub = orch.get_session(sub_id)
    assert sub.messages[0].role == "system"
    assert sub.messages[0].message == "SYS"


@pytest.mark.asyncio
async def test_no_default_prompt_leaves_history_empty():
    """未配置默认提示词 → 行为与从前一致，历史为空."""
    orch = AgentOrchestrator(default_provider=_mock_provider())

    session = await orch.create_session(name="Main")

    assert session.messages == []


@pytest.mark.asyncio
async def test_set_default_provider_affects_future_sessions_only():
    """set_default_provider → 只影响未来 session，已存在 session 不变."""
    first = _mock_provider()
    orch = AgentOrchestrator(default_provider=first)
    session = await orch.create_session(name="Main")

    second = _mock_provider()
    orch.set_default_provider(second)
    later = await orch.create_session(name="Later")

    assert session.provider is first
    assert later.provider is second
