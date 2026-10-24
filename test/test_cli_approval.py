"""cli.approval 模块单元测试。

覆盖：规则存储（读写/损坏降级/无效条目跳过）、两类模式匹配的边界
（词边界/路径分隔符）、规则推导（bash 首 token+子命令、文件父目录）、
审批三选各分支（规则命中免询问/一次允许/总是+确认持久化/拒绝）、
无效输入循环、跨动作不越权。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from mitocode.cli.approval import (
    ApprovalRule,
    ApprovalRuleStore,
    InteractiveApprover,
    derive_rule,
)
from mitocode.core.tool.builtin.approval import (
    ApprovalDecision,
    ApprovalRequest,
)


def _ask(*answers):
    """脚本化 ask 端口."""
    answers = list(answers)

    async def _read(prompt):
        return answers.pop(0) if answers else ""

    return _read


def _bash(command="npm install -g x"):
    return ApprovalRequest(action="bash", detail=command, reason="escape")


def _read_req(path="/etc/passwd"):
    return ApprovalRequest(action="read", detail=path, reason="outside root")


def _approver(tmp_path, answers):
    console = Console(record=True, width=200)
    return InteractiveApprover(tmp_path, _ask(*answers), console), console


# ──────────────────────────────────────────────
# ApprovalRule 匹配
# ──────────────────────────────────────────────


def test_bash_pattern_matches_word_boundary():
    """npm install:* 匹配子命令扩展，不匹配同前缀词."""
    rule = ApprovalRule(action="bash", pattern="npm install:*")
    assert rule.matches(_bash("npm install -g x")) is True
    assert rule.matches(_bash("npm install")) is True
    assert rule.matches(_bash("npm publish")) is False
    assert rule.matches(_bash("npmx install y")) is False
    assert rule.matches(_bash("npm installfoo")) is False


def test_bash_exact_pattern_requires_equality():
    """无 :* 后缀 → 精确相等才匹配."""
    rule = ApprovalRule(action="bash", pattern="git status")
    assert rule.matches(_bash("git status")) is True
    assert rule.matches(_bash("git status --short")) is False


def test_path_prefix_matches_separator_boundary():
    """/etc 匹配 /etc 与 /etc/...，不匹配 /etcetera."""
    rule = ApprovalRule(action="read", path_prefix="/etc")
    assert rule.matches(_read_req("/etc/passwd")) is True
    assert rule.matches(_read_req("/etc")) is True
    assert rule.matches(_read_req("/etcetera/x")) is False


def test_rule_never_crosses_actions():
    """read 规则不授权 bash，bash 规则不授权 write."""
    assert ApprovalRule(action="read", path_prefix="/etc").matches(_bash()) is False
    assert ApprovalRule(action="bash", pattern="npm:*").matches(_read_req()) is False


# ──────────────────────────────────────────────
# 规则推导
# ──────────────────────────────────────────────


def test_derive_bash_uses_first_two_tokens():
    """bash → 首 token+子命令粒度（绝不推导裸 npm:*）."""
    rule = derive_rule(_bash("npm install -g x"))
    assert rule == ApprovalRule(action="bash", pattern="npm install:*")


def test_derive_bash_single_token():
    """单 token 命令 → token:*."""
    rule = derive_rule(_bash("git"))
    assert rule.pattern == "git:*"


def test_derive_file_uses_parent_directory():
    """文件操作 → 目标父目录前缀."""
    rule = derive_rule(_read_req("/etc/passwd"))
    assert rule == ApprovalRule(action="read", path_prefix="/etc")


# ──────────────────────────────────────────────
# ApprovalRuleStore
# ──────────────────────────────────────────────


def test_store_roundtrip(tmp_path):
    """add → 文件持久化 → 新 store 加载同一规则."""
    store = ApprovalRuleStore(tmp_path)
    store.add(ApprovalRule(action="bash", pattern="npm install:*"))
    path = tmp_path / ".mitocode" / "approval.json"
    assert json.loads(path.read_text()) == {
        "rules": [{"action": "bash", "pattern": "npm install:*"}]
    }

    reloaded = ApprovalRuleStore(tmp_path)
    assert reloaded.rules() == [ApprovalRule(action="bash", pattern="npm install:*")]


def test_store_corrupt_file_degrades_to_empty(tmp_path):
    """损坏文件 → 空规则集，不抛异常."""
    path = tmp_path / ".mitocode" / "approval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert ApprovalRuleStore(tmp_path).rules() == []


def test_store_skips_invalid_entries(tmp_path):
    """无效条目（缺 action / 缺 pattern+path_prefix / 非对象）→ 跳过."""
    path = tmp_path / ".mitocode" / "approval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {"pattern": "x:*"},
                    {"action": "bash"},
                    "not-a-dict",
                    {"action": "read", "path_prefix": "/etc"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rules = ApprovalRuleStore(tmp_path).rules()
    assert rules == [ApprovalRule(action="read", path_prefix="/etc")]


# ──────────────────────────────────────────────
# InteractiveApprover 三选
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rule_hit_allows_without_asking(tmp_path):
    """已有规则命中 → 直接 ALLOW，不打扰用户."""
    ApprovalRuleStore(tmp_path).add(ApprovalRule(action="bash", pattern="npm:*"))
    approver, console = _approver(tmp_path, answers=[])

    decision = await approver.request(_bash("npm install x"))

    assert decision is ApprovalDecision.ALLOW
    assert "Choice" not in console.export_text()


@pytest.mark.asyncio
async def test_choice_one_allows_once_without_rule(tmp_path):
    """选 1 → ALLOW，规则不持久化."""
    approver, _ = _approver(tmp_path, answers=["1"])

    decision = await approver.request(_bash("git push"))

    assert decision is ApprovalDecision.ALLOW
    assert ApprovalRuleStore(tmp_path).rules() == []


@pytest.mark.asyncio
async def test_choice_three_denies(tmp_path):
    """选 3 → DENY 并提示."""
    approver, console = _approver(tmp_path, answers=["3"])

    decision = await approver.request(_bash("rm -rf /"))

    assert decision is ApprovalDecision.DENY
    assert "Denied" in console.export_text()


@pytest.mark.asyncio
async def test_choice_two_confirmed_persists_rule(tmp_path):
    """选 2 + 确认 y → 规则持久化，后续同类免询问."""
    approver, console = _approver(tmp_path, answers=["2", "y"])

    first = await approver.request(_bash("npm install lodash"))
    second = await approver.request(_bash("npm install react"))

    assert first is ApprovalDecision.ALLOW
    assert second is ApprovalDecision.ALLOW
    output = console.export_text()
    assert "bash(npm install:*)" in output  # 推导规则已展示并保存
    assert ApprovalRuleStore(tmp_path).rules() == [
        ApprovalRule(action="bash", pattern="npm install:*")
    ]


@pytest.mark.asyncio
async def test_choice_two_declined_does_not_persist(tmp_path):
    """选 2 + 确认 n → 本次 ALLOW 但规则不保存."""
    approver, _ = _approver(tmp_path, answers=["2", "n"])

    decision = await approver.request(_bash("git push"))

    assert decision is ApprovalDecision.ALLOW
    assert ApprovalRuleStore(tmp_path).rules() == []


@pytest.mark.asyncio
async def test_invalid_choice_loops_until_valid(tmp_path):
    """无效输入 → 提示后循环重问，直到合法选择."""
    approver, console = _approver(tmp_path, answers=["7", "", "1"])

    decision = await approver.request(_bash("curl http://x"))

    assert decision is ApprovalDecision.ALLOW
    output = console.export_text()
    assert output.count("Please enter 1, 2 or 3.") == 2


@pytest.mark.asyncio
async def test_file_request_path_rule_flow(tmp_path):
    """文件请求 → 选 2 确认 → 保存父目录前缀规则."""
    approver, _ = _approver(tmp_path, answers=["2", "y"])

    decision = await approver.request(_read_req("/etc/hosts"))

    assert decision is ApprovalDecision.ALLOW
    assert ApprovalRuleStore(tmp_path).rules() == [
        ApprovalRule(action="read", path_prefix="/etc")
    ]
