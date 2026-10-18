"""Unit tests for the built-in approval protocol."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.tool.builtin.approval import (
    AlwaysAllow,
    ApprovalDecision,
    ApprovalRequest,
    DenyAll,
)

REQUEST = ApprovalRequest(action="bash", detail="cat /etc/passwd", reason="outside root")


@pytest.mark.asyncio
async def test_always_allow_allows_everything():
    assert await AlwaysAllow().request(REQUEST) is ApprovalDecision.ALLOW


@pytest.mark.asyncio
async def test_deny_all_denies_everything():
    assert await DenyAll().request(REQUEST) is ApprovalDecision.DENY


def test_request_carries_action_detail_and_reason():
    assert (REQUEST.action, REQUEST.detail) == ("bash", "cat /etc/passwd")
    assert "outside root" in REQUEST.reason
