"""Tests for the CLI session persistence layer (``cli/sessions.py``).

These tests cover the on-disk session protocol only: id generation, message
block (de)serialization, whole-tree records, atomic writes, validated
listing/loading and the startup selection panel.  No test touches a real
terminal or network: UI output is read from a ``StringIO`` console and input
is driven by an injected fake prompt session.
"""

import asyncio
import json
import os
import re
import stat
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from rich.console import Console

from mitocode.cli.sessions import (
    AUTOSAVE_INTERVAL,
    SCHEMA_VERSION,
    SessionAutosaver,
    SessionLoadError,
    SessionManager,
    SessionStore,
    SessionSummary,
    _block_from_dict,
    block_to_dict,
    fallback_title,
    generate_session_title,
    is_valid_session_id,
    new_session_id,
    restore_agent_messages,
    select_session,
    serialize_tree,
    validate_record,
)

from mitocode.cli.pump import EventPump
from mitocode.cli.theme import MITOCODE_THEME
from mitocode.cli.ui import CliUI

from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.agent.agent_session import AgentSession

# noinspection PyProtectedMember
from mitocode.core.agent.agent_session import (
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider

# ---------------------------------------------------------------------------
# Task 5: session id generation and file-name safety
# ---------------------------------------------------------------------------


def test_new_session_id_matches_utc_stamp_plus_hex():
    """new_session_id 生成 UTC 时间戳 + 8 位十六进制的可排序 id."""
    value = new_session_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", value)


def test_new_session_id_is_unique_per_call():
    """两次调用生成不同的 id."""
    assert new_session_id() != new_session_id()


def test_is_valid_session_id_accepts_plain_ids():
    """合法样例（字母数字点下划线连字符）返回 True."""
    assert is_valid_session_id(new_session_id())
    assert is_valid_session_id("20261107T152000Z-1a2b3c4d")
    assert is_valid_session_id("abc_123.XYZ-09")


def test_is_valid_session_id_rejects_unsafe_values():
    """路径分隔符、..、空串与非字符串都返回 False."""
    assert not is_valid_session_id("a/b")
    assert not is_valid_session_id("a\\b")
    assert not is_valid_session_id("..")
    assert not is_valid_session_id("")
    assert not is_valid_session_id(None)
    assert not is_valid_session_id(123)
    assert not is_valid_session_id("a b")


def test_module_exports_schema_version_error_and_summary():
    """模块导出 SCHEMA_VERSION、SessionLoadError 与 SessionSummary."""
    assert SCHEMA_VERSION == 1
    assert issubclass(SessionLoadError, Exception)
    summary = SessionSummary(
        session_id="s",
        title="t",
        updated_at="2026-11-07T15:20:00Z",
        created_at="2026-11-07T15:00:00Z",
        message_count=3,
        model="m",
    )
    assert summary.message_count == 3
    with pytest.raises(Exception):
        summary.title = "x"  # type: ignore[misc]  # frozen dataclass


# ---------------------------------------------------------------------------
# Task 6: message block serialization
# ---------------------------------------------------------------------------


def test_text_block_round_trips_with_role_and_reasoning():
    """user/assistant 文本块 round-trip 后 role/message/reasoning 一致."""
    user = _PlainTextMessageBlock("user", "hi")
    record = block_to_dict(user)
    assert record == {
        "kind": "text",
        "role": "user",
        "message": "hi",
        "reasoning_content": "",
    }
    restored = _block_from_dict(record)
    assert isinstance(restored, _PlainTextMessageBlock)
    assert restored.role == "user"
    assert restored.message == "hi"
    assert restored.reasoning_content == ""
    assert restored.finished is True

    assistant = _PlainTextMessageBlock("assistant", "answer")
    assistant.reasoning_content = "why"
    restored_assistant = _block_from_dict(block_to_dict(assistant))
    assert isinstance(restored_assistant, _PlainTextMessageBlock)
    assert restored_assistant.role == "assistant"
    assert restored_assistant.message == "answer"
    assert restored_assistant.reasoning_content == "why"
    assert restored_assistant.finished is True


def test_tool_call_block_round_trips_arguments_and_parameter():
    """完成的 tool_call 块 round-trip 后 arguments/parameter/finished 一致."""
    call = _ToolCallMessageBlock("Read", "call_1")
    call.append_argument_chunk('{"file_path":')
    call.append_argument_chunk('"src/a.py"}')
    call.finish()
    record = block_to_dict(call)
    assert record == {
        "kind": "tool_call",
        "id": "call_1",
        "name": "Read",
        "arguments_string": '{"file_path":"src/a.py"}',
        "parameter": {"file_path": "src/a.py"},
        "reasoning_content": "",
        "finished": True,
        "parse_error": None,
    }
    restored = _block_from_dict(record)
    assert isinstance(restored, _ToolCallMessageBlock)
    assert restored.finished is True
    assert restored.arguments_string == call.arguments_string
    assert restored.parameter == {"file_path": "src/a.py"}


def test_unfinished_tool_call_block_stays_unfinished():
    """未 finish 的 tool_call 还原后仍是 finished=False."""
    call = _ToolCallMessageBlock("Read", "call_2")
    call.append_argument_chunk('{"file_')
    record = block_to_dict(call)
    assert record["finished"] is False
    restored = _block_from_dict(record)
    assert isinstance(restored, _ToolCallMessageBlock)
    assert restored.finished is False
    assert restored.arguments_string == '{"file_'


def test_tool_result_block_round_trips_and_keeps_the_success_flag():
    """tool_result 携带运行时决定的 success 字段."""
    ok = _ToolResultMessageBlock("call_1", "120 lines")
    record = block_to_dict(ok)
    assert record == {
        "kind": "tool_result",
        "tool_call_id": "call_1",
        "content": "120 lines",
        "success": True,
    }
    restored = _block_from_dict(record)
    assert isinstance(restored, _ToolResultMessageBlock)
    assert restored.tool_call_id == "call_1"
    assert restored.content == "120 lines"
    assert restored.success is True
    assert block_to_dict(restored)["success"] is True

    failed = _ToolResultMessageBlock("call_2", "Error: no such file", success=False)
    assert block_to_dict(failed)["success"] is False
    failed_upper = _ToolResultMessageBlock(
        "call_3", "  ERROR: permission denied", success=False
    )
    assert block_to_dict(failed_upper)["success"] is False


def test_tool_result_block_carries_the_runtime_success_flag():
    """success 由运行时决定，不再从 error: 前缀推断.

    The runtime knows whether a tool call succeeded; the content does not.
    A failure whose message does not start with "error:" -- "Invalid
    argument. missing 'path'" is one -- used to be written to disk as a
    success and read back as one.
    """
    failed = _ToolResultMessageBlock("call_1", "Invalid argument. missing 'path'")
    failed.success = False

    record = block_to_dict(failed)
    assert (
        record["success"] is False
    ), "a runtime failure must not be recorded as a success"

    restored = _block_from_dict(record)
    assert restored.success is False, "the success flag must survive the round trip"
    assert block_to_dict(restored)["success"] is False


def test_tool_result_block_does_not_infer_success_from_the_prefix():
    """前缀规则已移除：内容看起来像错误但运行时说是成功，就是成功."""
    ok = _ToolResultMessageBlock("call_1", "Error: this is part of the output")
    ok.success = True

    assert block_to_dict(ok)["success"] is True

    restored = _block_from_dict(block_to_dict(ok))
    assert restored.success is True


def test_tool_result_block_defaults_to_success():
    """没有 success 信息的块默认按成功处理."""
    block = _ToolResultMessageBlock("call_1", "output")

    assert block.success is True
    assert block_to_dict(block)["success"] is True


def test_block_from_dict_returns_none_for_unknown_or_bad_records():
    """未知 kind 或坏 dict 返回 None，不丢整段历史."""
    assert _block_from_dict({"kind": "mystery"}) is None
    assert _block_from_dict({}) is None
    assert _block_from_dict("not a dict") is None
    assert block_to_dict(object()) is None
    assert block_to_dict(None) is None


# ---------------------------------------------------------------------------
# Task 7: whole-tree serialization and restore
# ---------------------------------------------------------------------------


class _EchoProvider(_SampleChatProvider):
    """Offline provider satisfying the AgentSession constructor contract."""

    def __init__(self):
        self.stream = True

    @staticmethod
    def get_model_list():
        """Return the advertised model names."""
        return ["echo-model"]

    @classmethod
    def get_settings_schema(cls):
        """Return the settings schema."""
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @staticmethod
    def check_settings_schema(settings):
        """Accept every settings dict."""
        return True

    def init(self):
        """No async setup needed."""

    async def chat(self, tools=None, messages=None, **kwargs):
        """Never called by these tests; raise to prove it."""
        raise AssertionError("serialize/restore must not call the provider")


async def _orchestrator_with_tree():
    """Build an orchestrator with a main session and a child, main has history.

    Returns:
        ``(orchestrator, main_session, child_session)`` with the main session
        holding user text, assistant text with reasoning, a finished tool call
        and a tool result, in that order.
    """
    orchestrator = AgentOrchestrator(
        default_provider=_EchoProvider(), system_prompt="You are MitoCode."
    )
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    child = await orchestrator.create_session(
        name="Sub", description="helper", parent_id=main.agent_id
    )
    main.append_user_prompt("hello")
    main.append_reasoning_delta("thinking")
    main.append_assistant_delta("hi there")
    call = main.append_tool_call_block("Read", "call_1")
    call.append_argument_chunk('{"file_path":"src/a.py"}')
    call.finish()
    main.messages.append(_ToolResultMessageBlock("call_1", "120 lines"))
    return orchestrator, main, child


async def _orchestrator_with_clean_main():
    """Build an orchestrator whose main session has no history yet.

    Returns:
        ``(orchestrator, main_session)`` with only the seeded system block.
    """
    orchestrator = AgentOrchestrator(
        default_provider=_EchoProvider(), system_prompt="You are MitoCode."
    )
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    return orchestrator, main


@pytest.mark.asyncio
async def test_serialize_tree_produces_complete_schema():
    """serialize_tree 产出 schema 完整的记录（含两棵 agent 与消息块）."""
    orchestrator, main, child = await _orchestrator_with_tree()
    record = serialize_tree(
        orchestrator,
        main_agent_id=main.agent_id,
        session_id="20261107T152000Z-1a2b3c4d",
        title="Fix parser",
        created_at="2026-11-07T15:20:00Z",
        model="echo-model",
        workspace="/tmp/ws",
        updated_at="2026-11-07T15:35:12Z",
    )
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["session_id"] == "20261107T152000Z-1a2b3c4d"
    assert record["title"] == "Fix parser"
    assert record["created_at"] == "2026-11-07T15:20:00Z"
    assert record["updated_at"] == "2026-11-07T15:35:12Z"
    assert record["workspace"] == "/tmp/ws"
    assert record["model"] == "echo-model"
    assert record["main_agent_id"] == main.agent_id
    assert len(record["agents"]) == 2

    main_record = record["agents"][0]
    assert main_record["agent_id"] == main.agent_id
    assert main_record["name"] == "Main"
    assert main_record["parent_id"] is None
    assert main_record["sub_agent_ids"] == [child.agent_id]
    assert main_record["system_prompt"] == "You are MitoCode."
    assert [m["kind"] for m in main_record["messages"]] == [
        "text",
        "text",
        "text",
        "tool_call",
        "tool_result",
    ]
    assert main_record["messages"][0]["role"] == "system"
    assert main_record["messages"][1]["message"] == "hello"
    assert main_record["messages"][2]["reasoning_content"] == "thinking"
    assert main_record["messages"][4]["success"] is True

    child_record = record["agents"][1]
    assert child_record["agent_id"] == child.agent_id
    assert child_record["parent_id"] == main.agent_id
    # 子 agent 继承 orchestrator 的 system_prompt，因此只有一个 system 块
    assert [m["kind"] for m in child_record["messages"]] == ["text"]
    assert child_record["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_restore_agent_messages_rebuilds_history_in_order():
    """restore_agent_messages 按原序还原消息并写入 system_prompt."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    record = serialize_tree(
        orchestrator,
        main_agent_id=main.agent_id,
        session_id="s",
        title="t",
        created_at="c",
        model="m",
        workspace="/w",
        updated_at="u",
    )
    main_record = record["agents"][0]

    fresh = AgentSession(system_prompt="", provider=_EchoProvider())
    # system_prompt="" 时不追加 system 块（restore_tree 构造规则）
    assert fresh.messages == []
    restore_agent_messages(fresh, main_record)
    assert fresh.system_prompt == "You are MitoCode."
    assert len(fresh.messages) == 5
    assert isinstance(fresh.messages[0], _PlainTextMessageBlock)
    assert fresh.messages[0].role == "system"
    assert fresh.messages[1].message == "hello"
    assert fresh.messages[2].reasoning_content == "thinking"
    call = fresh.messages[3]
    assert isinstance(call, _ToolCallMessageBlock)
    assert call.id == "call_1"
    assert call.parameter == {"file_path": "src/a.py"}
    assert call.finished is True
    assert isinstance(fresh.messages[4], _ToolResultMessageBlock)
    assert fresh.messages[4].content == "120 lines"

    context = fresh.build_standard_context()
    assert context[0]["role"] == "system"
    assert context[-1]["tool_call_id"] == "call_1"


def test_restore_agent_messages_skips_unknown_blocks():
    """未知 kind 的块被跳过，不丢整段历史."""
    fresh = AgentSession(system_prompt="", provider=_EchoProvider())
    restore_agent_messages(
        fresh,
        {
            "system_prompt": "p",
            "messages": [
                {"kind": "mystery", "x": 1},
                {"kind": "text", "role": "user", "message": "ok"},
            ],
        },
    )
    assert fresh.system_prompt == "p"
    assert len(fresh.messages) == 1
    assert fresh.messages[0].role == "user"


def test_restore_agent_messages_rejects_bad_records():
    """非 mapping 或 messages 非 list 时抛 ValueError."""
    fresh = AgentSession(system_prompt="", provider=_EchoProvider())
    with pytest.raises(ValueError):
        restore_agent_messages(fresh, "not a mapping")
    with pytest.raises(ValueError):
        restore_agent_messages(fresh, {"system_prompt": "p", "messages": "nope"})


# ---------------------------------------------------------------------------
# Task 8: SessionStore.save — atomic writes with permissions
# ---------------------------------------------------------------------------


def _valid_record(session_id: str, **overrides) -> dict:
    """Return a minimal schema-valid session record for store tests.

    Args:
        session_id: Disk id for the record.
        **overrides: Fields replacing the defaults (``agents`` wins whole).

    Returns:
        A record dict accepted by ``validate_record``.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "title": "Fix parser",
        "created_at": "2026-11-07T15:00:00Z",
        "updated_at": "2026-11-07T15:20:00Z",
        "workspace": "/tmp/ws",
        "model": "echo-model",
        "main_agent_id": 0,
        "agents": [
            {
                "agent_id": 0,
                "name": "Main",
                "description": "",
                "parent_id": None,
                "sub_agent_ids": [],
                "system_prompt": "",
                "messages": [{"kind": "text", "role": "user", "message": "hi"}],
            }
        ],
    }
    record.update(overrides)
    return record


def test_save_writes_readable_json_with_tight_permissions(tmp_path):
    """save 后文件可读、0600、目录 0700."""
    store = SessionStore(tmp_path)
    path = store.save(_valid_record("20261107T152000Z-1a2b3c4d"))
    assert path == store.sessions_dir / "20261107T152000Z-1a2b3c4d.json"
    assert path.is_file()
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["session_id"] == "20261107T152000Z-1a2b3c4d"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.sessions_dir.stat().st_mode) == 0o700


def test_save_leaves_no_tmp_files_on_success(tmp_path):
    """成功路径无 .tmp-* 残留."""
    store = SessionStore(tmp_path)
    store.save(_valid_record("20261107T152000Z-1a2b3c4d"))
    residue = [
        p.name
        for p in store.sessions_dir.iterdir()
        if p.name != "20261107T152000Z-1a2b3c4d.json"
    ]
    assert residue == []


def test_save_failure_cleans_tmp_and_keeps_old_content(tmp_path, monkeypatch):
    """写入中途失败：抛错、无临时文件残留、旧内容保持."""
    store = SessionStore(tmp_path)
    session_id = "20261107T152000Z-1a2b3c4d"
    store.save(_valid_record(session_id, title="old title"))

    def _boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    import mitocode.cli.sessions as sessions_module

    monkeypatch.setattr(sessions_module.json, "dump", _boom)
    with pytest.raises(RuntimeError):
        store.save(_valid_record(session_id, title="new title"))

    residue = [p.name for p in store.sessions_dir.iterdir()]
    assert residue == [f"{session_id}.json"]
    with open(store.sessions_dir / f"{session_id}.json", encoding="utf-8") as fh:
        assert json.load(fh)["title"] == "old title"


def test_save_rejects_unsafe_session_id(tmp_path):
    """session_id 缺失或含路径分隔符时抛 SessionLoadError 且不写文件."""
    store = SessionStore(tmp_path)
    with pytest.raises(SessionLoadError):
        store.save(_valid_record("../evil"))
    with pytest.raises(SessionLoadError):
        store.save({"schema_version": SCHEMA_VERSION})
    with pytest.raises(SessionLoadError):
        store.save("not a mapping")
    assert not (tmp_path / ".mitocode").exists()


# ---------------------------------------------------------------------------
# Task 9: record validation, listing and loading
# ---------------------------------------------------------------------------


def test_validate_record_accepts_a_valid_record():
    """合法记录返回 None."""
    assert validate_record(_valid_record("20261107T152000Z-1a2b3c4d")) is None


def test_validate_record_reports_each_violation():
    """每类损坏都能给出原因字符串."""
    session_id = "20261107T152000Z-1a2b3c4d"
    good = _valid_record(session_id)

    assert validate_record("not a dict") is not None
    assert validate_record(None) is not None

    bad_version = dict(good, schema_version=2)
    assert validate_record(bad_version) is not None

    assert validate_record(dict(good, session_id="../evil")) is not None

    assert validate_record(dict(good, title=42)) is not None
    assert validate_record(dict(good, created_at=None)) is not None
    assert validate_record(dict(good, updated_at=1.5)) is not None
    assert validate_record(dict(good, model=3)) is not None
    assert validate_record(dict(good, workspace=[])) is not None

    assert validate_record(dict(good, agents=[])) is not None
    assert validate_record(dict(good, main_agent_id=7)) is not None
    assert validate_record(dict(good, main_agent_id="0")) is not None

    agent_missing_messages = [dict(good["agents"][0])]
    del agent_missing_messages[0]["messages"]
    assert validate_record(dict(good, agents=agent_missing_messages)) is not None

    bad_agent_id = [dict(good["agents"][0], agent_id="0")]
    assert validate_record(dict(good, agents=bad_agent_id)) is not None

    dangling = [
        good["agents"][0],
        {
            "agent_id": 1,
            "name": "Sub",
            "description": "",
            "parent_id": 99,
            "sub_agent_ids": [],
            "system_prompt": "",
            "messages": [],
        },
    ]
    assert validate_record(dict(good, agents=dangling)) is not None


def test_validate_record_rejects_duplicate_agent_ids():
    """两个 agent 共用同一个 agent_id 时校验必须拒绝."""
    good = _valid_record("20261107T152000Z-1a2b3c4d")
    duplicated = [
        dict(good["agents"][0]),
        {
            "agent_id": 0,
            "name": "Twin",
            "description": "",
            "parent_id": None,
            "sub_agent_ids": [],
            "system_prompt": "",
            "messages": [],
        },
    ]
    reason = validate_record(dict(good, agents=duplicated))
    assert reason is not None
    assert "duplicate" in reason.lower()


def test_validate_record_rejects_a_parent_cycle():
    """父子链成环时校验必须拒绝，而不是留给 restore_tree 报错."""
    good = _valid_record("20261107T152000Z-1a2b3c4d")
    cyclic = [
        {
            "agent_id": 0,
            "name": "Main",
            "description": "",
            "parent_id": 1,
            "sub_agent_ids": [1],
            "system_prompt": "",
            "messages": [],
        },
        {
            "agent_id": 1,
            "name": "Sub",
            "description": "",
            "parent_id": 0,
            "sub_agent_ids": [0],
            "system_prompt": "",
            "messages": [],
        },
    ]
    reason = validate_record(dict(good, agents=cyclic))
    assert reason is not None
    assert "cycle" in reason.lower()


def test_validate_record_accepts_a_valid_deep_parent_chain():
    """合法的多级父子链不应被环检测误伤."""
    good = _valid_record("20261107T152000Z-1a2b3c4d")
    chained = [
        dict(good["agents"][0], parent_id=None),
        {
            "agent_id": 1,
            "name": "Child",
            "description": "",
            "parent_id": 0,
            "sub_agent_ids": [2],
            "system_prompt": "",
            "messages": [],
        },
        {
            "agent_id": 2,
            "name": "Grandchild",
            "description": "",
            "parent_id": 1,
            "sub_agent_ids": [],
            "system_prompt": "",
            "messages": [],
        },
    ]
    assert validate_record(dict(good, agents=chained)) is None


def test_list_sessions_returns_sorted_summaries_and_skips_bad_files(tmp_path):
    """3 个合法按 updated_at 倒序；4 个坏文件计数跳过；.tmp 与非 json 忽略."""
    store = SessionStore(tmp_path)
    store.save(
        _valid_record(
            "20261107T120000Z-11111111", updated_at="2026-11-07T12:00:00Z", title="Noon"
        )
    )
    store.save(
        _valid_record(
            "20261107T150000Z-22222222",
            updated_at="2026-11-07T15:00:00Z",
            title="Afternoon",
        )
    )
    store.save(
        _valid_record(
            "20261107T090000Z-33333333",
            updated_at="2026-11-07T09:00:00Z",
            title="Morning",
        )
    )

    sessions_dir = store.sessions_dir
    (sessions_dir / "corrupt.json").write_text("{not json", encoding="utf-8")
    (sessions_dir / "wrong-version.json").write_text(
        json.dumps(_valid_record("20261107T100000Z-44444444", schema_version=99)),
        encoding="utf-8",
    )
    (sessions_dir / "no-main.json").write_text(
        json.dumps(_valid_record("20261107T110000Z-55555555", main_agent_id=9)),
        encoding="utf-8",
    )
    dangling = _valid_record("20261107T130000Z-66666666")
    dangling["agents"].append(
        {
            "agent_id": 5,
            "name": "Orphan",
            "description": "",
            "parent_id": 42,
            "sub_agent_ids": [],
            "system_prompt": "",
            "messages": [],
        }
    )
    (sessions_dir / "dangling-parent.json").write_text(
        json.dumps(dangling), encoding="utf-8"
    )
    (sessions_dir / ".tmp-partial.json").write_text("{}", encoding="utf-8")
    (sessions_dir / "notjson.txt").write_text("{}", encoding="utf-8")

    summaries, skipped = store.list_sessions()
    assert [s.session_id for s in summaries] == [
        "20261107T150000Z-22222222",
        "20261107T120000Z-11111111",
        "20261107T090000Z-33333333",
    ]
    assert [s.title for s in summaries] == ["Afternoon", "Noon", "Morning"]
    assert skipped == 4
    assert all(isinstance(s, SessionSummary) for s in summaries)
    assert summaries[0].message_count == 1
    assert summaries[0].model == "echo-model"
    assert summaries[0].created_at == "2026-11-07T15:00:00Z"


def test_list_sessions_still_lists_records_from_other_workspaces(tmp_path):
    """workspace 字段不同仍然列出（设计：只记 debug，不跳过）."""
    store = SessionStore(tmp_path)
    store.save(_valid_record("20261107T152000Z-1a2b3c4d", workspace="/elsewhere"))
    summaries, skipped = store.list_sessions()
    assert [s.session_id for s in summaries] == ["20261107T152000Z-1a2b3c4d"]
    assert skipped == 0


def test_list_sessions_on_missing_directory_returns_empty(tmp_path):
    """目录不存在时返回空列表且不计数."""
    store = SessionStore(tmp_path)
    assert store.list_sessions() == ([], 0)


def test_load_round_trips_a_saved_record(tmp_path):
    """load 返回与保存内容一致的完整 dict."""
    store = SessionStore(tmp_path)
    record = _valid_record("20261107T152000Z-1a2b3c4d")
    store.save(record)
    loaded = store.load("20261107T152000Z-1a2b3c4d")
    assert loaded == json.loads(json.dumps(record))


def test_load_raises_session_load_error_for_bad_inputs(tmp_path):
    """坏 JSON、不存在与非法 id 都抛 SessionLoadError."""
    store = SessionStore(tmp_path)
    store.save(_valid_record("20261107T152000Z-1a2b3c4d"))
    (store.sessions_dir / "20261107T130000Z-99999999.json").write_text(
        "{broken", encoding="utf-8"
    )
    (store.sessions_dir / "20261107T140000Z-88888888.json").write_text(
        json.dumps(_valid_record("20261107T140000Z-88888888", schema_version=7)),
        encoding="utf-8",
    )
    with pytest.raises(SessionLoadError):
        store.load("20261107T130000Z-99999999")
    with pytest.raises(SessionLoadError):
        store.load("20261107T140000Z-88888888")
    with pytest.raises(SessionLoadError):
        store.load("20261107T160000Z-00000000")
    with pytest.raises(SessionLoadError):
        store.load("../evil")


# ---------------------------------------------------------------------------
# Task 10: select_session — the saved-session selection panel
# ---------------------------------------------------------------------------


class _FakePromptSession:
    """Stand-in for ``PromptSession`` that replays scripted answers.

    Answers may be exceptions, which are raised instead of returned so tests
    can prove :func:`select_session` propagates them untouched.

    Attributes:
        answers: Remaining answers, popped from the front.
        messages: Messages passed to each ``prompt_async`` call.
    """

    def __init__(self, answers):
        """Store the scripted answers.

        Args:
            answers: Values or exceptions returned/raised in order.
        """
        self.answers = list(answers)
        self.messages = []

    async def prompt_async(self, message="", **kwargs):
        """Record *message* and return the next scripted answer.

        Args:
            message: Prompt shown to the user.
            **kwargs: Ignored prompt options.

        Returns:
            The next scripted answer.
        """
        self.messages.append(message)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _panel_ui(answers) -> tuple:
    """Build a UI writing to an in-memory console with scripted input.

    Args:
        answers: Values or exceptions the fake prompt session returns/raises
            in order.

    Returns:
        ``(ui, buffer, prompt_session)`` for assertions.
    """
    buffer = StringIO()
    console = Console(
        file=buffer, width=100, force_terminal=False, theme=MITOCODE_THEME
    )
    session = _FakePromptSession(answers)
    ui = CliUI(console, history_path=Path("cli-history"), prompt_session=session)
    return ui, buffer, session


def _summary(session_id: str, title: str, updated_at: str) -> SessionSummary:
    """Build one display summary for panel tests.

    Args:
        session_id: Disk id.
        title: Session title.
        updated_at: ISO-8601 UTC last-write timestamp.

    Returns:
        A ``SessionSummary`` with fixed created/model/count values.
    """
    return SessionSummary(
        session_id=session_id,
        title=title,
        updated_at=updated_at,
        created_at=updated_at,
        message_count=3,
        model="echo-model",
    )


def test_select_session_with_no_sessions_returns_new_without_panel():
    """0 个会话直接返回 new 且不打印面板."""
    ui, buffer, _ = _panel_ui([])
    assert asyncio.run(select_session(ui, [])) == "new"
    assert buffer.getvalue() == ""


@pytest.mark.asyncio
async def test_select_session_with_one_session_returns_it_directly():
    """1 个会话且不允许新建时不打印面板直接返回该 id."""
    ui, buffer, _ = _panel_ui([])
    only = _summary("id-one", "Only session", "2026-11-07T15:00:00Z")
    assert await select_session(ui, [only], allow_new=False) == "id-one"
    assert buffer.getvalue() == ""


@pytest.mark.asyncio
async def test_select_session_prints_table_and_picks_by_number():
    """≥2 个会话打印表（含 New session 行），输入 1 返回最新一个."""
    ui, buffer, session = _panel_ui(["1"])
    newest = _summary("id-new", "Afternoon", "2026-11-07T15:00:00Z")
    older = _summary("id-old", "Morning", "2026-11-07T09:00:00Z")
    assert await select_session(ui, [newest, older]) == "id-new"
    text = buffer.getvalue()
    assert "Afternoon" in text and "Morning" in text
    assert "New session" in text
    assert "echo-model" in text
    expected = (
        datetime.strptime("2026-11-07T15:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M")
    )
    assert expected in text
    assert session.messages and "[1-2, n]" in session.messages[0]


@pytest.mark.asyncio
async def test_select_session_n_returns_new():
    """输入 n 返回 new."""
    ui, _, _ = _panel_ui(["n"])
    two = [
        _summary("a", "A", "2026-11-07T15:00:00Z"),
        _summary("b", "B", "2026-11-07T14:00:00Z"),
    ]
    assert await select_session(ui, two) == "new"


@pytest.mark.asyncio
async def test_select_session_reasks_on_illegal_and_out_of_range_input():
    """非法输入 x 与越界 9 都警告后重新询问."""
    ui, buffer, _ = _panel_ui(["x", "9", "2"])
    first = _summary("a", "A", "2026-11-07T15:00:00Z")
    second = _summary("b", "B", "2026-11-07T14:00:00Z")
    assert await select_session(ui, [first, second]) == "b"
    text = buffer.getvalue()
    assert text.count("Please enter") == 2


@pytest.mark.asyncio
async def test_select_session_treats_n_as_illegal_when_new_not_allowed():
    """allow_new=False 时 n 视为非法输入."""
    ui, buffer, _ = _panel_ui(["n", "1"])
    first = _summary("a", "A", "2026-11-07T15:00:00Z")
    second = _summary("b", "B", "2026-11-07T14:00:00Z")
    assert await select_session(ui, [first, second], allow_new=False) == "a"
    assert buffer.getvalue().count("Please enter") == 1
    assert "New session" not in buffer.getvalue()


@pytest.mark.asyncio
async def test_select_session_propagates_eof_and_interrupt():
    """EOFError 与 KeyboardInterrupt 原样传播."""
    ui, _, _ = _panel_ui([EOFError()])
    two = [
        _summary("a", "A", "2026-11-07T15:00:00Z"),
        _summary("b", "B", "2026-11-07T14:00:00Z"),
    ]
    with pytest.raises(EOFError):
        await select_session(ui, two)

    ui, _, _ = _panel_ui([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        await select_session(ui, two)


@pytest.mark.asyncio
async def test_select_session_truncates_overwide_titles():
    """超宽标题被截断并以 … 结尾."""
    ui, buffer, _ = _panel_ui(["1"])
    long_title = (
        "A very long session title that keeps going and going and should be cut"
    )
    first = _summary("a", long_title, "2026-11-07T15:00:00Z")
    second = _summary("b", "B", "2026-11-07T14:00:00Z")
    await select_session(ui, [first, second])
    text = buffer.getvalue()
    assert long_title not in text
    assert long_title[:20] in text
    assert "…" in text


# ---------------------------------------------------------------------------
# Shared fakes for SessionManager tests (tasks 11-15)
# ---------------------------------------------------------------------------


class _FakeUI:
    """Record warn/note calls and expose the agent_count slot."""

    def __init__(self):
        self.warnings = []
        self.notes = []
        self.agent_count = 0

    def warn(self, message):
        """Record one warning."""
        self.warnings.append(message)

    def note(self, message):
        """Record one note."""
        self.notes.append(message)


class _FrozenClock:
    """Callable clock returning a controllable UTC moment."""

    def __init__(self, start="2026-11-07T15:00:00Z"):
        self.moment = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )

    def __call__(self):
        return self.moment

    def advance(self, seconds=1):
        """Move the frozen moment forward."""
        self.moment = self.moment + timedelta(seconds=seconds)


class _ManagerProvider:
    """Duck-typed provider for manager/title tests; captures chat calls.

    Attributes:
        calls: One dict per chat call: the clone's ``reasoning_effort`` plus
            every keyword argument the caller passed.
    """

    def __init__(
        self, default_model="echo-model", content=None, error=None, hang=False
    ):
        self.default_model = default_model
        self.reasoning_effort = "high"
        self.calls = []
        self._content = content
        self._error = error
        self._hang = hang

    async def chat(self, **kwargs):
        """Record the call, then fail, hang or answer as configured."""
        self.calls.append(
            {"reasoning_effort": getattr(self, "reasoning_effort", None), **kwargs}
        )
        if self._error is not None:
            raise self._error
        if self._hang:
            await asyncio.Event().wait()
        return {"choices": [{"message": {"content": self._content}}]}


def _manager_state(main_id=0, tool_context=None, pump=None):
    """Build a SimpleNamespace state with the fields SessionManager touches."""
    return SimpleNamespace(
        main_id=main_id,
        turn_count=3,
        tool_call_count=2,
        total_chars=100,
        agents=[],
        tool_context=tool_context,
        pump=pump,
    )


def _counting_save(store):
    """Wrap ``store.save`` with a write counter and return the counter list."""
    original = store.save
    writes = []

    def counting(record):
        writes.append(1)
        return original(record)

    store.save = counting
    return writes


def _make_manager(tmp_path, orchestrator, state, provider=None, clock=None):
    """Build a SessionManager over a real store with fake ui/clock defaults.

    Returns:
        ``(manager, ui, store, clock)``.
    """
    store = SessionStore(tmp_path)
    ui = _FakeUI()
    provider = provider if provider is not None else _ManagerProvider()
    clock = clock if clock is not None else _FrozenClock()
    manager = SessionManager(
        store,
        orchestrator,
        ui,
        state=state,
        provider=provider,
        workspace=str(tmp_path),
        clock=clock,
    )
    return manager, ui, store, clock


# ---------------------------------------------------------------------------
# Task 11: SessionManager — saving and registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_new_session_registers_id_title_and_writes_file(tmp_path):
    """start_new_session 生成新 id、标题 New Session 并落盘."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)

    await manager.start_new_session(main)

    assert is_valid_session_id(manager.current_session_id)
    assert manager.current_title == "New Session"
    path = store.sessions_dir / f"{manager.current_session_id}.json"
    assert path.is_file()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["title"] == "New Session"
    assert record["created_at"] == "2026-11-07T15:00:00Z"
    assert record["main_agent_id"] == main.agent_id


@pytest.mark.asyncio
async def test_save_current_skips_rewrite_when_content_unchanged(tmp_path):
    """内容未变时不重写文件、不更新时间戳."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    await manager.start_new_session(main)
    writes = _counting_save(store)

    second = await manager.save_current()

    assert second is None
    assert writes == []
    record = json.loads(
        (store.sessions_dir / f"{manager.current_session_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["updated_at"] == "2026-11-07T15:00:00Z"


@pytest.mark.asyncio
async def test_save_current_rewrites_and_updates_timestamp_when_changed(tmp_path):
    """内容变化时重写且 updated_at 更新."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    clock = _FrozenClock()
    manager, ui, store, _ = _make_manager(tmp_path, orchestrator, state, clock=clock)
    await manager.start_new_session(main)
    await manager.save_current()

    clock.advance(5)
    main.append_user_prompt("second message")
    path = await manager.save_current()

    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["updated_at"] == "2026-11-07T15:00:05Z"
    assert any(
        block.get("message") == "second message"
        for block in record["agents"][0]["messages"]
    )


@pytest.mark.asyncio
async def test_save_failure_warns_once_per_session(tmp_path, monkeypatch):
    """保存失败不抛出；同一会话只 warn 一次."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    await manager.start_new_session(main)

    def _boom(record):
        raise OSError("disk full")

    store.save = _boom
    await manager.save_current(force=True)
    assert len(ui.warnings) == 1

    await manager.save_current(force=True)
    assert len(ui.warnings) == 1


@pytest.mark.asyncio
async def test_save_failure_reports_every_time_when_requested(tmp_path):
    """report_failure=True 时每次失败都 warn."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    await manager.start_new_session(main)

    def _boom(record):
        raise OSError("disk full")

    store.save = _boom
    await manager.save_current(force=True, report_failure=True)
    await manager.save_current(force=True, report_failure=True)
    assert len(ui.warnings) == 2


@pytest.mark.asyncio
async def test_attach_record_adopts_id_title_and_skips_rewrite(tmp_path):
    """attach_record 认领记录的 id/标题，且未变化时不重写."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    session_id = "20261107T152000Z-1a2b3c4d"
    record = serialize_tree(
        orchestrator,
        main_agent_id=main.agent_id,
        session_id=session_id,
        title="Attach me",
        created_at="2026-11-07T15:00:00Z",
        model="echo-model",
        workspace=str(tmp_path),
        updated_at="2026-11-07T15:20:00Z",
    )
    store = SessionStore(tmp_path)
    store.save(record)
    state = _manager_state(main_id=main.agent_id)
    manager, ui, _, _ = _make_manager(tmp_path, orchestrator, state)

    manager.attach_record(record)

    assert manager.current_session_id == session_id
    assert manager.current_title == "Attach me"
    writes = _counting_save(store)
    assert await manager.save_current() is None
    assert writes == []


# ---------------------------------------------------------------------------
# Task 12: SessionAutosaver — periodic background saving
# ---------------------------------------------------------------------------


def test_autosave_interval_constant_is_thirty_seconds():
    """AUTOSAVE_INTERVAL 固定为 30.0 秒."""
    assert AUTOSAVE_INTERVAL == 30.0


@pytest.mark.asyncio
async def test_autosaver_saves_periodically_until_stopped(tmp_path):
    """有会话时按 interval 周期调用保存；stop 后任务干净退出."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    await manager.start_new_session(main)
    original_save = manager.save_current
    saves = []

    async def counting_save(*args, **kwargs):
        saves.append(1)
        return await original_save(*args, **kwargs)

    manager.save_current = counting_save

    autosaver = SessionAutosaver(manager, interval=0.01)
    autosaver.start()
    await asyncio.sleep(0.06)
    await autosaver.stop()

    assert len(saves) >= 2
    assert autosaver._task is None


@pytest.mark.asyncio
async def test_autosaver_without_session_writes_nothing(tmp_path):
    """无会话（current_session_id 为 None）时不写任何文件."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    writes = _counting_save(store)

    autosaver = SessionAutosaver(manager, interval=0.01)
    autosaver.start()
    await asyncio.sleep(0.03)
    await autosaver.stop()

    assert writes == []


@pytest.mark.asyncio
async def test_autosaver_stop_is_safe_without_start(tmp_path):
    """未 start 时 stop 幂等不抛错."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    autosaver = SessionAutosaver(manager, interval=0.01)
    await autosaver.stop()
    assert autosaver._task is None


# ---------------------------------------------------------------------------
# Task 13: generate_session_title / fallback_title
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_session_title_sends_clean_request_and_clears_reasoning():
    """请求带 system+user、stream=False、tools=[]；克隆关闭 reasoning 且原 provider 不变."""
    provider = _ManagerProvider(content='  "Fix the parser bug"\n')
    title = await generate_session_title(provider, "Fix the parser bug now")

    assert title == "Fix the parser bug"
    assert provider.reasoning_effort == "high"
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["reasoning_effort"] is None
    assert call["stream"] is False
    assert call["tools"] == []
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][1]["content"] == "Fix the parser bug now"


@pytest.mark.asyncio
async def test_generate_session_title_takes_first_line_and_truncates():
    """多行取第一行；超 60 字符截断."""
    provider = _ManagerProvider(content="Fix the parser\nsecond line")
    assert await generate_session_title(provider, "x") == "Fix the parser"

    provider = _ManagerProvider(content="x" * 70)
    assert await generate_session_title(provider, "x") == "x" * 60


@pytest.mark.asyncio
async def test_generate_session_title_raises_on_provider_error():
    """provider 抛出的异常原样传播，由调用方 fallback."""
    provider = _ManagerProvider(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await generate_session_title(provider, "x")


@pytest.mark.asyncio
async def test_generate_session_title_raises_on_empty_result():
    """空结果抛 ValueError，由调用方 fallback."""
    provider = _ManagerProvider(content="   ")
    with pytest.raises(ValueError):
        await generate_session_title(provider, "x")


def test_fallback_title_first_line_collapsed_and_truncated():
    """fallback 取首行、压缩空白、截 40 字符超出加 …；空消息返回空串."""
    assert fallback_title("Fix the\n parser bug") == "Fix the"
    assert fallback_title("  Fix   the  bug  ") == "Fix the bug"
    assert fallback_title("x" * 50) == "x" * 39 + "…"
    assert fallback_title("") == ""
    assert fallback_title("   \n  ") == ""


# ---------------------------------------------------------------------------
# Task 14: SessionManager.maybe_generate_title / shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_generate_title_updates_title_and_force_saves(tmp_path):
    """标题为 New Session 且有用户消息时启动任务；完成后写回标题并 force 保存."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    main.append_user_prompt("Fix the parser bug now")
    state = _manager_state(main_id=main.agent_id)
    provider = _ManagerProvider(content="Parser Bug")
    manager, ui, store, clock = _make_manager(
        tmp_path, orchestrator, state, provider=provider
    )
    await manager.start_new_session(main)
    writes = _counting_save(store)

    manager.maybe_generate_title()
    assert manager._title_task is not None
    await manager._title_task

    assert manager.current_title == "Parser Bug"
    assert len(writes) >= 1
    record = json.loads(
        (store.sessions_dir / f"{manager.current_session_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["title"] == "Parser Bug"


@pytest.mark.asyncio
async def test_maybe_generate_title_runs_once_per_session(tmp_path):
    """同一会话第二次调用不再启动新任务、不再请求 provider."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    main.append_user_prompt("Fix the parser bug now")
    state = _manager_state(main_id=main.agent_id)
    provider = _ManagerProvider(content="Parser Bug")
    manager, ui, store, clock = _make_manager(
        tmp_path, orchestrator, state, provider=provider
    )
    await manager.start_new_session(main)

    manager.maybe_generate_title()
    task = manager._title_task
    await task
    manager.maybe_generate_title()

    assert manager._title_task is task
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_maybe_generate_title_skips_without_user_message_or_default_title(
    tmp_path,
):
    """无用户消息或标题已非 New Session 时不启动任务."""
    orchestrator, main = await _orchestrator_with_clean_main()
    state = _manager_state(main_id=main.agent_id)
    provider = _ManagerProvider(content="Parser Bug")
    manager, ui, store, clock = _make_manager(
        tmp_path, orchestrator, state, provider=provider
    )
    await manager.start_new_session(main)

    manager.maybe_generate_title()
    assert manager._title_task is None

    main.append_user_prompt("hello")
    manager._current_title = "Already Named"
    manager.maybe_generate_title()
    assert manager._title_task is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_maybe_generate_title_discards_result_after_session_change(tmp_path):
    """会话切换后旧任务结果被丢弃，不影响新会话的标题与文件."""
    orchestrator, main = await _orchestrator_with_clean_main()
    main.append_user_prompt("Fix the parser bug now")
    state = _manager_state(main_id=main.agent_id)
    provider = _ManagerProvider(content="Parser Bug")
    manager, ui, store, clock = _make_manager(
        tmp_path, orchestrator, state, provider=provider
    )
    await manager.start_new_session(main)

    manager.maybe_generate_title()
    other = _valid_record("20261107T160000Z-abcdef01", title="Other Session")
    store.save(other)
    manager.attach_record(other)
    await manager._title_task

    assert manager.current_title == "Other Session"
    assert len(provider.calls) == 1
    reloaded = json.loads(
        (store.sessions_dir / "20261107T160000Z-abcdef01.json").read_text(
            encoding="utf-8"
        )
    )
    assert reloaded["title"] == "Other Session"


@pytest.mark.asyncio
async def test_maybe_generate_title_falls_back_on_provider_error(tmp_path):
    """provider 失败时标题回退为用户消息首行."""
    orchestrator, main = await _orchestrator_with_clean_main()
    main.append_user_prompt("Fix the parser bug now")
    state = _manager_state(main_id=main.agent_id)
    provider = _ManagerProvider(error=RuntimeError("boom"))
    manager, ui, store, clock = _make_manager(
        tmp_path, orchestrator, state, provider=provider
    )
    await manager.start_new_session(main)

    manager.maybe_generate_title()
    await manager._title_task

    assert manager.current_title == fallback_title("Fix the parser bug now")


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_title_task(tmp_path):
    """shutdown 取消未完成的标题任务并 await，不抛异常."""
    orchestrator, main = await _orchestrator_with_clean_main()
    main.append_user_prompt("Fix the parser bug now")
    state = _manager_state(main_id=main.agent_id)
    provider = _ManagerProvider(hang=True)
    manager, ui, store, clock = _make_manager(
        tmp_path, orchestrator, state, provider=provider
    )
    await manager.start_new_session(main)

    manager.maybe_generate_title()
    task = manager._title_task
    assert task is not None and not task.done()

    await manager.shutdown()

    assert task.done()
    assert task.cancelled()


# ---------------------------------------------------------------------------
# Task 15: SessionManager.switch_to / switch_new
# ---------------------------------------------------------------------------


class _FakeSwitchOrchestrator:
    """Small orchestrator fake with real create/restore/reset behaviour.

    Attributes:
        sessions: Live sessions keyed by agent id.
        counter: Next agent id a create allocates.
        reset_count: How many times ``reset_sessions`` ran.
        fail_restore: When set, ``restore_tree`` inserts the first agent and
            then raises, leaving half-restored state for the caller to clean.
        fail_create: When set, ``create_session`` raises instead of
            registering a session.
    """

    def __init__(self):
        self.sessions = {}
        self.counter = 0
        self.reset_count = 0
        self.fail_restore = False
        self.fail_create = False

    async def reset_sessions(self):
        """Clear the registry and restart ids at zero."""
        self.reset_count += 1
        self.sessions.clear()
        self.counter = 0

    async def create_session(self, name="", description="", **kwargs):
        """Create and register one session with a fresh id."""
        if self.fail_create:
            raise ValueError("create boom")
        session = AgentSession(
            agent_id=self.counter,
            name=name,
            description=description,
            provider=_EchoProvider(),
            system_prompt="You are MitoCode.",
        )
        self.sessions[self.counter] = session
        self.counter += 1
        return session

    async def restore_tree(self, agents):
        """Rebuild sessions from records, keeping original ids."""
        if self.fail_restore:
            # Fail only the first attempt: a rollback restores the previous
            # tree through the same method and must be able to succeed.
            self.fail_restore = False
            first = agents[0]
            self.sessions[int(first["agent_id"])] = AgentSession(
                agent_id=int(first["agent_id"]),
                name=str(first.get("name", "") or ""),
                parent_id=first.get("parent_id"),
                provider=_EchoProvider(),
                system_prompt="",
            )
            raise ValueError("restore boom")
        ids = [int(agent["agent_id"]) for agent in agents]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate agent ids")
        sessions = {}
        for agent in agents:
            sessions[int(agent["agent_id"])] = AgentSession(
                agent_id=int(agent["agent_id"]),
                name=str(agent.get("name", "") or ""),
                description=str(agent.get("description", "") or ""),
                parent_id=agent.get("parent_id"),
                provider=_EchoProvider(),
                system_prompt="",
            )
        for agent in agents:
            parent_id = agent.get("parent_id")
            if parent_id is not None:
                sessions[parent_id].sub_agent_ids.append(int(agent["agent_id"]))
        self.sessions = sessions
        self.counter = max(ids) + 1
        return dict(sessions)

    def get_session(self, agent_id):
        """Return the session under *agent_id* or ``None``."""
        return self.sessions.get(agent_id)

    def get_tree_snapshot(self):
        """Return the current tree as node dicts."""
        return [s.to_agent_node().to_dict() for s in self.sessions.values()]


class _FakeBackground:
    """Record whether ``stop_all`` was awaited."""

    def __init__(self):
        self.stopped = False

    async def stop_all(self):
        """Mark the registry as stopped."""
        self.stopped = True


def _switch_record(session_id="20261107T170000Z-7e7e7e7e", workspace="/tmp/ws"):
    """Return a valid record for switch tests whose main holds one user text.

    The record mirrors what ``serialize_tree`` emits for the fake
    orchestrator's restored tree, so an unchanged post-switch save is a no-op.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "title": "Other Session",
        "created_at": "2026-11-07T17:00:00Z",
        "updated_at": "2026-11-07T17:10:00Z",
        "workspace": workspace,
        "model": "echo-model",
        "main_agent_id": 0,
        "agents": [
            {
                "agent_id": 0,
                "name": "Main",
                "description": "Interactive session",
                "parent_id": None,
                "sub_agent_ids": [],
                "system_prompt": "You are MitoCode.",
                "messages": [
                    {
                        "kind": "text",
                        "role": "user",
                        "message": "restored message",
                        "reasoning_content": "",
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_switch_to_saves_old_tree_and_restores_new_one(tmp_path):
    """切换在锁内完成：保存旧树 → reset → 恢复 → 状态与登记全部更新."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)
    old_id = manager.current_session_id
    main.append_user_prompt("old message")
    await manager.save_current()
    old_record = json.loads(
        (store.sessions_dir / f"{old_id}.json").read_text(encoding="utf-8")
    )
    assert any(
        block.get("message") == "old message"
        for block in old_record["agents"][0]["messages"]
    )

    record = _switch_record(workspace=str(tmp_path))
    assert await manager.switch_to(record) is True

    assert manager.current_session_id == "20261107T170000Z-7e7e7e7e"
    assert manager.current_title == "Other Session"
    assert state.main_id == 0
    assert orchestrator.reset_count == 1
    restored = orchestrator.get_session(0)
    assert restored.system_prompt == "You are MitoCode."
    assert [block.message for block in restored.messages] == ["restored message"]
    assert state.turn_count == 0
    assert state.tool_call_count == 0
    assert state.total_chars == 0
    assert state.agents == orchestrator.get_tree_snapshot()
    assert ui.agent_count == len(state.agents)
    # 恢复即认领指纹：紧随的未变化保存不重写
    writes = _counting_save(store)
    assert await manager.save_current() is None
    assert writes == []


@pytest.mark.asyncio
async def test_switch_to_failure_cleans_registry_and_keeps_current(tmp_path):
    """恢复失败返回 False：半恢复残留被清掉，当前会话被还原."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)
    old_id = manager.current_session_id
    orchestrator.fail_restore = True

    record = _switch_record(workspace=str(tmp_path))
    assert await manager.switch_to(record) is False

    assert manager.current_session_id == old_id
    assert manager.current_title == "New Session"
    assert state.main_id == main.agent_id
    # The half-restored leftovers are gone and the previous tree is back,
    # so the registry is not left empty.
    assert orchestrator.sessions
    assert orchestrator.get_session(main.agent_id) is not None


@pytest.mark.asyncio
async def test_switch_to_failure_keeps_the_live_session_usable(tmp_path):
    """恢复失败后当前会话仍能取到，且后续保存不再静默失效."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)
    orchestrator.fail_restore = True

    record = _switch_record(workspace=str(tmp_path))
    assert await manager.switch_to(record) is False

    # The live session must still be reachable through the orchestrator, so
    # the next turn can run instead of raising "Main agent #N does not exist".
    assert orchestrator.get_session(state.main_id) is not None

    # And saving must still reach the disk instead of silently no-oping.
    orchestrator.fail_restore = False
    main.append_user_prompt("after the failed switch")
    path = await manager.save_current(force=True)
    assert path is not None
    assert path.is_file()


@pytest.mark.asyncio
async def test_switch_to_failure_restores_the_previous_session(tmp_path):
    """恢复失败时旧会话被重建，而不是留下一个空注册表."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)
    main.append_user_prompt("keep me")
    orchestrator.fail_restore = True

    record = _switch_record(workspace=str(tmp_path))
    assert await manager.switch_to(record) is False

    # The old session is back in the registry with its messages intact.
    restored = orchestrator.get_session(state.main_id)
    assert restored is not None
    assert any(
        getattr(block, "message", "") == "keep me" for block in restored.messages
    )


@pytest.mark.asyncio
async def test_switch_new_failure_keeps_the_cli_usable(tmp_path):
    """switch_new 建新会话失败时不得把异常抛给 REPL，也不得留下空注册表."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)

    orchestrator.fail_create = True
    # The failure must be contained: no exception escapes to the REPL.
    await manager.switch_new()

    assert orchestrator.get_session(state.main_id) is not None


@pytest.mark.asyncio
async def test_switch_to_stops_background_tasks(tmp_path):
    """state.tool_context 带 background 时切换会 await stop_all."""
    orchestrator = _FakeSwitchOrchestrator()
    background = _FakeBackground()
    state = _manager_state(
        main_id=0, tool_context=SimpleNamespace(background=background)
    )
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)

    assert await manager.switch_to(_switch_record(workspace=str(tmp_path))) is True
    assert background.stopped is True


@pytest.mark.asyncio
async def test_switch_to_drains_real_pump_without_hanging(tmp_path):
    """state.pump 为真实 EventPump（未启动任务）时 wait_idle 正常返回."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    pump = EventPump(asyncio.Queue(), _FakeUI(), state)
    state.pump = pump
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)

    assert await manager.switch_to(_switch_record(workspace=str(tmp_path))) is True


class _OrderRecordingPump:
    """Pump stand-in that records when the drain happened."""

    def __init__(self, events):
        self.events = events

    async def wait_idle(self):
        """Record that the pump was drained."""
        self.events.append("drain")


class _OrderRecordingOrchestrator(_FakeSwitchOrchestrator):
    """Orchestrator fake that records when the tree was reset."""

    def __init__(self, events):
        super().__init__()
        self.events = events

    async def reset_sessions(self):
        """Record the reset before clearing the registry."""
        self.events.append("reset")
        await super().reset_sessions()


@pytest.mark.asyncio
async def test_switch_to_stops_the_event_source_before_draining(tmp_path):
    """切换先停事件源再排空：排空一个仍在入队的活跃 turn 会一直等."""
    events = []
    orchestrator = _OrderRecordingOrchestrator(events)
    state = _manager_state(main_id=0)
    state.pump = _OrderRecordingPump(events)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)

    assert await manager.switch_to(_switch_record(workspace=str(tmp_path))) is True

    # The event source must be stopped first: draining a queue a live turn
    # keeps filling can block the switch indefinitely.
    assert events.index("reset") < events.index("drain")


@pytest.mark.asyncio
async def test_switch_new_stops_the_event_source_before_draining(tmp_path):
    """switch_new 同样先停事件源再排空."""
    events = []
    orchestrator = _OrderRecordingOrchestrator(events)
    state = _manager_state(main_id=0)
    state.pump = _OrderRecordingPump(events)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)

    await manager.switch_new()

    assert events.index("reset") < events.index("drain")


@pytest.mark.asyncio
async def test_switch_new_destroys_old_tree_and_starts_fresh(tmp_path):
    """switch_new 销毁旧树、新建 #0 Main，新 id 与 New Session 标题落盘."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)
    old_id = manager.current_session_id
    main.append_user_prompt("old message")
    state.turn_count = 7
    state.tool_call_count = 3
    state.total_chars = 500

    await manager.switch_new()

    assert manager.current_session_id != old_id
    assert is_valid_session_id(manager.current_session_id)
    assert manager.current_title == "New Session"
    assert state.main_id == 0
    new_main = orchestrator.get_session(0)
    assert new_main is not None and new_main.name == "Main"
    assert orchestrator.reset_count == 1
    assert state.turn_count == 0
    assert state.tool_call_count == 0
    assert state.total_chars == 0
    assert state.agents == orchestrator.get_tree_snapshot()
    assert ui.agent_count == 1
    record = json.loads(
        (store.sessions_dir / f"{manager.current_session_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["title"] == "New Session"


@pytest.mark.asyncio
async def test_switch_lock_blocks_concurrent_save(tmp_path):
    """锁被持有时 save_current 等待，不与切换交错."""
    orchestrator = _FakeSwitchOrchestrator()
    state = _manager_state(main_id=0)
    manager, ui, store, clock = _make_manager(tmp_path, orchestrator, state)
    main = await orchestrator.create_session(
        name="Main", description="Interactive session"
    )
    state.main_id = main.agent_id
    await manager.start_new_session(main)
    main.append_user_prompt("more content")

    await manager._lock.acquire()
    try:
        task = asyncio.create_task(manager.save_current())
        await asyncio.sleep(0.02)
        assert not task.done()
    finally:
        manager._lock.release()

    path = await task
    assert path is not None
    assert path.is_file()


def test_display_messages_from_record_rebuilds_blocks(tmp_path):
    """/demo 展示消息从 record 重建块实例，未知 kind 跳过，空 record 返回空."""
    from mitocode.cli.sessions import display_messages_from_record

    record = {
        "schema_version": 1,
        "session_id": "20261107T100000Z-11111111",
        "title": "T",
        "created_at": "2026-11-07T09:00:00Z",
        "updated_at": "2026-11-07T10:00:00Z",
        "workspace": "/tmp/w",
        "model": "m",
        "main_agent_id": 0,
        "agents": [
            {
                "agent_id": 0,
                "name": "Main",
                "description": "Interactive session",
                "parent_id": None,
                "system_prompt": "",
                "messages": [
                    {
                        "kind": "text",
                        "role": "user",
                        "message": "hello",
                        "reasoning_content": "",
                    },
                    {"kind": "mystery", "junk": True},
                ],
            }
        ],
    }
    blocks = display_messages_from_record(record)
    assert len(blocks) == 1
    assert getattr(blocks[0], "role", "") == "user"
    assert getattr(blocks[0], "message", "") == "hello"
    assert display_messages_from_record({"agents": [], "main_agent_id": 0}) == []


@pytest.mark.asyncio
async def test_saved_records_do_not_contain_the_api_key(tmp_path):
    """会话记录不得写入 API key：工具输出可能包含凭据文件内容."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    store = SessionStore(tmp_path)
    manager = SessionManager(
        store,
        orchestrator,
        _FakeUI(),
        state=state,
        provider=_ManagerProvider(),
        workspace=str(tmp_path),
        clock=_FrozenClock(),
        redact_secrets=("sk-super-secret-key",),
    )
    await manager.start_new_session(main)

    secret = "sk-super-secret-key"
    main.messages.append(
        _ToolResultMessageBlock("call_1", f"the key is {secret}", success=True)
    )

    path = await manager.save_current(force=True)

    assert path is not None
    written = path.read_text(encoding="utf-8")
    assert secret not in written, (
        "the API key must never be written into a session record; the record is "
        "read back and echoed at startup"
    )
    assert "***" in written or "REDACTED" in written


@pytest.mark.asyncio
async def test_saved_records_redact_the_key_from_tool_arguments(tmp_path):
    """工具参数里的凭据同样不得落盘."""
    orchestrator, main, _ = await _orchestrator_with_tree()
    state = _manager_state(main_id=main.agent_id)
    store = SessionStore(tmp_path)
    manager = SessionManager(
        store,
        orchestrator,
        _FakeUI(),
        state=state,
        provider=_ManagerProvider(),
        workspace=str(tmp_path),
        clock=_FrozenClock(),
        redact_secrets=("sk-super-secret-key",),
    )
    await manager.start_new_session(main)

    call = _ToolCallMessageBlock("Read", "call_2")
    call.arguments_string = '{"api_key": "sk-super-secret-key"}'
    call.parameter = {"api_key": "sk-super-secret-key"}
    call.finished = True
    main.messages.append(call)

    path = await manager.save_current(force=True)

    assert path is not None
    assert "sk-super-secret-key" not in path.read_text(encoding="utf-8")


def test_list_sessions_reports_a_directory_that_cannot_be_read(tmp_path, monkeypatch):
    """会话目录无法枚举时必须报错，而不是让异常穿透或静默当成空目录."""
    store = SessionStore(tmp_path)
    store.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _boom():
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(
        type(store.sessions_dir), "iterdir", lambda self: _boom(), raising=False
    )

    with pytest.raises(SessionLoadError) as info:
        store.list_sessions()

    assert str(store.sessions_dir) in str(info.value)


@pytest.mark.asyncio
async def test_select_session_offers_new_with_one_session_when_allowed():
    """allow_new=True 时即使只有一个会话也要显示面板，让 n 可达."""
    ui, buffer, _ = _panel_ui(["n"])
    only = _summary("id-one", "Only session", "2026-11-07T15:00:00Z")

    assert await select_session(ui, [only], allow_new=True) == "new"
    assert "New session" in buffer.getvalue(), (
        "the new-session row must be offered; otherwise the user cannot start "
        "a new session while one is saved"
    )


@pytest.mark.asyncio
async def test_select_session_with_one_session_and_no_new_stays_quiet():
    """allow_new=False 时单会话仍直接返回，不打印面板."""
    ui, buffer, _ = _panel_ui([])
    only = _summary("id-one", "Only session", "2026-11-07T15:00:00Z")

    assert await select_session(ui, [only], allow_new=False) == "id-one"
    assert buffer.getvalue() == ""
