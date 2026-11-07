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
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from rich.console import Console

from mitocode.cli.sessions import (
    SCHEMA_VERSION,
    SessionLoadError,
    SessionStore,
    SessionSummary,
    _block_from_dict,
    block_to_dict,
    is_valid_session_id,
    new_session_id,
    restore_agent_messages,
    select_session,
    serialize_tree,
    validate_record,
)

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


def test_tool_result_block_round_trips_and_infers_success():
    """tool_result 无 success 字段；按 error: 前缀推定 success."""
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
    assert block_to_dict(restored)["success"] is True

    failed = _ToolResultMessageBlock("call_2", "Error: no such file")
    assert block_to_dict(failed)["success"] is False
    failed_upper = _ToolResultMessageBlock("call_3", "  ERROR: permission denied")
    assert block_to_dict(failed_upper)["success"] is False


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
    """1 个会话不打印面板直接返回该 id."""
    ui, buffer, _ = _panel_ui([])
    only = _summary("id-one", "Only session", "2026-11-07T15:00:00Z")
    assert await select_session(ui, [only]) == "id-one"
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
