import asyncio
import os
import sys
import unittest.mock as mock
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.agent.agent_runtime import AgentRuntime
from mitocode.core.agent.agent_session import AgentSession
from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
from mitocode.core.agent.agent_state import AgentState
from mitocode.core.bridge.event_adapter import EventBridge
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.core.tool.tool_manager import ToolManager


class FakeChunk:
    def __init__(self, content=None, finish_reason=None):
        self.choices = [mock.MagicMock()]
        self.choices[0].delta.content = content
        self.choices[0].delta.tool_calls = None
        self.choices[0].finish_reason = finish_reason


class EchoProvider(_SampleChatProvider):
    def __init__(self):
        self.stream = True

    @staticmethod
    def get_model_list():
        return ["echo-model"]

    @classmethod
    def get_settings_schema(cls):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @staticmethod
    def check_settings_schema(settings):
        return True

    def init(self):
        pass

    async def chat(self, tools=None, messages=None, **kwargs):
        async def _stream():
            last_msg = messages[-1]["content"] if messages else ""
            words = last_msg.split()
            for i, word in enumerate(words):
                yield FakeChunk(content=word + (" " if i < len(words) - 1 else ""))
            yield FakeChunk(finish_reason="stop")

        return _stream()


@pytest.mark.asyncio
async def test_orchestrator_create_and_send_message():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    agent_id = await orchestrator.create_agent(name="TestAgent")

    snapshot = orchestrator.get_tree_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["id"] == agent_id
    assert snapshot[0]["name"] == "TestAgent"
    assert snapshot[0]["state"] == "idle"

    task = await orchestrator.send_message(agent_id, "hello world")
    assert task is not None

    events = []
    while True:
        try:
            agent_id_evt, event = await asyncio.wait_for(
                orchestrator.event_queue.get(), timeout=2.0
            )
            events.append((agent_id_evt, event))
            if agent_id_evt < 0 and event.get("type") == "sync":
                snapshot = event.get("agents", [])
                agent_state = next(
                    (a["state"] for a in snapshot if a["id"] == agent_id), None
                )
                if agent_state in ("finished", "cancelled", "error"):
                    break
        except asyncio.TimeoutError:
            break

    token_events = [e for aid, e in events if aid == agent_id and e["type"] == "token"]
    text = "".join(e["content"] for e in token_events)
    assert "hello" in text
    assert "world" in text


@pytest.mark.asyncio
async def test_orchestrator_cancel_agent():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    agent_id = await orchestrator.create_agent(name="CancelTest")

    await orchestrator.send_message(agent_id, "test")
    await orchestrator.cancel_agent(agent_id)

    session = orchestrator._sessions[agent_id]
    assert session.is_cancelled()


@pytest.mark.asyncio
async def test_orchestrator_parent_child_tree():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    parent_id = await orchestrator.create_agent(name="Parent")
    child_id = await orchestrator.create_agent(name="Child", parent_id=parent_id)

    snapshot = orchestrator.get_tree_snapshot()
    assert len(snapshot) == 2

    parent = next(a for a in snapshot if a["id"] == parent_id)
    child = next(a for a in snapshot if a["id"] == child_id)
    assert child_id in parent["sub_agent"]


@pytest.mark.asyncio
async def test_create_session_registers_and_returns_session():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())

    session = await orchestrator.create_session(name="Worker", description="does work")

    assert isinstance(session, AgentSession)
    assert session.agent_id == 0
    assert session.name == "Worker"
    assert session.description == "does work"
    assert session.parent_id is None
    assert orchestrator.get_session(session.agent_id) is session

    snapshot = orchestrator.get_tree_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["id"] == session.agent_id
    assert snapshot[0]["name"] == "Worker"
    assert snapshot[0]["description"] == "does work"


@pytest.mark.asyncio
async def test_create_session_links_child_to_parent():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    parent = await orchestrator.create_session(name="Parent")
    child = await orchestrator.create_session(name="Child", parent_id=parent.agent_id)

    assert child.parent_id == parent.agent_id
    assert child.agent_id in parent.sub_agent_ids

    snapshot = orchestrator.get_tree_snapshot()
    parent_node = next(a for a in snapshot if a["id"] == parent.agent_id)
    assert child.agent_id in parent_node["sub_agent"]


@pytest.mark.asyncio
async def test_create_session_without_provider_raises():
    orchestrator = AgentOrchestrator()
    with pytest.raises(ValueError):
        await orchestrator.create_session()


@pytest.mark.asyncio
async def test_create_agent_alias_matches_create_session_id():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    agent_id = await orchestrator.create_agent(name="Alias")
    session = orchestrator.get_session(agent_id)
    assert session is not None
    assert session.name == "Alias"


def test_event_bridge_token():
    bridge = EventBridge()
    result = bridge.adapt({"event": "agent.text", "delta": "hello"}, agent_id=1)
    assert result == {"type": "token", "agent_id": 1, "content": "hello"}


def test_event_bridge_tool_call_header():
    tm = ToolManager()

    @tm.tool(
        human_doc="Write a file", display_action="file.write", display_param="file_path"
    )
    def write_file(file_path: str, content: str) -> str:
        return "ok"

    bridge = EventBridge(tm)
    result = bridge.adapt(
        {"event": "agent.tool_call.header", "id": "call_1", "name": "write_file"},
        agent_id=1,
    )
    assert result["type"] == "tool_call"
    assert result["action"] == "file.write"
    assert result["description"] == "Write a file"


def test_event_bridge_tool_call_delta():
    bridge = EventBridge()
    result = bridge.adapt(
        {"event": "agent.tool_call.delta", "id": "call_1", "delta": '{"path":'},
        agent_id=1,
    )
    assert result == {
        "type": "tool_call",
        "agent_id": 1,
        "id": "call_1",
        "content": '{"path":',
    }


def test_event_bridge_tool_call_result():
    bridge = EventBridge()
    result = bridge.adapt(
        {
            "event": "agent.tool_call.result",
            "id": "call_1",
            "success": True,
            "result": "done",
        },
        agent_id=1,
    )
    assert result == {
        "type": "tool_result",
        "agent_id": 1,
        "id": "call_1",
        "result": "done",
        "success": True,
    }


def test_event_bridge_tool_start_carries_args_and_display():
    manager = ToolManager()
    manager.register(
        lambda **kw: "",
        name="Read",
        human_doc="read",
        display_action="file.read",
        display_param="file_path",
    )
    bridge = EventBridge(manager)
    result = bridge.adapt(
        {
            "event": "agent.tool_call.execution",
            "id": "call_1",
            "name": "Read",
            "args": {"file_path": "a.py"},
        },
        agent_id=2,
    )
    assert result == {
        "type": "tool_start",
        "agent_id": 2,
        "id": "call_1",
        "name": "Read",
        "args": {"file_path": "a.py"},
        "action": "file.read",
        "description": "read",
        "display_param": "file_path",
    }


def test_event_bridge_forwards_turn_end_and_cancel():
    bridge = EventBridge()
    assert bridge.adapt(
        {"event": "system.finished.standard", "reason": "stop"}, agent_id=3
    ) == {"type": "turn_end", "agent_id": 3, "reason": "stop"}
    assert bridge.adapt({"event": "system.cancelled"}, agent_id=3) == {
        "type": "turn_cancelled",
        "agent_id": 3,
    }


def test_event_bridge_header_and_delta_and_result_keep_call_id():
    bridge = EventBridge()
    header = bridge.adapt(
        {"event": "agent.tool_call.header", "id": "c1", "name": "Read"}, 1
    )
    assert header["id"] == "c1" and header["name"] == "Read"
    delta = bridge.adapt(
        {"event": "agent.tool_call.delta", "id": "c1", "delta": "{"}, 1
    )
    assert delta["id"] == "c1"
    result = bridge.adapt(
        {
            "event": "agent.tool_call.result",
            "id": "c1",
            "success": True,
            "result": "ok",
        },
        1,
    )
    assert result["id"] == "c1"


def test_event_bridge_skips_internal():
    bridge = EventBridge()
    assert bridge.adapt({"event": "system.finished.tool_calls"}, agent_id=1) is None


class _BoomProvider(EchoProvider):
    """Provider that raises an unexpected exception carrying a fake API key."""

    async def chat(self, tools=None, messages=None, **kwargs):
        raise RuntimeError("boom sk-abc123")


@pytest.mark.asyncio
async def test_run_session_emits_error_event_and_redacts_key():
    orchestrator = AgentOrchestrator(default_provider=_BoomProvider())
    agent_id = await orchestrator.create_agent(name="boom")
    task = await orchestrator.send_message(agent_id, "hi")
    await task
    events = []
    while not orchestrator.event_queue.empty():
        events.append(orchestrator.event_queue.get_nowait())
    errors = [e for _, e in events if e.get("type") == "error"]
    assert errors and errors[0]["agent_id"] == agent_id
    assert "sk-abc123" not in errors[0]["message"]
    assert "sk-***" in errors[0]["message"]
    assert orchestrator.get_session(agent_id).state is AgentState.ERROR


@pytest.mark.asyncio
async def test_system_prompt_reaches_every_session():
    orchestrator = AgentOrchestrator(
        default_provider=EchoProvider(), system_prompt="SYS"
    )
    parent = await orchestrator.create_agent(name="p")
    child = await orchestrator.create_agent(name="c", parent_id=parent)
    for agent_id in (parent, child):
        first = orchestrator.get_session(agent_id).build_standard_context()[0]
        assert first == {"role": "system", "content": "SYS"}


@pytest.mark.asyncio
async def test_system_prompt_is_empty_by_default_and_overridable():
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="no prompt")
    assert session.system_prompt == ""
    assert session.build_standard_context() == []

    overridden = await orchestrator.create_session(
        name="override", system_prompt="OTHER"
    )
    assert overridden.build_standard_context()[0] == {
        "role": "system",
        "content": "OTHER",
    }


@pytest.mark.asyncio
async def test_redact_api_keys_leaves_other_text_intact():
    assert (
        AgentOrchestrator.redact_api_keys("bad sk-a1_B-2 and sk-zzz! done")
        == "bad sk-*** and sk-***! done"
    )
    assert AgentOrchestrator.redact_api_keys("no secrets here") == "no secrets here"


# ──────────────────────────────────────────────
# EventBridge: 未知事件与缺键默认值
# ──────────────────────────────────────────────


class TestEventBridgeSkipUnknownEvents:
    """EventBridge 对没有类型或类型未知的事件返回 None 且不改动输入."""

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param({"id": "c1", "name": "Read"}, id="missing-event-key"),
            pytest.param({"event": "agent.reasoning", "id": "c1"}, id="unknown-type"),
        ],
    )
    def test_adapt_missing_or_unknown_event_returns_none(self, raw):
        """缺 event 键或未知事件类型 → 返回 None 表示不转发."""
        bridge = EventBridge()
        snapshot = dict(raw)

        result = bridge.adapt(raw, agent_id=8)

        assert result is None, f"事件 {raw} 不应被转发到 TUI"
        assert raw == snapshot, "跳过事件不应改动调用方传入的原始 dict"


class TestEventBridgeHeaderDefaults:
    """header 事件的 id 在缺键时回退为空串且其余字段保持文档化取值."""

    def test_event_bridge_header_missing_id_defaults_to_empty_string(self):
        """header 事件缺少 id 键 → id 回退为空串而 action/description 仍取工具名."""
        bridge = EventBridge()

        result = bridge.adapt(
            {"event": "agent.tool_call.header", "name": "Read"}, agent_id=8
        )

        assert result == {
            "type": "tool_call",
            "agent_id": 8,
            "id": "",
            "name": "Read",
            "action": "Read",
            "description": "Read",
            "additional": "",
        }, "缺 id 的 header 应回退为空串且其余字段保持文档化取值"


# ──────────────────────────────────────────────
# EventBridge: 部分元数据回退
# ──────────────────────────────────────────────


class TestEventBridgePartialMetadataFallback:
    """已注册工具的 action/description/display_param 三个回退相互独立."""

    def test_adapt_registered_tool_with_action_but_no_doc_keeps_action(self):
        """display_action 非空但 human_doc 为空 → action 保留，description 回退为工具名."""
        manager = ToolManager()
        manager.register(lambda **kw: "", name="sync_tool", display_action="file.write")
        bridge = EventBridge(manager)

        header = bridge.adapt(
            {"event": "agent.tool_call.header", "id": "c1", "name": "sync_tool"},
            agent_id=9,
        )

        assert header["action"] == "file.write", "非空 display_action 应优先于工具名"
        assert (
            header["description"] == "sync_tool"
        ), "空 human_doc 应独立回退为工具名而不影响 action"
        assert header["additional"] == "", "header 的 additional 字段恒为空串"

    def test_adapt_registered_tool_with_display_param_but_no_doc_keeps_param(self):
        """display_param 非空但 human_doc 为空 → tool_start 保留 display_param."""
        manager = ToolManager()
        manager.register(lambda **kw: "", name="sync_tool", display_param="file_path")
        bridge = EventBridge(manager)

        started = bridge.adapt(
            {
                "event": "agent.tool_call.execution",
                "id": "c1",
                "name": "sync_tool",
                "args": {"file_path": "a.py"},
            },
            agent_id=9,
        )

        assert started["display_param"] == "file_path", "非空 display_param 应被透出"
        assert (
            started["description"] == "sync_tool"
        ), "空 human_doc 不应掩盖 display_param 的透出"
        assert started["action"] == "sync_tool", "空 display_action 应回退为工具名"


# ──────────────────────────────────────────────
# EventBridge: 元数据回退
# ──────────────────────────────────────────────


def test_event_bridge_unknown_tool_with_manager_falls_back_to_tool_name():
    """ToolManager 存在但工具未注册 → action/description 回退为工具名，display_param 为空."""
    manager = ToolManager()
    manager.register(lambda **kw: "", name="known")
    bridge = EventBridge(manager)

    header = bridge.adapt(
        {"event": "agent.tool_call.header", "id": "c9", "name": "missing"}, agent_id=4
    )
    assert header["action"] == "missing", "未注册工具的 action 应回退为工具名"
    assert header["description"] == "missing", "未注册工具的 description 应回退为工具名"
    assert header["additional"] == "", "header 的 additional 字段恒为空串"

    started = bridge.adapt(
        {
            "event": "agent.tool_call.execution",
            "id": "c9",
            "name": "missing",
            "args": {},
        },
        agent_id=4,
    )
    assert started["display_param"] == "", "未注册工具不应暴露 display_param"


def test_event_bridge_registered_tool_without_display_metadata_falls_back():
    """已注册工具但 display_action/human_doc 为空 → action/description 回退为工具名."""
    manager = ToolManager()
    manager.register(lambda **kw: "", name="bare")
    bridge = EventBridge(manager)

    header = bridge.adapt(
        {"event": "agent.tool_call.header", "id": "c1", "name": "bare"}, agent_id=5
    )
    assert header["action"] == "bare", "空 display_action 应回退为工具名"
    assert header["description"] == "bare", "空 human_doc 应回退为工具名"

    started = bridge.adapt(
        {"event": "agent.tool_call.execution", "id": "c1", "name": "bare", "args": {}},
        agent_id=5,
    )
    assert started["display_param"] == "", "空 display_param 应为空串"


@pytest.mark.parametrize(
    "event, expected",
    [
        pytest.param(
            {"event": "agent.text"},
            {"type": "token", "content": ""},
            id="text-missing-delta",
        ),
        pytest.param(
            {"event": "agent.tool_call.header", "id": "c1"},
            {
                "type": "tool_call",
                "id": "c1",
                "name": "",
                "action": "",
                "description": "",
                "additional": "",
            },
            id="header-missing-name",
        ),
        pytest.param(
            {"event": "agent.tool_call.delta", "id": "c1"},
            {"type": "tool_call", "id": "c1", "content": ""},
            id="delta-missing-delta",
        ),
        pytest.param(
            {"event": "agent.tool_call.execution", "id": "c1", "name": "Read"},
            {
                "type": "tool_start",
                "id": "c1",
                "name": "Read",
                "args": {},
                "action": "Read",
                "description": "Read",
                "display_param": "",
            },
            id="execution-missing-args",
        ),
        pytest.param(
            {"event": "agent.tool_call.result", "id": "c1", "result": "ok"},
            {"type": "tool_result", "id": "c1", "result": "ok", "success": False},
            id="result-missing-success",
        ),
        pytest.param(
            {"event": "system.finished.standard"},
            {"type": "turn_end", "reason": ""},
            id="turn-end-missing-reason",
        ),
    ],
)
def test_event_bridge_missing_payload_keys_use_documented_defaults(event, expected):
    """缺键事件 → 使用文档化默认值而非抛出 KeyError."""
    bridge = EventBridge()

    result = bridge.adapt(event, agent_id=6)

    assert result == {
        "agent_id": 6,
        **expected,
    }, f"缺键事件 {event} 应回退到文档化默认值"
    assert result["type"] == expected["type"], "缺键不应改变事件类型映射"


# ──────────────────────────────────────────────
# AgentRuntime → EventBridge 端到端映射
# ──────────────────────────────────────────────


def _delta_chunk(content=None, tool_calls=None, finish_reason=None):
    """构造一个 OpenAI 风格流式 chunk（choices 单元素，delta 为纯数据）."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


class _ToolRoundProvider(EchoProvider):
    """第一轮输出文本+一次工具调用，第二轮输出文本并 stop."""

    def __init__(self):
        super().__init__()
        self.chat_count = 0

    async def chat(self, tools=None, messages=None, **kwargs):
        self.chat_count += 1
        if self.chat_count == 1:
            chunks = [
                _delta_chunk(content="先看"),
                _delta_chunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="probe", arguments=None),
                        )
                    ]
                ),
                _delta_chunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=None,
                            function=SimpleNamespace(name=None, arguments='{"x": "p"}'),
                        )
                    ]
                ),
                _delta_chunk(finish_reason="tool_calls"),
            ]
        else:
            chunks = [
                _delta_chunk(content="完成"),
                _delta_chunk(finish_reason="stop"),
            ]

        async def _stream():
            for chunk in chunks:
                yield chunk

        return _stream()


@pytest.mark.asyncio
async def test_runtime_events_map_to_tui_types_without_dropping_any():
    """真实 run() 事件流经 EventBridge → 只有 finished.tool_calls 被丢弃."""
    manager = ToolManager()

    async def probe(x: str) -> str:
        return f"probe:{x}"

    manager.register(
        probe,
        name="probe",
        human_doc="probe doc",
        display_action="probe.run",
        display_param="x",
    )
    provider = _ToolRoundProvider()
    session = AgentSession(provider=provider, tool_manager=manager, agent_id=3)
    session.append_user_prompt("go")
    bridge = EventBridge(manager)

    raw_events = [event async for event in AgentRuntime.run(session)]
    mapped = [bridge.adapt(event, agent_id=3) for event in raw_events]
    dropped = [event for event, result in zip(raw_events, mapped) if result is None]
    forwarded = [result for result in mapped if result is not None]

    assert [event["event"] for event in raw_events] == [
        "agent.text",
        "agent.tool_call.header",
        "agent.tool_call.delta",
        "system.finished.tool_calls",
        "agent.tool_call.execution",
        "agent.tool_call.result",
        "agent.text",
        "system.finished.standard",
    ], "run() 的事件序列是本测试的前置契约"
    assert dropped == [
        {"event": "system.finished.tool_calls"}
    ], "内部信号 finished.tool_calls 是唯一应被 EventBridge 丢弃的事件"
    assert [result["type"] for result in forwarded] == [
        "token",
        "tool_call",
        "tool_call",
        "tool_start",
        "tool_result",
        "token",
        "turn_end",
    ], "每个运行时事件都应映射为 TUI 类型，顺序不变"
    assert all(
        result["agent_id"] == 3 for result in forwarded
    ), "所有映射结果都应携带产生事件的 agent_id"
    assert forwarded[0]["content"] == "先看", "首个文本增量应转发为 token 内容"
    started = forwarded[3]
    assert started["action"] == "probe.run", "tool_start 的 action 取自 ToolManager"
    assert (
        started["description"] == "probe doc"
    ), "tool_start 的 description 取自 human_doc"
    assert (
        started["display_param"] == "x"
    ), "tool_start 的 display_param 取自 ToolManager"
    assert started["args"] == {"x": "p"}, "tool_start 应携带聚合后的工具参数"
    finished = forwarded[4]
    assert finished["success"] is True, "工具成功的 result 事件应映射为 success=True"
    assert finished["result"] == "probe:p", "tool_result 应携带工具返回值"
    assert forwarded[6] == {
        "type": "turn_end",
        "agent_id": 3,
        "reason": "stop",
    }, "标准终止应把 reason 透传到 turn_end"
