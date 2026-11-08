import asyncio
import os
import sys
import unittest.mock as mock
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

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


class CancelAfterFinalChunkProvider(_SampleChatProvider):
    """交出最后一个 chunk 后才调用 ``session.cancel()`` 的 provider."""

    def __init__(self):
        self.session = None

    @staticmethod
    def get_model_list():
        return ["cancel-model"]

    @classmethod
    def get_settings_schema(cls):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @staticmethod
    def check_settings_schema(settings):
        return True

    def init(self):
        pass

    async def chat(self, tools=None, messages=None, **kwargs):
        session = self.session

        async def _stream():
            yield FakeChunk(content="partial ")
            yield FakeChunk(finish_reason="stop")
            session.cancel()

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
async def test_orchestrator_cancel_after_stream_reaches_bridge_as_turn_cancelled():
    """流结束后才落地的取消仍经 EventBridge 以 turn_cancelled 到达消费者."""
    provider = CancelAfterFinalChunkProvider()
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    agent_id = await orchestrator.create_agent(
        name="PostStreamCancel", provider=provider
    )
    provider.session = orchestrator.get_session(agent_id)

    await orchestrator.send_message(agent_id, "hello world")

    events = []
    while True:
        try:
            agent_id_evt, event = await asyncio.wait_for(
                orchestrator.event_queue.get(), timeout=2.0
            )
        except asyncio.TimeoutError:
            break
        events.append((agent_id_evt, event))
        if event.get("type") == "sync":
            agent_state = next(
                (a["state"] for a in event.get("agents", []) if a["id"] == agent_id),
                None,
            )
            if agent_state in ("finished", "cancelled", "error"):
                break

    forwarded = [e for aid, e in events if aid == agent_id]
    assert {"type": "turn_cancelled", "agent_id": agent_id} in forwarded, (
        "流后取消必须作为 turn_cancelled 事件到达 TUI 消费者，"
        f"否则终止事件永远缺失；实际收到 {forwarded}"
    )
    assert all(
        e["type"] != "turn_end" for e in forwarded
    ), "取消轮不应同时产出 turn_end"


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


def test_bridge_maps_agent_reasoning_to_reasoning_payload():
    bridge = EventBridge()
    result = bridge.adapt({"event": "agent.reasoning", "delta": "hmm"}, agent_id=1)
    assert result == {"type": "reasoning", "agent_id": 1, "content": "hmm"}


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
            pytest.param({"event": "agent.unknown", "id": "c1"}, id="unknown-type"),
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


# ──────────────────────────────────────────────
# AgentOrchestrator: 会话生命周期补充覆盖
# ──────────────────────────────────────────────


def _drain_events(orchestrator) -> list:
    """取出事件队列中当前的全部事件，便于做确定性计数断言."""
    events = []
    while not orchestrator.event_queue.empty():
        events.append(orchestrator.event_queue.get_nowait())
    return events


class _GateProvider(_SampleChatProvider):
    """在交出任何 chunk 之前先等待 gate 的 provider，用于确定性占住运行中的任务."""

    def __init__(self):
        self.stream = True
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    @staticmethod
    def get_model_list():
        return ["gate-model"]

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
            self.started.set()
            await self.gate.wait()
            yield FakeChunk(content="done")
            yield FakeChunk(finish_reason="stop")

        return _stream()


class _TurnProvider(EchoProvider):
    """沿用 Echo 行为但额外暴露 turn 计数器的 provider，用于验证按会话深拷贝."""

    def __init__(self):
        super().__init__()
        self.turn = 0


class _ToolCallTurnProvider(_SampleChatProvider):
    """首轮产出一次工具调用、次轮输出文本并 stop 的 provider."""

    def __init__(self):
        self.stream = True
        self.chat_count = 0

    @staticmethod
    def get_model_list():
        return ["tool-turn-model"]

    @classmethod
    def get_settings_schema(cls):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @staticmethod
    def check_settings_schema(settings):
        return True

    def init(self):
        pass

    async def chat(self, tools=None, messages=None, **kwargs):
        self.chat_count += 1
        if self.chat_count == 1:
            chunks = [
                _delta_chunk(content="pre"),
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
                            function=SimpleNamespace(name=None, arguments='{"x": 1}'),
                        )
                    ]
                ),
                _delta_chunk(finish_reason="tool_calls"),
            ]
        else:
            chunks = [
                _delta_chunk(content="post"),
                _delta_chunk(finish_reason="stop"),
            ]

        async def _stream():
            for chunk in chunks:
                yield chunk

        return _stream()


@pytest.mark.asyncio
async def test_create_session_unknown_parent_raises_value_error_with_parent_id():
    """parent_id 指向未注册会话 → ValueError 文案带上该 id 且不注册任何会话."""
    orchestrator = AgentOrchestrator()

    with pytest.raises(ValueError, match=r"Parent agent 99 does not exist"):
        await orchestrator.create_session(provider=EchoProvider(), parent_id=99)

    assert orchestrator.get_tree_snapshot() == [], "失败创建不应留下会话节点"
    assert orchestrator.get_session(99) is None, "未知 parent_id 不应被注册"


@pytest.mark.asyncio
async def test_create_session_deep_copies_turn_provider_and_resets_turn():
    """带 turn 属性的 provider → 会话拿到独立副本且副本 turn 归零，原对象不变."""
    provider = _TurnProvider()
    provider.turn = 5
    orchestrator = AgentOrchestrator()

    session = await orchestrator.create_session(name="scenario", provider=provider)

    assert session.provider is not provider, "带 turn 的 provider 必须按会话深拷贝隔离"
    assert session.provider.turn == 0, "会话副本的 turn 应从 0 重新开始"
    assert provider.turn == 5, "深拷贝不应修改调用方传入的 provider"

    second = await orchestrator.create_session(name="scenario-2", provider=provider)
    assert second.provider.turn == 0, "每个会话都应拿到自己的 turn 计数器"
    assert (
        session.provider is not second.provider
    ), "两个会话的 provider 副本必须互不相同"


@pytest.mark.asyncio
async def test_create_session_provider_without_turn_is_used_by_reference():
    """不带 turn 属性的 provider → 会话直接持有传入的同一实例，不做深拷贝."""
    provider = EchoProvider()
    orchestrator = AgentOrchestrator()

    session = await orchestrator.create_session(name="plain", provider=provider)

    assert session.provider is provider, "无 turn 的 provider 应按引用共享而非复制"


@pytest.mark.asyncio
async def test_destroy_agent_unknown_id_is_noop_and_queues_nothing():
    """销毁未知 id → 既有会话保留且不推送树同步事件."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="keep")
    _drain_events(orchestrator)

    await orchestrator.destroy_agent(999)

    assert (
        orchestrator.get_session(session.agent_id) is session
    ), "未知 id 不应动到其它会话"
    assert orchestrator.event_queue.empty(), "未知 id 的销毁不应推送树同步事件"


@pytest.mark.asyncio
async def test_destroy_agent_child_detaches_parent_and_pushes_single_sync():
    """销毁子会话 → 从父会话子列表摘除、自身槽位清空且只推一次树同步."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    parent = await orchestrator.create_session(name="p")
    child = await orchestrator.create_session(name="c", parent_id=parent.agent_id)
    _drain_events(orchestrator)

    await orchestrator.destroy_agent(child.agent_id)

    assert orchestrator.get_session(child.agent_id) is None, "子会话应从注册表移除"
    assert orchestrator.get_session(parent.agent_id) is parent, "父会话应保留"
    assert parent.sub_agent_ids == [], "父会话的子代理列表应摘除被销毁的子 id"
    assert orchestrator.event_queue.qsize() == 1, "顶层销毁应恰好推送一次树同步"
    agent_evt, payload = orchestrator.event_queue.get_nowait()
    assert agent_evt == -1, "树同步事件应使用 -1 作为 agent 标识"
    assert payload["type"] == "sync", "销毁后推送的应是 sync 事件"
    assert [node["id"] for node in payload["agents"]] == [
        parent.agent_id
    ], "同步快照应只剩父会话"


@pytest.mark.asyncio
async def test_destroy_agent_parent_cascades_to_grandchildren():
    """销毁父会话 → 递归销毁子与孙，整棵树清空且顶层只推送一次同步."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    root = await orchestrator.create_session(name="r")
    child = await orchestrator.create_session(name="c", parent_id=root.agent_id)
    grandchild = await orchestrator.create_session(name="g", parent_id=child.agent_id)
    _drain_events(orchestrator)

    await orchestrator.destroy_agent(root.agent_id)

    assert orchestrator.get_session(root.agent_id) is None, "根会话应被销毁"
    assert orchestrator.get_session(child.agent_id) is None, "子会话应被级联销毁"
    assert orchestrator.get_session(grandchild.agent_id) is None, "孙会话应被级联销毁"
    assert orchestrator.get_tree_snapshot() == [], "级联销毁后树应为空"
    assert orchestrator.event_queue.qsize() == 1, "递归销毁不应额外推送同步事件"


@pytest.mark.asyncio
async def test_destroy_agent_sync_false_removes_without_push():
    """顶层调用传 _sync=False → 仍然完成移除但完全不推送同步事件."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="silent")
    _drain_events(orchestrator)

    await orchestrator.destroy_agent(session.agent_id, _sync=False)

    assert (
        orchestrator.get_session(session.agent_id) is None
    ), "_sync=False 仍应移除会话"
    assert orchestrator.event_queue.empty(), "_sync=False 不应推送树同步事件"


@pytest.mark.asyncio
async def test_destroy_agent_child_already_detached_from_parent_is_removed():
    """父会话子列表已缺失该 id → 销毁不得抛错，仍要清空会话本身."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    parent = await orchestrator.create_session(name="p")
    child = await orchestrator.create_session(name="c", parent_id=parent.agent_id)
    parent.sub_agent_ids.remove(child.agent_id)
    _drain_events(orchestrator)

    await orchestrator.destroy_agent(child.agent_id)

    assert orchestrator.get_session(child.agent_id) is None, "已脱钩的子会话仍应被销毁"
    assert parent.sub_agent_ids == [], "父会话子列表应保持为空且不重复摘除"


@pytest.mark.asyncio
async def test_destroy_agent_while_running_cancels_child_task():
    """销毁运行中的子会话 → 任务被真正打断，而不只是从注册表里消失."""
    provider = _TextThenGateProvider()
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    parent = await orchestrator.create_session(name="p")
    child = await orchestrator.create_session(
        name="c", parent_id=parent.agent_id, provider=provider
    )
    _drain_events(orchestrator)

    task = await orchestrator.send_message(child.agent_id, "hi")
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)
    before = _drain_events(orchestrator)
    assert any(
        aid == child.agent_id and payload["type"] == "token" for aid, payload in before
    ), "前置：子会话应已产出 token 并停在 gate 上"
    assert child.state is AgentState.RUNNING, "前置：子会话任务应仍在运行"

    await orchestrator.destroy_agent(child.agent_id)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert (
        task.cancelled() is True
    ), "销毁必须取消运行中的任务，而非把它留在后台继续推送"
    assert child.state is AgentState.CANCELLED, "被中断的会话状态应落到 CANCELLED"
    assert (
        orchestrator.get_session(child.agent_id) is None
    ), "销毁后注册表不应残留子会话"
    assert parent.sub_agent_ids == [], "销毁后父会话子列表应摘除该 id"
    assert [node["id"] for node in orchestrator.get_tree_snapshot()] == [
        parent.agent_id
    ], "树快照应只剩父会话"
    assert (
        await orchestrator.send_message(child.agent_id, "again") is None
    ), "销毁后再向该 id 发消息应返回 None"

    after = _drain_events(orchestrator)
    assert len(after) == 3, (
        "销毁推送一次树同步，被取消的任务补发恰好一条中止事件、"
        "其 finally 再推送一次树同步"
    )
    assert (
        after[0][0] == -1 and after[0][1]["type"] == "sync"
    ), "第一条应是销毁自身推送的树同步"
    assert after[1] == (
        child.agent_id,
        {"type": "turn_cancelled", "agent_id": child.agent_id},
    ), "被硬取消的任务必须补发恰好一条 turn_cancelled，否则 TUI 永远不知道轮次被打断"
    assert (
        after[2][0] == -1 and after[2][1]["type"] == "sync"
    ), "最后一条应是任务 finally 推送的树同步"
    assert [payload["agents"] for aid, payload in after if aid == -1] == [
        [
            {
                "id": parent.agent_id,
                "name": "p",
                "description": "",
                "parent_id": None,
                "sub_agent": [],
                "state": "idle",
            }
        ],
        [
            {
                "id": parent.agent_id,
                "name": "p",
                "description": "",
                "parent_id": None,
                "sub_agent": [],
                "state": "idle",
            }
        ],
    ], "两次同步快照都应只含父会话且父会话保持 idle"


@pytest.mark.asyncio
async def test_cancel_agent_unknown_id_is_noop():
    """取消未知 id → 既有会话保持未取消且不产生事件."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="safe")
    _drain_events(orchestrator)

    await orchestrator.cancel_agent(999)

    assert session.is_cancelled() is False, "取消未知 id 不应波及其它会话"
    assert orchestrator.event_queue.empty(), "取消未知 id 不应产生事件"


@pytest.mark.asyncio
async def test_cancel_agent_without_task_marks_session_cancelled():
    """会话没有关联任务时取消 → 只置位取消标记，不抛错也不产生事件."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="idle")
    _drain_events(orchestrator)
    assert orchestrator._tasks[session.agent_id] is None, "前置：新会话的任务槽应为空"

    await orchestrator.cancel_agent(session.agent_id)

    assert session.is_cancelled() is True, "无任务时取消仍应置位会话取消标记"
    assert orchestrator.event_queue.empty(), "取消不应额外推送树同步事件"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cascade, child_should_be_cancelled",
    [(False, False), (True, True)],
    ids=["cascade-false-只取消父", "cascade-true-级联取消子"],
)
async def test_cancel_agent_cascade_flag_controls_child_cancellation(
    cascade, child_should_be_cancelled
):
    """cascade 标志决定子会话是否被递归取消，父会话总是被取消."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    parent = await orchestrator.create_session(name="p")
    child = await orchestrator.create_session(name="c", parent_id=parent.agent_id)

    await orchestrator.cancel_agent(parent.agent_id, cascade=cascade)

    assert parent.is_cancelled() is True, "父会话无论如何都应被取消"
    assert (
        child.is_cancelled() is child_should_be_cancelled
    ), f"cascade={cascade} 时子会话取消状态应为 {child_should_be_cancelled}"


@pytest.mark.asyncio
async def test_send_message_unknown_agent_returns_none_and_appends_nothing():
    """向未知会话发消息 → 返回 None 且不改动任何既有会话的消息历史."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="a")
    _drain_events(orchestrator)

    result = await orchestrator.send_message(999, "hello")

    assert result is None, "未知 agent_id 应返回 None"
    assert session.messages == [], "未知 agent_id 不应把消息写进别的会话"
    assert orchestrator.event_queue.empty(), "未知 agent_id 不应推送树同步事件"


@pytest.mark.asyncio
async def test_send_message_while_running_replaces_previous_task():
    """RUNNING 状态下再次发消息 → 旧任务被取消并等待，随后创建新任务."""
    provider = _GateProvider()
    orchestrator = AgentOrchestrator(default_provider=provider)
    session = await orchestrator.create_session(name="gated")

    first = await orchestrator.send_message(session.agent_id, "one")
    assert isinstance(first, asyncio.Task), "首次发送应返回 asyncio.Task"
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)
    assert session.state is AgentState.RUNNING, "前置：首个任务应仍在运行"

    second = await orchestrator.send_message(session.agent_id, "two")

    assert isinstance(second, asyncio.Task), "再次发送应返回新的 asyncio.Task"
    assert second is not first, "再次发送必须换成新任务而非复用旧任务"
    assert first.cancelled() is True, "旧任务应已被取消"
    assert first.done() is True, "send_message 应在返回前等待旧任务结束"

    provider.gate.set()
    await asyncio.wait_for(second, timeout=2.0)

    assert session.state is AgentState.FINISHED, "新任务正常结束应落到 FINISHED"
    assert session.build_standard_context() == [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "done"},
    ], "两条用户消息应保留且助手输出只来自新任务"


class _AlwaysFullQueue(asyncio.Queue):
    """put 立即抛出 QueueFull 的队列替身。

    ``asyncio.Queue.put`` 在满时是等待空位而不是抛错，只有 ``put_nowait`` 抛
    ``QueueFull``；这里用队列子类把"已满"这一边界变成确定性可测的输入。
    """

    async def put(self, item):
        """模拟满队列，立即抛出 QueueFull."""
        raise asyncio.QueueFull


class _TextThenGateProvider(_SampleChatProvider):
    """先交出首个 token、随后阻塞在 gate 的 provider，用于验证销毁会打断运行中的任务."""

    def __init__(self):
        self.stream = True
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    @staticmethod
    def get_model_list():
        return ["gated-text-model"]

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
            yield FakeChunk(content="working")
            self.started.set()
            await self.gate.wait()
            yield FakeChunk(content="never", finish_reason="stop")

        return _stream()


@pytest.mark.asyncio
async def test_push_tree_sync_swallows_queue_full():
    """事件队列已满 → _push_tree_sync 吞掉 QueueFull 且不写入任何事件."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    await orchestrator.create_session(name="root")
    orchestrator.event_queue = _AlwaysFullQueue()

    await orchestrator._push_tree_sync()

    assert orchestrator.event_queue.empty(), "满队列不应收到同步事件"
    assert [node["id"] for node in orchestrator.get_tree_snapshot()] == [
        0
    ], "QueueFull 只影响推送，不应改动树结构"


@pytest.mark.asyncio
async def test_run_session_unknown_id_returns_early_without_events():
    """运行未知会话 → 立即返回，不产生事件也不登记任务."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    _drain_events(orchestrator)

    await orchestrator._run_session(999)

    assert orchestrator.get_session(999) is None, "未知会话不应因运行而被创建"
    assert orchestrator.event_queue.empty(), "未知会话的早退不应推送任何事件"


@pytest.mark.asyncio
async def test_run_session_skips_adapted_none_events():
    """运行时事件适配结果为 None → 不进入队列，只有可适配事件到达消费者."""
    provider = _ToolCallTurnProvider()
    orchestrator = AgentOrchestrator(default_provider=provider)
    session = await orchestrator.create_session(name="round")
    _drain_events(orchestrator)

    task = await orchestrator.send_message(session.agent_id, "go")
    await asyncio.wait_for(task, timeout=2.0)

    events = _drain_events(orchestrator)
    agent_events = [payload for aid, payload in events if aid == session.agent_id]

    assert session.state is AgentState.FINISHED, "两轮结束后状态应为 FINISHED"
    assert len(agent_events) == 7, "8 个运行时事件中恰有 1 个适配为 None 被丢弃"
    assert all(
        isinstance(payload, dict) and "type" in payload for payload in agent_events
    ), "队列中不应出现 None 或缺失 type 的载荷"
    assert [payload["type"] for payload in agent_events] == [
        "token",
        "tool_call",
        "tool_call",
        "tool_start",
        "tool_result",
        "token",
        "turn_end",
    ], "被丢弃的应是 finished.tool_calls，其余事件顺序不变"


@pytest.mark.asyncio
async def test_run_session_cancelled_sets_state_and_propagates():
    """运行中被取消 → CancelledError 传到任务、状态落到 CANCELLED 且任务槽清空."""
    provider = _GateProvider()
    orchestrator = AgentOrchestrator(default_provider=provider)
    session = await orchestrator.create_session(name="cancel-me")

    task = await orchestrator.send_message(session.agent_id, "hi")
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)

    await orchestrator.cancel_agent(session.agent_id)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert session.state is AgentState.CANCELLED, "任务中止应把会话状态置为 CANCELLED"
    assert session.is_cancelled() is True, "取消标记应保持置位"
    assert orchestrator._tasks[session.agent_id] is None, "结束的任务应从任务槽清空"
    assert (
        orchestrator.get_tree_snapshot()[0]["state"] == "cancelled"
    ), "树快照应把取消状态展示给 TUI"


@pytest.mark.asyncio
async def test_run_session_finally_keeps_foreign_task_slot():
    """任务槽属于别的任务时 → finally 不得清空它，但会话仍正常跑完并推送同步."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="guarded")
    _drain_events(orchestrator)

    foreign = asyncio.create_task(asyncio.sleep(0))
    await foreign
    orchestrator._tasks[session.agent_id] = foreign

    await orchestrator._run_session(session.agent_id)

    assert session.state is AgentState.FINISHED, "直接运行仍应把会话跑到 FINISHED"
    assert (
        orchestrator._tasks[session.agent_id] is foreign
    ), "任务槽属于别的任务时不应被本任务清空"

    events = _drain_events(orchestrator)
    assert len(events) == 2, "本次运行应只产出 turn_end 与其后 finally 推送的树同步"
    agent_evt, payload = events[0]
    assert (
        agent_evt == session.agent_id and payload["type"] == "turn_end"
    ), "第一条应是运行时事件经 EventBridge 适配后的 turn_end"
    sync_evt, sync_payload = events[-1]
    assert sync_evt == -1, "最后一条必须是 finally 推送的树同步（-1 标识）"
    assert sync_payload == {
        "type": "sync",
        "agents": [
            {
                "id": session.agent_id,
                "name": "guarded",
                "description": "",
                "parent_id": None,
                "sub_agent": [],
                "state": "finished",
            }
        ],
    }, "finally 推送的同步快照应反映本会话已 finished"


@pytest.mark.asyncio
async def test_get_tree_snapshot_exposes_the_six_tui_keys():
    """树快照节点恰好暴露 TUI 消费的六个键，新会话状态为 idle."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    await orchestrator.create_session(name="root", description="describe")

    node = orchestrator.get_tree_snapshot()[0]

    assert set(node.keys()) == {
        "id",
        "name",
        "description",
        "parent_id",
        "sub_agent",
        "state",
    }, "快照节点键集合必须与 TUI 契约一致"
    assert node["state"] == "idle", "从未运行过的会话状态应为 idle"
    assert node["sub_agent"] == [], "新会话的子代理列表应为空"


# ──────────────────────────────────────────────
# AgentOrchestrator: 硬取消（task.cancel）时的中止事件补发
# ──────────────────────────────────────────────


def _turn_cancelled_events(events, agent_id):
    """从队列事件中筛出某个 agent 的 turn_cancelled 载荷列表."""
    return [
        payload
        for aid, payload in events
        if aid == agent_id and payload["type"] == "turn_cancelled"
    ]


@pytest.mark.asyncio
async def test_cancel_during_streaming_emits_exactly_one_turn_cancelled():
    """流式输出中被硬取消 → 恰好一条 turn_cancelled，状态 CANCELLED，异常照常上抛."""
    provider = _TextThenGateProvider()
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="stream-cancel", provider=provider)
    _drain_events(orchestrator)

    task = await orchestrator.send_message(session.agent_id, "hi")
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)
    _drain_events(orchestrator)

    # cancel_agent 与 Repl.handle_sigint 同构：置位取消标记并立即 cancel 任务。
    await orchestrator.cancel_agent(session.agent_id)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    events = _drain_events(orchestrator)
    assert (
        len(_turn_cancelled_events(events, session.agent_id)) == 1
    ), f"硬取消必须补发恰好一条 turn_cancelled，实际事件 {events}"
    assert session.state is AgentState.CANCELLED, "硬取消后状态应落到 CANCELLED"
    assert all(
        payload["type"] != "turn_end"
        for aid, payload in events
        if aid == session.agent_id
    ), "被取消的轮次不应同时产出 turn_end"


@pytest.mark.asyncio
async def test_cancel_during_tool_execution_emits_exactly_one_turn_cancelled():
    """工具执行中被硬取消 → 恰好一条 turn_cancelled 且会话历史保持合法."""
    tool_started = asyncio.Event()

    async def probe(x: int) -> str:
        tool_started.set()
        await asyncio.sleep(30)
        return f"probe:{x}"

    manager = ToolManager()
    manager.register(probe, name="probe", human_doc="probe doc")
    provider = _ToolCallTurnProvider()
    orchestrator = AgentOrchestrator(
        tool_manager=manager, default_provider=EchoProvider()
    )
    session = await orchestrator.create_session(name="tool-cancel", provider=provider)
    _drain_events(orchestrator)

    task = await orchestrator.send_message(session.agent_id, "go")
    await asyncio.wait_for(tool_started.wait(), timeout=2.0)

    await orchestrator.cancel_agent(session.agent_id)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    events = _drain_events(orchestrator)
    assert (
        len(_turn_cancelled_events(events, session.agent_id)) == 1
    ), f"工具执行中的硬取消必须补发恰好一条 turn_cancelled，实际事件 {events}"
    assert session.state is AgentState.CANCELLED, "硬取消后状态应落到 CANCELLED"

    context = session.build_standard_context()
    call_ids = {
        call["id"]
        for message in context
        if message.get("role") == "assistant" and message.get("tool_calls")
        for call in message["tool_calls"]
    }
    result_ids = {
        message["tool_call_id"] for message in context if message.get("role") == "tool"
    }
    assert (
        call_ids == result_ids == {"call_1"}
    ), "被打断的工具调用应补上合成结果，保持 assistant/tool 消息成对"


@pytest.mark.asyncio
async def test_cooperative_cancel_emits_exactly_one_turn_cancelled_without_raising():
    """只置位会话取消标记（不 cancel 任务）→ 恰好一条 turn_cancelled 且任务正常结束."""
    provider = _TextThenGateProvider()
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="coop-cancel", provider=provider)
    _drain_events(orchestrator)

    task = await orchestrator.send_message(session.agent_id, "hi")
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)
    _drain_events(orchestrator)

    session.cancel()
    provider.gate.set()

    await asyncio.wait_for(task, timeout=2.0)

    events = _drain_events(orchestrator)
    assert (
        len(_turn_cancelled_events(events, session.agent_id)) == 1
    ), f"协作式取消必须恰好产出一条 turn_cancelled，实际事件 {events}"
    assert session.state is AgentState.CANCELLED, "协作式取消后状态应落到 CANCELLED"
    assert task.cancelled() is False, "协作式取消由运行时自查结束，任务不应是 cancelled"


# ──────────────────────────────────────────────
# AgentOrchestrator: 配置 secret 的字面抹除
# ──────────────────────────────────────────────


class _SecretBoomProvider(EchoProvider):
    """Provider that raises with a configured non-``sk-`` secret in the message."""

    async def chat(self, tools=None, messages=None, **kwargs):
        raise RuntimeError("request failed token=super-secret-42")


@pytest.mark.asyncio
async def test_run_session_redacts_configured_secret_from_event_and_log():
    """配置的普通 API key 在错误事件消息与日志 traceback 中都被字面抹除."""
    records: list = []
    orchestrator = AgentOrchestrator(
        default_provider=_SecretBoomProvider(),
        redact_secrets=("super-secret-42",),
    )
    agent_id = await orchestrator.create_agent(name="secret-boom")

    handler_id = loguru_logger.add(records.append, level="ERROR")
    try:
        task = await orchestrator.send_message(agent_id, "hi")
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        loguru_logger.remove(handler_id)

    events = _drain_events(orchestrator)
    errors = [payload for _, payload in events if payload.get("type") == "error"]
    assert errors, "provider 失败必须产生一条 error 事件"
    assert (
        "super-secret-42" not in errors[0]["message"]
    ), "配置的 key 不得进入 error 事件消息"
    assert (
        errors[0]["message"] == "request failed token=***"
    ), "字面 secret 应被替换为 *** 且其余文本保持可诊断"
    assert orchestrator.get_session(agent_id).state is AgentState.ERROR

    log_text = "".join(str(record) for record in records)
    assert "run failed" in log_text, "错误日志应被捕获（含 traceback）"
    assert "super-secret-42" not in log_text, "配置的 key 不得进入日志 traceback"
    assert "request failed token=***" in log_text, "日志应保留抹除后的可诊断文本"


class _BlankSecretBoomProvider(EchoProvider):
    """Provider that raises a plain message, used against blank secrets."""

    async def chat(self, tools=None, messages=None, **kwargs):
        raise RuntimeError("plain boom")


@pytest.mark.asyncio
async def test_blank_redact_secrets_are_ignored():
    """空白 secret 被忽略 → 不会把错误消息替换成星号碎片."""
    orchestrator = AgentOrchestrator(
        default_provider=_BlankSecretBoomProvider(),
        redact_secrets=("", "   "),
    )
    agent_id = await orchestrator.create_agent(name="blank-secret")
    task = await orchestrator.send_message(agent_id, "hi")
    await asyncio.wait_for(task, timeout=2.0)

    events = _drain_events(orchestrator)
    errors = [payload for _, payload in events if payload.get("type") == "error"]
    assert (
        errors and errors[0]["message"] == "plain boom"
    ), "空串/空白 secret 必须被忽略，否则 replace('') 会撕碎整条消息"


# ──────────────────────────────────────────────
# AgentOrchestrator: 会话切换钩子（忙碌检测 / reset / restore）
# ──────────────────────────────────────────────


def test_has_active_tasks_on_fresh_orchestrator_is_false():
    """新建 orchestrator 没有任何任务 → has_active_tasks() 为 False."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())

    assert orchestrator.has_active_tasks() is False, "空注册表不应报告有活动任务"


@pytest.mark.asyncio
async def test_has_active_tasks_tracks_running_and_finished_task():
    """运行中的任务 → True；任务结束后（槽位已被 finally 清空）→ False."""
    provider = _GateProvider()
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="busy", provider=provider)
    assert (
        orchestrator.has_active_tasks() is False
    ), "仅有空任务槽（None）的会话不应算作忙碌"

    task = await orchestrator.send_message(session.agent_id, "hi")
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)
    assert orchestrator.has_active_tasks() is True, "运行中的任务必须被报告为忙碌"

    provider.gate.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert orchestrator.has_active_tasks() is False, "任务结束后不应再报告为忙碌"


@pytest.mark.asyncio
async def test_reset_sessions_cancels_tasks_clears_registry_and_pushes_sync():
    """reset_sessions → 任务被取消并等待、注册表清空、计数器归零、推一次空树 sync."""
    provider = _GateProvider()
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    first = await orchestrator.create_session(name="one", provider=provider)
    second = await orchestrator.create_session(name="two")
    task = await orchestrator.send_message(first.agent_id, "hi")
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)
    _drain_events(orchestrator)

    await orchestrator.reset_sessions()

    assert task.cancelled() is True, "reset 必须硬取消运行中的任务并等待其结束"
    assert first.state is AgentState.CANCELLED, "被取消任务的会话状态应为 CANCELLED"
    assert orchestrator._sessions == {}, "整个注册表应被清空"
    assert orchestrator._tasks == {}, "任务表应被清空"
    assert orchestrator._counter == 0, "计数器应归零，供下次 create/restore 重头分配"
    assert orchestrator.has_active_tasks() is False, "reset 后不应再有活动任务"

    events = _drain_events(orchestrator)
    assert events[-1] == (-1, {"type": "sync", "agents": []}), (
        "reset 应以一次空树 sync 收尾，实际事件 " f"{events!r}"
    )


@pytest.mark.asyncio
async def test_reset_sessions_on_empty_registry_is_safe_and_zeroes_counter():
    """空注册表（销毁后会话已不在）→ reset 不抛异常、计数器归零且推一次 sync."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    session = await orchestrator.create_session(name="gone")
    await orchestrator.destroy_agent(session.agent_id)
    _drain_events(orchestrator)
    assert orchestrator._counter == 1, "前置：destroy 不重置计数器，reset 应负责归零"

    await orchestrator.reset_sessions()

    assert orchestrator._counter == 0, "空注册表下 reset 也应把计数器归零"
    assert orchestrator._sessions == {}, "注册表应保持为空"
    assert _drain_events(orchestrator) == [
        (-1, {"type": "sync", "agents": []})
    ], "空注册表的 reset 应恰好推送一次空树 sync"


@pytest.mark.asyncio
async def test_restore_tree_rebuilds_sessions_with_original_ids_and_tree():
    """合法记录 → 原 id 保留、树关系重建、counter 就位且只推一次 sync."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    agents = [
        {"agent_id": 0, "parent_id": None, "name": "Main", "description": "root"},
        {"agent_id": 1, "parent_id": 0, "name": "Child", "description": ""},
    ]
    _drain_events(orchestrator)

    sessions = await orchestrator.restore_tree(agents)

    assert set(sessions) == {0, 1}, "返回 dict 应以原 agent_id 为键"
    assert orchestrator.get_session(0) is sessions[0], "原 id 0 应重新注册"
    assert orchestrator.get_session(1) is sessions[1], "原 id 1 应重新注册"
    assert sessions[0].name == "Main"
    assert sessions[0].description == "root"
    assert sessions[1].parent_id == 0, "子会话的 parent_id 应来自记录"
    assert sessions[0].sub_agent_ids == [1], "父会话的子列表应按记录重建"
    assert sessions[1].sub_agent_ids == [], "子会话不应有子代理"
    assert orchestrator._counter == 2, "计数器应停在最大恢复 id 之上"
    assert orchestrator._tasks == {0: None, 1: None}, "每个恢复会话应有空任务槽"
    assert sessions[0].messages == [], "应以 system_prompt='' 构造，不追加 system 块"

    events = _drain_events(orchestrator)
    assert len(events) == 1, "restore_tree 应恰好推送一次树同步"
    evt_id, payload = events[0]
    assert evt_id == -1 and payload["type"] == "sync", "应推送 -1 标识的 sync 事件"
    assert [node["id"] for node in payload["agents"]] == [
        0,
        1,
    ], "同步快照应按记录顺序包含全部恢复会话"


@pytest.mark.asyncio
async def test_restore_tree_accepts_child_records_before_their_parent():
    """记录先列子后父 → 两遍构造应正确重建树，不依赖记录顺序."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())
    agents = [
        {"agent_id": 1, "parent_id": 0, "name": "Child"},
        {"agent_id": 0, "parent_id": None, "name": "Main"},
    ]

    sessions = await orchestrator.restore_tree(agents)

    assert sessions[0].sub_agent_ids == [1], "父会话子列表应重建而非依赖记录顺序"
    assert sessions[1].parent_id == 0, "子会话 parent_id 应保留"


@pytest.mark.asyncio
async def test_restore_tree_duplicate_ids_raise_value_error():
    """重复 agent_id → ValueError，且注册表不被污染."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())

    with pytest.raises(ValueError):
        await orchestrator.restore_tree(
            [
                {"agent_id": 0, "parent_id": None},
                {"agent_id": 0, "parent_id": None},
            ]
        )

    assert orchestrator._sessions == {}, "校验失败不应留下任何会话"


@pytest.mark.asyncio
async def test_restore_tree_unknown_parent_raises_value_error():
    """parent_id 指向记录中不存在的 id → ValueError."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())

    with pytest.raises(ValueError):
        await orchestrator.restore_tree([{"agent_id": 0, "parent_id": 9}])

    assert orchestrator._sessions == {}, "校验失败不应留下任何会话"


@pytest.mark.asyncio
async def test_restore_tree_parent_cycle_raises_value_error():
    """parent 链成环（0→1→0）→ ValueError."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())

    with pytest.raises(ValueError):
        await orchestrator.restore_tree(
            [
                {"agent_id": 0, "parent_id": 1},
                {"agent_id": 1, "parent_id": 0},
            ]
        )

    assert orchestrator._sessions == {}, "校验失败不应留下任何会话"


@pytest.mark.asyncio
async def test_restore_tree_empty_agents_raises_value_error():
    """空 agents 列表 → ValueError."""
    orchestrator = AgentOrchestrator(default_provider=EchoProvider())

    with pytest.raises(ValueError):
        await orchestrator.restore_tree([])

    assert orchestrator._sessions == {}, "校验失败不应留下任何会话"


@pytest.mark.asyncio
async def test_restore_tree_without_default_provider_raises_value_error():
    """orchestrator 无 default_provider → ValueError."""
    orchestrator = AgentOrchestrator()

    with pytest.raises(ValueError):
        await orchestrator.restore_tree([{"agent_id": 0, "parent_id": None}])


@pytest.mark.asyncio
async def test_restore_tree_deep_copies_turn_provider_per_session():
    """带 turn 的默认 provider → 每个会话拿到独立副本且 turn 归零，原对象不变."""
    provider = _TurnProvider()
    provider.turn = 5
    orchestrator = AgentOrchestrator(default_provider=provider)

    sessions = await orchestrator.restore_tree(
        [
            {"agent_id": 0, "parent_id": None},
            {"agent_id": 1, "parent_id": 0},
        ]
    )

    assert sessions[0].provider is not provider, "带 turn 的 provider 必须深拷贝隔离"
    assert sessions[0].provider.turn == 0, "会话副本的 turn 应从 0 重新开始"
    assert sessions[1].provider is not sessions[0].provider, "各会话副本互不相同"
    assert sessions[1].provider.turn == 0, "每个副本的 turn 都应归零"
    assert provider.turn == 5, "深拷贝不应修改调用方的默认 provider"
