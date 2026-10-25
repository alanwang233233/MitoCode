import asyncio
import os
import sys
import unittest.mock as mock

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
