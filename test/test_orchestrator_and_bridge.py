import asyncio
import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.agent.agent_runtime import AgentRuntime
from mitocode.backend.agent.agent_session import AgentSession
from mitocode.backend.agent.agent_orchestrator import AgentOrchestrator
from mitocode.backend.agent.agent import AgentState
from mitocode.backend.bridge.event_adapter import EventBridge
from mitocode.backend.tool.tool_manager import ToolManager


class FakeChunk:
    def __init__(self, content=None, finish_reason=None):
        self.choices = [mock.MagicMock()]
        self.choices[0].delta.content = content
        self.choices[0].delta.tool_calls = None
        self.choices[0].finish_reason = finish_reason


class EchoProvider:
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

    agent = orchestrator._agents[agent_id]
    assert agent.session._cancel_event.is_set()


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
    assert result == {"type": "tool_call", "agent_id": 1, "content": '{"path":'}


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
        "result": "done",
        "success": True,
    }


def test_event_bridge_skips_internal():
    bridge = EventBridge()
    assert bridge.adapt({"event": "system.cancelled"}, agent_id=1) is None
    assert (
        bridge.adapt(
            {"event": "system.finished.standard", "reason": "stop"}, agent_id=1
        )
        is None
    )
    assert bridge.adapt({"event": "system.finished.tool_calls"}, agent_id=1) is None
