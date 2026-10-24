"""The runtime must publish the calling agent's id while a tool executes."""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.agent.agent_runtime import AgentRuntime
from mitocode.core.agent.agent_session import AgentSession
from mitocode.core.chat_provider.sample_chat_provider import _SampleChatProvider
from mitocode.core.tool.builtin.context import current_agent_context


def _chunk(**kw):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=kw.get("content"), tool_calls=kw.get("tool_calls")),
        finish_reason=kw.get("finish_reason"),
    )])


def _tool_chunk(index=0, call_id="c1", name="peek", arguments=""):
    return SimpleNamespace(
        index=index, id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _Provider(_SampleChatProvider):
    def __init__(self, rounds):
        self.rounds = list(rounds)

    async def chat(self, tools=None, messages=None, **kw):
        chunks = self.rounds.pop(0)

        async def _stream():
            for c in chunks:
                yield c

        return _stream()

    async def get_model_list(self):
        return []

    async def init(self):
        pass

    @classmethod
    def get_settings_schema(cls):
        return {}

    @classmethod
    def check_settings_schema(cls, settings):
        return True


@pytest.mark.asyncio
async def test_tool_execution_sees_the_calling_agent_id():
    seen = {}

    async def peek() -> str:
        seen["agent_id"] = current_agent_context()[0]
        return "ok"

    provider = _Provider([
        [_chunk(tool_calls=[_tool_chunk(arguments="")], finish_reason="tool_calls")],
        [_chunk(content="done", finish_reason="stop")],
    ])
    session = AgentSession(
        agent_id=7,
        name="worker",
        provider=provider,
        tools_map={"peek": peek},
        tool_schemas=[{"type": "function", "function": {"name": "peek",
                         "description": "d", "parameters": {"type": "object", "properties": {}}}}],
    )
    async for _ in AgentRuntime.run(session):
        pass
    assert seen["agent_id"] == 7
