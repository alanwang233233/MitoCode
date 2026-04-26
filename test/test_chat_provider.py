import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.agent.agent import Agent
from backend.chat_provider._sample_chat_provider import _SampleChatProvider
from backend.chat_provider.chat_provider_openai import OpenAISubAgent


@pytest.mark.asyncio
async def test_mock_provider_works_with_single_agent_stream():
    provider = _SampleChatProvider(settings={"chunk_size": 8, "mock_reply_prefix": "[mock]"})
    agent = Agent(provider=provider)
    agent.append_user_prompt("hello")

    events = []
    async for event in agent.run():
        events.append(event)

    text = "".join(e["delta"] for e in events if e["event"] == "agent.text")
    assert text.startswith("[mock]")
    assert "hello" in text
    assert any(e["event"] == "system.finished.standard" for e in events)


def test_openai_subagent_settings_schema_validation():
    assert OpenAISubAgent.check_settings_schema({"api_key": "sk-test"}) is True
    assert OpenAISubAgent.check_settings_schema({}) is False
    assert OpenAISubAgent.check_settings_schema([]) is False


def test_openai_subagent_schema_declares_required_api_key():
    schema = OpenAISubAgent.get_settings_schema()
    assert "api_key" in schema["required"]

