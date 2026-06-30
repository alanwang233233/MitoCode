import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.agent.agent import Agent
from mitocode.backend.agent.agent_session import (
    _PlainTextMessageBlock,
    _ToolResultMessageBlock,
    _ToolCallMessageBlock,
    _MessageBlock,
)

# ---------------------------------------------------------------------------
# _MessageBlock tests
# ---------------------------------------------------------------------------


class TestMessageBlock:
    def test_init_sets_finished_false(self):
        block = _MessageBlock()
        assert block.finished is False


# ---------------------------------------------------------------------------
# _PlainTextMessageBlock tests
# ---------------------------------------------------------------------------


class TestPlainTextMessageBlock:
    def test_init_with_message(self):
        block = _PlainTextMessageBlock("user", "hello")
        assert block.role == "user"
        assert block.message == "hello"
        assert block.finished is True

    def test_init_without_message(self):
        block = _PlainTextMessageBlock("assistant")
        assert block.role == "assistant"
        assert block.message == ""
        assert block.finished is True

    def test_append_adds_to_message(self):
        block = _PlainTextMessageBlock("user", "hello")
        block.append(" world")
        assert block.message == "hello world"

    def test_append_multiple_times(self):
        block = _PlainTextMessageBlock("assistant", "")
        block.append("a")
        block.append("b")
        block.append("c")
        assert block.message == "abc"


# ---------------------------------------------------------------------------
# _ToolCallMessageBlock tests
# ---------------------------------------------------------------------------


class TestToolCallMessageBlock:
    def test_init(self):
        block = _ToolCallMessageBlock("my_tool", "call_123")
        assert block.name == "my_tool"
        assert block.id == "call_123"
        assert block.arguments_string == ""
        assert block.parameter == {}
        assert block.parse_error is None
        assert block.finished is False

    def test_append_argument_chunk(self):
        block = _ToolCallMessageBlock("tool", "id1")
        block.append_argument_chunk('{"key":')
        block.append_argument_chunk('"value"}')
        assert block.arguments_string == '{"key":"value"}'

    def test_finish_with_valid_json(self):
        block = _ToolCallMessageBlock("tool", "id1")
        block.append_argument_chunk('{"name": "test"}')
        block.finish()
        assert block.finished is True
        assert block.parameter == {"name": "test"}
        assert block.parse_error is None

    def test_finish_with_invalid_json(self):
        block = _ToolCallMessageBlock("tool", "id1")
        block.append_argument_chunk("{invalid json}")
        block.finish()
        assert block.finished is True
        assert block.parse_error is not None
        assert "JSON parsing failed" in block.parse_error
        assert block.parameter == {}

    def test_finish_with_empty_arguments(self):
        block = _ToolCallMessageBlock("tool", "id1")
        block.finish()
        assert block.finished is True
        assert block.parameter == {}

    def test_to_standard_when_not_finished(self):
        block = _ToolCallMessageBlock("tool", "id1")
        result = block.to_standard()
        assert result == {}

    def test_to_standard_when_finished(self):
        block = _ToolCallMessageBlock("my_tool", "call_abc")
        block.append_argument_chunk('{"arg": 1}')
        block.finish()
        result = block.to_standard()
        assert result == {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "my_tool",
                "arguments": '{"arg": 1}',
            },
        }


# ---------------------------------------------------------------------------
# _ToolResultMessageBlock tests
# ---------------------------------------------------------------------------


class TestToolResultMessageBlock:
    def test_init(self):
        block = _ToolResultMessageBlock("call_123", "result data")
        assert block.role == "tool"
        assert block.tool_call_id == "call_123"
        assert block.content == "result data"
        assert block.finished is True


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestSingleAgentInit:
    def test_init_with_provider_only(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        assert agent.provider is provider
        assert agent.messages == []
        assert agent.tools_map == {}
        assert agent.tool_schemas == []
        assert agent.on_end is None

    def test_init_with_all_settings(self):
        provider = mock.MagicMock()
        tools_map = {"tool1": lambda x: x}
        tool_schemas = [{"name": "tool1"}]
        on_end = mock.MagicMock()

        agent = Agent(
            provider=provider,
            tools_map=tools_map,
            tool_schemas=tool_schemas,
            on_end=on_end,
        )
        assert agent.provider is provider
        assert agent.tools_map == tools_map
        assert agent.tool_schemas == tool_schemas
        assert agent.on_end is on_end

    def test_init_without_provider_raises_error(self):
        with pytest.raises(ValueError, match="Provider is required"):
            Agent()


class TestSingleAgentCancel:
    def test_cancel_sets_event(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        assert agent._cancel_event.is_set() is False
        agent.cancel()
        assert agent._cancel_event.is_set() is True


class TestSingleAgentAppendUserPrompt:
    def test_append_user_prompt_adds_message_and_clears_cancel(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.cancel()  # Set cancel event
        agent.append_user_prompt("hello")
        assert len(agent.messages) == 1
        assert isinstance(agent.messages[0], _PlainTextMessageBlock)
        assert agent.messages[0].role == "user"
        assert agent.messages[0].message == "hello"
        assert agent._cancel_event.is_set() is False


class TestSingleAgentAppendAssistantDelta:
    def test_append_assistant_delta_creates_new_block_if_none(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_assistant_delta("hello")
        assert len(agent.messages) == 1
        assert isinstance(agent.messages[0], _PlainTextMessageBlock)
        assert agent.messages[0].role == "assistant"
        assert agent.messages[0].message == "hello"

    def test_append_assistant_delta_appends_to_existing(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_assistant_delta("hello")
        agent.append_assistant_delta(" world")
        assert len(agent.messages) == 1
        assert agent.messages[0].message == "hello world"

    def test_append_assistant_delta_creates_new_if_last_is_user(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_user_prompt("user message")
        agent.append_assistant_delta("assistant message")
        assert len(agent.messages) == 2
        assert agent.messages[0].role == "user"
        assert agent.messages[1].role == "assistant"
        assert agent.messages[1].message == "assistant message"


class TestSingleAgentBuildStandardContext:
    def test_empty_messages(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        context = agent.build_standard_context()
        assert context == []

    def test_only_finished_blocks(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_user_prompt("hello")
        agent.append_assistant_delta("world")
        context = agent.build_standard_context()
        assert context == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_skips_unfinished_blocks(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_user_prompt("hello")
        # Add an unfinished tool call block
        tool_block = _ToolCallMessageBlock("tool", "id1")
        agent.messages.append(tool_block)
        context = agent.build_standard_context()
        # Only the user message should be included
        assert context == [{"role": "user", "content": "hello"}]

    def test_tool_calls_grouped_correctly(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_user_prompt("hello")

        # Add a finished tool call
        tool_block = _ToolCallMessageBlock("my_tool", "call_1")
        tool_block.append_argument_chunk('{"x": 1}')
        tool_block.finish()
        agent.messages.append(tool_block)

        # Add tool result
        result_block = _ToolResultMessageBlock("call_1", "result")
        agent.messages.append(result_block)

        context = agent.build_standard_context()
        assert len(context) == 3
        assert context[0] == {"role": "user", "content": "hello"}
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] is None
        assert len(context[1]["tool_calls"]) == 1
        assert context[1]["tool_calls"][0]["id"] == "call_1"
        assert context[2]["role"] == "tool"
        assert context[2]["tool_call_id"] == "call_1"
        assert context[2]["content"] == "result"

    def test_trailing_tool_calls_at_end(self):
        provider = mock.MagicMock()
        agent = Agent(provider=provider)
        agent.append_user_prompt("hello")

        # Add a finished tool call without result
        tool_block = _ToolCallMessageBlock("my_tool", "call_1")
        tool_block.append_argument_chunk("{}")
        tool_block.finish()
        agent.messages.append(tool_block)

        context = agent.build_standard_context()
        assert len(context) == 2
        assert context[0] == {"role": "user", "content": "hello"}
        assert context[1]["role"] == "assistant"
        assert context[1]["tool_calls"] is not None


class AsyncIterator:
    """Helper class to create async iterators for testing."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.items:
            return self.items.pop(0)
        raise StopAsyncIteration


class TestSingleAgentRun:
    @pytest.fixture
    def provider(self):
        return mock.MagicMock()

    @pytest.fixture
    def agent(self, provider):
        return Agent(provider=provider)

    @pytest.mark.asyncio
    async def test_run_yields_cancelled_when_pre_cancelled(self, agent):
        agent.cancel()
        events = []
        async for event in agent.run():
            events.append(event)
        assert len(events) == 1
        assert events[0]["event"] == "system.cancelled"

    @pytest.mark.asyncio
    async def test_run_yields_text_delta(self, agent, provider):
        # Create mock chunks
        chunk1 = mock.MagicMock()
        chunk1.choices = [mock.MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].finish_reason = None

        chunk2 = mock.MagicMock()
        chunk2.choices = [mock.MagicMock()]
        chunk2.choices[0].delta.content = " World"
        chunk2.choices[0].finish_reason = "stop"

        async def mock_chat(**kwargs):
            return AsyncIterator([chunk1, chunk2])

        provider.chat = mock_chat

        events = []
        async for event in agent.run():
            events.append(event)

        text_events = [e for e in events if e["event"] == "agent.text"]
        assert len(text_events) == 2
        assert text_events[0]["delta"] == "Hello"
        assert text_events[1]["delta"] == " World"

        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1
        assert finish_events[0]["reason"] == "stop"

    @pytest.mark.asyncio
    async def test_run_yields_tool_call_events(self, agent, provider):
        # First chunk with tool call
        chunk1 = mock.MagicMock()
        chunk1.choices = [mock.MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.tool_calls = [mock.MagicMock()]
        chunk1.choices[0].delta.tool_calls[0].index = 0
        chunk1.choices[0].delta.tool_calls[0].id = "call_1"
        chunk1.choices[0].delta.tool_calls[0].function.name = "my_tool"
        chunk1.choices[0].delta.tool_calls[0].function.arguments = '{"x": 1}'
        chunk1.choices[0].finish_reason = "tool_calls"

        # Second chunk (after tool result is sent back) returns normal response
        chunk2 = mock.MagicMock()
        chunk2.choices = [mock.MagicMock()]
        chunk2.choices[0].delta.content = "Tool executed successfully"
        chunk2.choices[0].finish_reason = "stop"

        call_count = [0]

        async def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return AsyncIterator([chunk1])
            else:
                return AsyncIterator([chunk2])

        provider.chat = mock_chat

        # Register a tool
        def my_tool(x):
            return f"result_{x}"

        agent.tools_map = {"my_tool": my_tool}

        events = []
        async for event in agent.run():
            events.append(event)

        header_events = [e for e in events if e["event"] == "agent.tool_call.header"]
        assert len(header_events) == 1
        assert header_events[0]["id"] == "call_1"
        assert header_events[0]["name"] == "my_tool"

        delta_events = [e for e in events if e["event"] == "agent.tool_call.delta"]
        assert len(delta_events) == 1
        assert delta_events[0]["delta"] == '{"x": 1}'

        execution_events = [
            e for e in events if e["event"] == "agent.tool_call.execution"
        ]
        assert len(execution_events) == 1
        assert execution_events[0]["name"] == "my_tool"

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is True

        # Should also have final finish event
        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1

    @pytest.mark.asyncio
    async def test_run_handles_tool_not_registered(self, agent, provider):
        # First chunk triggers tool call
        chunk1 = mock.MagicMock()
        chunk1.choices = [mock.MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.tool_calls = [mock.MagicMock()]
        chunk1.choices[0].delta.tool_calls[0].index = 0
        chunk1.choices[0].delta.tool_calls[0].id = "call_1"
        chunk1.choices[0].delta.tool_calls[0].function.name = "unknown_tool"
        chunk1.choices[0].delta.tool_calls[0].function.arguments = "{}"
        chunk1.choices[0].finish_reason = "tool_calls"

        # Second chunk (after tool result is sent back) returns normal response
        chunk2 = mock.MagicMock()
        chunk2.choices = [mock.MagicMock()]
        chunk2.choices[0].delta.content = "Tool not found"
        chunk2.choices[0].finish_reason = "stop"

        call_count = [0]

        async def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return AsyncIterator([chunk1])
            else:
                return AsyncIterator([chunk2])

        provider.chat = mock_chat

        events = []
        async for event in agent.run():
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is False
        assert "not registered" in result_events[0]["result"]

        # Should also have final finish event
        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1

    @pytest.mark.asyncio
    async def test_run_handles_tool_execution_error(self, agent, provider):
        # First chunk triggers tool call
        chunk1 = mock.MagicMock()
        chunk1.choices = [mock.MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.tool_calls = [mock.MagicMock()]
        chunk1.choices[0].delta.tool_calls[0].index = 0
        chunk1.choices[0].delta.tool_calls[0].id = "call_1"
        chunk1.choices[0].delta.tool_calls[0].function.name = "failing_tool"
        chunk1.choices[0].delta.tool_calls[0].function.arguments = "{}"
        chunk1.choices[0].finish_reason = "tool_calls"

        # Second chunk (after tool result is sent back) returns normal response
        chunk2 = mock.MagicMock()
        chunk2.choices = [mock.MagicMock()]
        chunk2.choices[0].delta.content = "Error occurred"
        chunk2.choices[0].finish_reason = "stop"

        call_count = [0]

        async def mock_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return AsyncIterator([chunk1])
            else:
                return AsyncIterator([chunk2])

        provider.chat = mock_chat

        def failing_tool():
            raise Exception("Tool error")

        agent.tools_map = {"failing_tool": failing_tool}

        events = []
        async for event in agent.run():
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is False
        assert "Error executing tool" in result_events[0]["result"]

        # Should also have final finish event
        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1

    @pytest.mark.asyncio
    async def test_run_calls_on_end_callback(self, agent, provider):
        on_end = mock.MagicMock()
        agent.on_end = on_end

        chunk = mock.MagicMock()
        chunk.choices = [mock.MagicMock()]
        chunk.choices[0].delta.content = "Done"
        chunk.choices[0].finish_reason = "stop"

        async def mock_chat(**kwargs):
            return AsyncIterator([chunk])

        provider.chat = mock_chat

        events = []
        async for event in agent.run():
            events.append(event)

        assert on_end.called
        assert len(events) >= 1
        assert any(e["event"] == "system.finished.standard" for e in events)

    @pytest.mark.asyncio
    async def test_run_handles_async_on_end_callback(self, agent, provider):
        async def async_on_end(agent_obj, messages):
            pass

        agent.on_end = async_on_end

        chunk = mock.MagicMock()
        chunk.choices = [mock.MagicMock()]
        chunk.choices[0].delta.content = "Done"
        chunk.choices[0].finish_reason = "stop"

        async def mock_chat(**kwargs):
            return AsyncIterator([chunk])

        provider.chat = mock_chat

        events = []
        async for event in agent.run():
            events.append(event)

        assert any(e["event"] == "system.finished.standard" for e in events)

    @pytest.mark.asyncio
    async def test_run_skips_chunks_without_choices(self, agent, provider):
        chunk_no_choices = mock.MagicMock()
        chunk_no_choices.choices = None

        chunk_with_content = mock.MagicMock()
        chunk_with_content.choices = [mock.MagicMock()]
        chunk_with_content.choices[0].delta.content = "text"
        chunk_with_content.choices[0].finish_reason = "stop"

        async def mock_chat(**kwargs):
            return AsyncIterator([chunk_no_choices, chunk_with_content])

        provider.chat = mock_chat

        events = []
        async for event in agent.run():
            events.append(event)

        text_events = [e for e in events if e["event"] == "agent.text"]
        assert len(text_events) == 1
        assert text_events[0]["delta"] == "text"

    @pytest.mark.asyncio
    async def test_run_handles_length_finish_reason(self, agent, provider):
        chunk = mock.MagicMock()
        chunk.choices = [mock.MagicMock()]
        chunk.choices[0].delta.content = "truncated"
        chunk.choices[0].finish_reason = "length"

        async def mock_chat(**kwargs):
            return AsyncIterator([chunk])

        provider.chat = mock_chat

        events = []
        async for event in agent.run():
            events.append(event)

        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1
        assert finish_events[0]["reason"] == "length"
