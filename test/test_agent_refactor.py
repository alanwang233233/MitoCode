import sys
import os
import pytest
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.agent.agent_session import (
    AgentSession,
    _MessageBlock,
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from backend.agent.agent_runtime import AgentRuntime
from backend.agent.session_factory import SessionFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AsyncIterator:
    """Minimal async iterator used to mock provider streams."""
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.items:
            return self.items.pop(0)
        raise StopAsyncIteration


def make_chunk(content=None, tool_calls=None, finish_reason=None):
    """Build a minimal mock stream chunk."""
    chunk = mock.MagicMock()
    choice = mock.MagicMock()
    choice.delta.content = content
    choice.delta.tool_calls = tool_calls
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


# ===========================================================================
# AgentSession tests
# ===========================================================================

class TestAgentSessionInit:
    def test_requires_provider(self):
        with pytest.raises(ValueError, match="Provider is required"):
            AgentSession()

    def test_default_attributes(self):
        provider = mock.MagicMock()
        session = AgentSession(provider=provider)
        assert session.provider is provider
        assert session.messages == []
        assert session.tools_map == {}
        assert session.tool_schemas == []
        assert session.on_end is None
        assert not session._cancel_event.is_set()

    def test_custom_attributes(self):
        provider = mock.MagicMock()
        tools_map = {"t": lambda: None}
        tool_schemas = [{"name": "t"}]
        on_end = mock.MagicMock()
        session = AgentSession(
            provider=provider,
            tools_map=tools_map,
            tool_schemas=tool_schemas,
            on_end=on_end,
        )
        assert session.tools_map is tools_map
        assert session.tool_schemas is tool_schemas
        assert session.on_end is on_end


class TestAgentSessionCancel:
    def test_cancel_sets_event(self):
        session = AgentSession(provider=mock.MagicMock())
        assert not session._cancel_event.is_set()
        session.cancel()
        assert session._cancel_event.is_set()


class TestAgentSessionAppendUserPrompt:
    def test_adds_message(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("hello")
        assert len(session.messages) == 1
        assert isinstance(session.messages[0], _PlainTextMessageBlock)
        assert session.messages[0].role == "user"
        assert session.messages[0].message == "hello"

    def test_clears_cancel_event(self):
        session = AgentSession(provider=mock.MagicMock())
        session.cancel()
        session.append_user_prompt("hello")
        assert not session._cancel_event.is_set()


class TestAgentSessionAppendAssistantDelta:
    def test_creates_new_block_when_empty(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_assistant_delta("hi")
        assert len(session.messages) == 1
        assert session.messages[0].role == "assistant"
        assert session.messages[0].message == "hi"

    def test_appends_to_existing_assistant_block(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_assistant_delta("hello")
        session.append_assistant_delta(" world")
        assert len(session.messages) == 1
        assert session.messages[0].message == "hello world"

    def test_creates_new_block_after_user_message(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("question")
        session.append_assistant_delta("answer")
        assert len(session.messages) == 2
        assert session.messages[1].role == "assistant"


class TestAgentSessionBuildStandardContext:
    def test_empty(self):
        session = AgentSession(provider=mock.MagicMock())
        assert session.build_standard_context() == []

    def test_user_and_assistant(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("hi")
        session.append_assistant_delta("hello")
        ctx = session.build_standard_context()
        assert ctx == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_skips_unfinished_blocks(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("q")
        session.messages.append(_ToolCallMessageBlock("t", "id1"))  # unfinished
        ctx = session.build_standard_context()
        assert ctx == [{"role": "user", "content": "q"}]

    def test_tool_call_and_result_grouped(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("q")
        tc = _ToolCallMessageBlock("my_tool", "call_1")
        tc.append_argument_chunk('{"x": 1}')
        tc.finish()
        session.messages.append(tc)
        session.messages.append(_ToolResultMessageBlock("call_1", "ok"))
        ctx = session.build_standard_context()
        assert len(ctx) == 3
        assert ctx[1]["role"] == "assistant"
        assert ctx[1]["content"] is None
        assert len(ctx[1]["tool_calls"]) == 1
        assert ctx[2]["role"] == "tool"
        assert ctx[2]["tool_call_id"] == "call_1"

    def test_trailing_tool_calls(self):
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("q")
        tc = _ToolCallMessageBlock("t", "call_2")
        tc.append_argument_chunk("{}")
        tc.finish()
        session.messages.append(tc)
        ctx = session.build_standard_context()
        assert len(ctx) == 2
        assert ctx[1]["role"] == "assistant"
        assert ctx[1]["tool_calls"] is not None


# ===========================================================================
# AgentRuntime tests
# ===========================================================================

class TestAgentRuntimeSingleton:
    def test_get_instance_returns_same_object(self):
        a = AgentRuntime.get_instance()
        b = AgentRuntime.get_instance()
        assert a is b

    def test_constructor_returns_same_object(self):
        a = AgentRuntime()
        b = AgentRuntime()
        assert a is b

    def test_get_instance_equals_constructor(self):
        assert AgentRuntime.get_instance() is AgentRuntime()


class TestAgentRuntimeRun:
    @pytest.fixture
    def provider(self):
        return mock.MagicMock()

    @pytest.fixture
    def session(self, provider):
        return AgentSession(provider=provider)

    @pytest.fixture
    def runtime(self):
        return AgentRuntime.get_instance()

    @pytest.mark.asyncio
    async def test_run_yields_cancelled_when_pre_cancelled(self, session, runtime):
        session.cancel()
        events = []
        async for event in runtime.run(session):
            events.append(event)
        assert events == [{"event": "system.cancelled"}]

    @pytest.mark.asyncio
    async def test_run_yields_text_events(self, session, provider, runtime):
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([
                make_chunk(content="Hello", finish_reason=None),
                make_chunk(content=" World", finish_reason="stop"),
            ])
        )
        events = []
        async for event in runtime.run(session):
            events.append(event)

        text_events = [e for e in events if e["event"] == "agent.text"]
        assert len(text_events) == 2
        assert text_events[0]["delta"] == "Hello"
        assert text_events[1]["delta"] == " World"

        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1
        assert finish_events[0]["reason"] == "stop"

    @pytest.mark.asyncio
    async def test_run_yields_length_reason(self, session, provider, runtime):
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([
                make_chunk(content="truncated", finish_reason="length"),
            ])
        )
        events = []
        async for event in runtime.run(session):
            events.append(event)

        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1
        assert finish_events[0]["reason"] == "length"

    @pytest.mark.asyncio
    async def test_run_skips_chunks_without_choices(self, session, provider, runtime):
        no_choices = mock.MagicMock()
        no_choices.choices = None
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([
                no_choices,
                make_chunk(content="text", finish_reason="stop"),
            ])
        )
        events = []
        async for event in runtime.run(session):
            events.append(event)

        text_events = [e for e in events if e["event"] == "agent.text"]
        assert len(text_events) == 1
        assert text_events[0]["delta"] == "text"

    @pytest.mark.asyncio
    async def test_run_executes_tool_call(self, session, provider, runtime):
        tc_chunk = mock.MagicMock()
        tc_chunk.index = 0
        tc_chunk.id = "call_1"
        tc_chunk.function.name = "greet"
        tc_chunk.function.arguments = '{"name": "world"}'

        chunk1 = make_chunk(tool_calls=[tc_chunk], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk2 = make_chunk(content="Done", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk2])

        provider.chat = mock_chat
        session.tools_map = {"greet": lambda name: f"Hello, {name}!"}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        assert any(e["event"] == "agent.tool_call.header" for e in events)
        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is True
        assert "Hello, world!" in result_events[0]["result"]

    @pytest.mark.asyncio
    async def test_run_tool_not_registered(self, session, provider, runtime):
        tc_chunk = mock.MagicMock()
        tc_chunk.index = 0
        tc_chunk.id = "call_x"
        tc_chunk.function.name = "unknown"
        tc_chunk.function.arguments = "{}"

        chunk1 = make_chunk(tool_calls=[tc_chunk], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk2 = make_chunk(content="ok", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk2])

        provider.chat = mock_chat

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is False
        assert "not registered" in result_events[0]["result"]

    @pytest.mark.asyncio
    async def test_run_tool_execution_error(self, session, provider, runtime):
        tc_chunk = mock.MagicMock()
        tc_chunk.index = 0
        tc_chunk.id = "call_e"
        tc_chunk.function.name = "bad"
        tc_chunk.function.arguments = "{}"

        chunk1 = make_chunk(tool_calls=[tc_chunk], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk2 = make_chunk(content="ok", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk2])

        provider.chat = mock_chat

        def bad():
            raise RuntimeError("boom")

        session.tools_map = {"bad": bad}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is False
        assert "Error executing tool" in result_events[0]["result"]

    @pytest.mark.asyncio
    async def test_run_async_tool(self, session, provider, runtime):
        tc_chunk = mock.MagicMock()
        tc_chunk.index = 0
        tc_chunk.id = "call_async"
        tc_chunk.function.name = "async_tool"
        tc_chunk.function.arguments = "{}"

        chunk1 = make_chunk(tool_calls=[tc_chunk], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk2 = make_chunk(content="done", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk2])

        provider.chat = mock_chat

        async def async_tool():
            return "async_result"

        session.tools_map = {"async_tool": async_tool}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is True
        assert result_events[0]["result"] == "async_result"

    @pytest.mark.asyncio
    async def test_run_calls_on_end(self, session, provider, runtime):
        on_end = mock.MagicMock()
        session.on_end = on_end

        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="hi", finish_reason="stop")])
        )

        events = []
        async for event in runtime.run(session):
            events.append(event)

        assert on_end.called
        assert any(e["event"] == "system.finished.standard" for e in events)

    @pytest.mark.asyncio
    async def test_run_state_is_isolated_across_sessions(self, provider, runtime):
        """Two sessions driven by the same runtime accumulate state independently."""
        session_a = AgentSession(provider=provider)
        session_b = AgentSession(provider=provider)

        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="reply", finish_reason="stop")])
        )

        session_a.append_user_prompt("question_a")
        async for _ in runtime.run(session_a):
            pass

        assert len(session_a.messages) == 2  # user + assistant
        assert len(session_b.messages) == 0  # unaffected


# ===========================================================================
# SessionFactory tests
# ===========================================================================

class TestSessionFactory:
    def test_create_session_with_no_defaults(self):
        provider = mock.MagicMock()
        factory = SessionFactory()
        session = factory.create_session(provider=provider)
        assert isinstance(session, AgentSession)
        assert session.provider is provider

    def test_factory_defaults_applied(self):
        provider = mock.MagicMock()
        tools = {"t": lambda: None}
        factory = SessionFactory(provider=provider, tools_map=tools)
        session = factory.create_session()
        assert session.provider is provider
        assert session.tools_map is tools

    def test_call_settings_override_defaults(self):
        default_provider = mock.MagicMock()
        override_provider = mock.MagicMock()
        factory = SessionFactory(provider=default_provider)
        session = factory.create_session(provider=override_provider)
        assert session.provider is override_provider

    def test_each_call_returns_independent_session(self):
        provider = mock.MagicMock()
        factory = SessionFactory(provider=provider)
        s1 = factory.create_session()
        s2 = factory.create_session()
        assert s1 is not s2
        assert s1.messages is not s2.messages

    def test_without_provider_raises(self):
        factory = SessionFactory()
        with pytest.raises(ValueError, match="Provider is required"):
            factory.create_session()

    def test_on_end_forwarded(self):
        provider = mock.MagicMock()
        on_end = mock.MagicMock()
        factory = SessionFactory(provider=provider, on_end=on_end)
        session = factory.create_session()
        assert session.on_end is on_end

    def test_tool_schemas_forwarded(self):
        provider = mock.MagicMock()
        schemas = [{"name": "search"}]
        factory = SessionFactory(provider=provider, tool_schemas=schemas)
        session = factory.create_session()
        assert session.tool_schemas is schemas
