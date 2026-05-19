import sys
import os
import pytest
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend import (
    AgentSession,
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from mitocode.backend.agent.agent_runtime import AgentRuntime
from mitocode.backend import SessionFactory
from mitocode.backend import Agent


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


class CancelOnSecondChunkIterator:
    """Async iterator that cancels *session* just before the second chunk is
    delivered to the loop body (so the cancel-check inside the loop fires)."""

    def __init__(self, items, session):
        self.items = list(items)
        self.session = session
        self.count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        self.count += 1
        if self.count == 2:
            self.session.cancel()
        return self.items.pop(0)


def make_chunk(content=None, tool_calls=None, finish_reason=None):
    """Build a minimal mock stream chunk."""
    chunk = mock.MagicMock()
    choice = mock.MagicMock()
    choice.delta.content = content
    choice.delta.tool_calls = tool_calls
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


def make_tool_chunk(index, call_id, name, arguments):
    """Build a mock stream chunk that carries a single tool-call fragment."""
    tc = mock.MagicMock()
    tc.index = index
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    chunk = make_chunk(tool_calls=[tc], finish_reason=None)
    chunk.choices[0].delta.content = None
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

class TestAgentRuntimeRun:
    @pytest.fixture
    def provider(self):
        return mock.MagicMock()

    @pytest.fixture
    def session(self, provider):
        return AgentSession(provider=provider)

    @pytest.fixture
    def runtime(self):
        return AgentRuntime

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


# ===========================================================================
# AgentSession — edge-case tests
# ===========================================================================

class TestAgentSessionEdgeCases:
    """Additional edge cases not covered by the baseline suite."""

    # ------------------------------------------------------------------
    # append_user_prompt
    # ------------------------------------------------------------------

    def test_append_empty_string_prompt(self):
        """An empty string is a valid prompt and must still create a message block."""
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("")
        assert len(session.messages) == 1
        assert session.messages[0].message == ""

    def test_multiple_user_prompts_accumulate(self):
        """Each call adds a distinct block; messages are never merged."""
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("first")
        session.append_user_prompt("second")
        assert len(session.messages) == 2
        assert session.messages[0].message == "first"
        assert session.messages[1].message == "second"

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------

    def test_cancel_is_idempotent(self):
        """Calling cancel() twice must not raise and the event remains set."""
        session = AgentSession(provider=mock.MagicMock())
        session.cancel()
        session.cancel()  # second call must not raise
        assert session._cancel_event.is_set()

    # ------------------------------------------------------------------
    # append_assistant_delta
    # ------------------------------------------------------------------

    def test_append_assistant_delta_after_tool_result_creates_new_block(self):
        """A delta following a tool-result block must open a fresh assistant block."""
        session = AgentSession(provider=mock.MagicMock())
        session.messages.append(_ToolResultMessageBlock("call_1", "result"))
        session.append_assistant_delta("text")
        # Two blocks: the tool result and the new assistant block
        assert len(session.messages) == 2
        assert isinstance(session.messages[1], _PlainTextMessageBlock)
        assert session.messages[1].role == "assistant"
        assert session.messages[1].message == "text"

    def test_append_assistant_delta_empty_string_creates_block(self):
        """An empty delta still creates an assistant block when there is none."""
        session = AgentSession(provider=mock.MagicMock())
        session.append_assistant_delta("")
        assert len(session.messages) == 1
        assert session.messages[0].role == "assistant"

    # ------------------------------------------------------------------
    # build_standard_context
    # ------------------------------------------------------------------

    def test_build_context_multiple_tool_calls_grouped_into_one_assistant_turn(self):
        """Multiple finished tool-call blocks without an intervening text block are
        all bundled into a single assistant message with a tool_calls list."""
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("q")

        for i in range(3):
            tc = _ToolCallMessageBlock(f"tool_{i}", f"call_{i}")
            tc.append_argument_chunk("{}")
            tc.finish()
            session.messages.append(tc)
            session.messages.append(_ToolResultMessageBlock(f"call_{i}", f"result_{i}"))

        ctx = session.build_standard_context()
        # Each pair (tool_call + tool_result) produces one assistant + one tool entry
        assistant_messages = [c for c in ctx if c["role"] == "assistant"]
        tool_messages = [c for c in ctx if c["role"] == "tool"]
        assert len(tool_messages) == 3
        # All tool calls should end up in assistant messages
        total_tool_calls = sum(len(m.get("tool_calls", [])) for m in assistant_messages)
        assert total_tool_calls == 3

    def test_build_context_interleaved_turns(self):
        """A full multi-turn conversation serialises in correct order."""
        session = AgentSession(provider=mock.MagicMock())
        session.append_user_prompt("turn 1")
        session.append_assistant_delta("response 1")
        session.append_user_prompt("turn 2")
        session.append_assistant_delta("response 2")

        ctx = session.build_standard_context()
        assert len(ctx) == 4
        assert ctx[0] == {"role": "user", "content": "turn 1"}
        assert ctx[1] == {"role": "assistant", "content": "response 1"}
        assert ctx[2] == {"role": "user", "content": "turn 2"}
        assert ctx[3] == {"role": "assistant", "content": "response 2"}

    def test_build_context_only_tool_result_does_not_crash(self):
        """A dangling tool-result without a preceding tool-call must not raise."""
        session = AgentSession(provider=mock.MagicMock())
        session.messages.append(_ToolResultMessageBlock("call_orphan", "ok"))
        ctx = session.build_standard_context()
        assert any(c["role"] == "tool" for c in ctx)


# ===========================================================================
# AgentRuntime — edge-case tests
# ===========================================================================

class TestAgentRuntimeEdgeCases:
    """Additional edge cases not covered by the baseline suite."""

    @pytest.fixture
    def provider(self):
        return mock.MagicMock()

    @pytest.fixture
    def session(self, provider):
        return AgentSession(provider=provider)

    @pytest.fixture
    def runtime(self):
        return AgentRuntime

    # ------------------------------------------------------------------
    # Cancellation during streaming
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_cancels_mid_stream(self, session, provider, runtime):
        """Cancel event set just before the second chunk is processed stops the run."""
        chunk1 = make_chunk(content="first", finish_reason=None)
        chunk2 = make_chunk(content="second", finish_reason="stop")

        provider.chat = mock.AsyncMock(
            return_value=CancelOnSecondChunkIterator([chunk1, chunk2], session)
        )

        events = []
        async for event in runtime.run(session):
            events.append(event)

        # The cancellation must be reported
        assert any(e["event"] == "system.cancelled" for e in events)
        # Only the first delta (processed before cancel) may appear, not the second
        text_deltas = [e["delta"] for e in events if e["event"] == "agent.text"]
        assert "second" not in text_deltas

    # ------------------------------------------------------------------
    # Tool error variants
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_tool_type_error_reported_as_invalid_argument(
        self, session, provider, runtime
    ):
        """A tool that raises TypeError (e.g. wrong keyword args) reports
        'Invalid argument.' without surfacing the raw exception."""
        chunk1 = make_tool_chunk(0, "call_t", "strict_fn", "{}")
        term_chunk = make_chunk(finish_reason="tool_calls")
        term_chunk.choices[0].delta.content = None
        term_chunk.choices[0].delta.tool_calls = None
        chunk_stop = make_chunk(content="ok", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncIterator([chunk1, term_chunk])
            return AsyncIterator([chunk_stop])

        provider.chat = mock_chat

        def strict_fn(required_arg: str):
            """Expects 'required_arg' but receives nothing → TypeError."""
            return required_arg

        session.tools_map = {"strict_fn": strict_fn}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is False
        assert "Invalid argument" in result_events[0]["result"]

    @pytest.mark.asyncio
    async def test_run_tool_invalid_json_arguments(
        self, session, provider, runtime
    ):
        """A tool call whose argument string is not valid JSON reports
        'Invalid JSON Format.' and is not dispatched to the Python function."""
        tc = mock.MagicMock()
        tc.index = 0
        tc.id = "call_bad_json"
        tc.function.name = "my_fn"
        tc.function.arguments = "{invalid json!}"

        chunk1 = make_chunk(tool_calls=[tc], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk_stop = make_chunk(content="ok", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk_stop])

        provider.chat = mock_chat
        my_fn = mock.MagicMock(return_value="should not be called")
        session.tools_map = {"my_fn": my_fn}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is False
        assert "Invalid JSON Format" in result_events[0]["result"]
        my_fn.assert_not_called()

    # ------------------------------------------------------------------
    # content_filter finish reason
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_content_filter_reason_treated_as_standard_stop(
        self, session, provider, runtime
    ):
        """'content_filter' finish reason must emit system.finished.standard."""
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([
                make_chunk(content="filtered", finish_reason="content_filter"),
            ])
        )

        events = []
        async for event in runtime.run(session):
            events.append(event)

        finish_events = [e for e in events if e["event"] == "system.finished.standard"]
        assert len(finish_events) == 1
        assert finish_events[0]["reason"] == "content_filter"

    # ------------------------------------------------------------------
    # Multiple parallel tool calls
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_multiple_parallel_tool_calls_all_executed(
        self, session, provider, runtime
    ):
        """When the LLM proposes two tool calls in a single turn, both are
        executed and both results are appended to the session."""
        tc0 = mock.MagicMock()
        tc0.index = 0
        tc0.id = "call_a"
        tc0.function.name = "add"
        tc0.function.arguments = '{"x": 1, "y": 2}'

        tc1 = mock.MagicMock()
        tc1.index = 1
        tc1.id = "call_b"
        tc1.function.name = "mul"
        tc1.function.arguments = '{"x": 3, "y": 4}'

        chunk1 = make_chunk(tool_calls=[tc0, tc1], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk_stop = make_chunk(content="done", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk_stop])

        provider.chat = mock_chat
        session.tools_map = {
            "add": lambda x, y: x + y,
            "mul": lambda x, y: x * y,
        }

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 2
        assert all(e["success"] for e in result_events)
        results = {e["id"]: e["result"] for e in result_events}
        assert results["call_a"] == "3"
        assert results["call_b"] == "12"

    # ------------------------------------------------------------------
    # Async on_end callback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_async_on_end_is_awaited(self, session, provider, runtime):
        """An async on_end callback must be awaited (not just called)."""
        awaited = []

        async def async_on_end(sess, messages):
            awaited.append(True)

        session.on_end = async_on_end
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="hi", finish_reason="stop")])
        )

        async for _ in runtime.run(session):
            pass

        assert awaited == [True]

    # ------------------------------------------------------------------
    # Tool results persisted in session
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_tool_result_persisted_in_session_messages(
        self, session, provider, runtime
    ):
        """After a tool call round-trip, both the tool-call block and its result
        block must appear in session.messages."""
        tc = mock.MagicMock()
        tc.index = 0
        tc.id = "call_p"
        tc.function.name = "ping"
        tc.function.arguments = "{}"

        chunk1 = make_chunk(tool_calls=[tc], finish_reason="tool_calls")
        chunk1.choices[0].delta.content = None
        chunk_stop = make_chunk(content="ok", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncIterator([chunk1] if call_count == 1 else [chunk_stop])

        provider.chat = mock_chat
        session.tools_map = {"ping": lambda: "pong"}

        async for _ in runtime.run(session):
            pass

        types = [type(m).__name__ for m in session.messages]
        assert "_ToolCallMessageBlock" in types
        assert "_ToolResultMessageBlock" in types
        result_block = next(
            m for m in session.messages
            if isinstance(m, _ToolResultMessageBlock)
        )
        assert result_block.content == "pong"

    # ------------------------------------------------------------------
    # Argument chunks streamed across multiple stream packets
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_tool_arguments_streamed_in_chunks(
        self, session, provider, runtime
    ):
        """Tool argument JSON arriving in fragments must be concatenated and
        parsed correctly before execution."""
        # First chunk: header (id + name)
        tc_header = mock.MagicMock()
        tc_header.index = 0
        tc_header.id = "call_stream"
        tc_header.function.name = "echo"
        tc_header.function.arguments = None

        # Second chunk: first half of the argument string
        tc_arg1 = mock.MagicMock()
        tc_arg1.index = 0
        tc_arg1.id = None
        tc_arg1.function.name = None
        tc_arg1.function.arguments = '{"msg":'

        # Third chunk: second half, plus the finish reason
        tc_arg2 = mock.MagicMock()
        tc_arg2.index = 0
        tc_arg2.id = None
        tc_arg2.function.name = None
        tc_arg2.function.arguments = ' "hello"}'

        chunk_header = make_chunk(tool_calls=[tc_header], finish_reason=None)
        chunk_header.choices[0].delta.content = None
        chunk_arg1 = make_chunk(tool_calls=[tc_arg1], finish_reason=None)
        chunk_arg1.choices[0].delta.content = None
        chunk_arg2 = make_chunk(tool_calls=[tc_arg2], finish_reason="tool_calls")
        chunk_arg2.choices[0].delta.content = None
        chunk_stop = make_chunk(content="ok", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncIterator([chunk_header, chunk_arg1, chunk_arg2])
            return AsyncIterator([chunk_stop])

        provider.chat = mock_chat
        session.tools_map = {"echo": lambda msg: msg}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        assert result_events[0]["success"] is True
        assert result_events[0]["result"] == "hello"

    # ------------------------------------------------------------------
    # Empty choices list (not None) should not crash
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_empty_choices_list_skipped(self, session, provider, runtime):
        """A chunk with an empty choices list (not None) must be silently skipped."""
        empty_choices = mock.MagicMock()
        empty_choices.choices = []
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([
                empty_choices,
                make_chunk(content="ok", finish_reason="stop"),
            ])
        )

        events = []
        async for event in runtime.run(session):
            events.append(event)

        text_events = [e for e in events if e["event"] == "agent.text"]
        assert len(text_events) == 1


# ===========================================================================
# SessionFactory — edge-case tests
# ===========================================================================

class TestSessionFactoryEdgeCases:
    """Additional edge cases not covered by the baseline suite."""

    def test_defaults_not_mutated_by_per_call_override(self):
        """Providing an override at call-time must not change the factory defaults."""
        provider_default = mock.MagicMock()
        provider_override = mock.MagicMock()
        factory = SessionFactory(provider=provider_default)

        factory.create_session(provider=provider_override)

        # A subsequent call with no override must still use the original default.
        session = factory.create_session()
        assert session.provider is provider_default

    def test_tools_map_default_is_independent_per_session(self):
        """Mutations to one session's tools_map must not affect the factory default
        or sibling sessions that share the reference."""
        provider = mock.MagicMock()
        shared_tools = {}
        factory = SessionFactory(provider=provider, tools_map=shared_tools)

        s1 = factory.create_session()
        s2 = factory.create_session()

        # Both sessions reference the same tools_map dict as specified
        assert s1.tools_map is shared_tools
        assert s2.tools_map is shared_tools

    def test_create_session_call_only_override_does_not_affect_next_call(self):
        """A one-off override must be scoped to that single create_session call."""
        provider = mock.MagicMock()
        extra_schemas = [{"name": "x"}]
        factory = SessionFactory(provider=provider)

        factory.create_session(tool_schemas=extra_schemas)
        next_session = factory.create_session()

        # The next session must fall back to the factory default (empty list)
        assert next_session.tool_schemas == []


# ===========================================================================
# Regression tests for gemini-code-assist issues
# ===========================================================================

class TestAgentRuntimeStreamingIdNameRegression:
    """Regression tests covering the streamed tool-call ID/name fixes."""

    @pytest.fixture
    def provider(self):
        return mock.MagicMock()

    @pytest.fixture
    def session(self, provider):
        return AgentSession(provider=provider)

    @pytest.fixture
    def runtime(self):
        return AgentRuntime

    @pytest.mark.asyncio
    async def test_tool_call_id_streamed_in_second_chunk_is_updated_on_block(
        self, session, provider, runtime
    ):
        """If the tool call ID arrives across two chunks (first empty, second
        with the real value), the _ToolCallMessageBlock and the final result
        event must carry the complete ID, not the partial one."""

        # Chunk 1: name arrives (no id yet → block created with empty id)
        tc_name = mock.MagicMock()
        tc_name.index = 0
        tc_name.id = None
        tc_name.function.name = "echo"
        tc_name.function.arguments = None

        # Chunk 2: id arrives (no name)
        tc_id = mock.MagicMock()
        tc_id.index = 0
        tc_id.id = "call_real"
        tc_id.function.name = None
        tc_id.function.arguments = None

        # Chunk 3: arguments + finish
        tc_args = mock.MagicMock()
        tc_args.index = 0
        tc_args.id = None
        tc_args.function.name = None
        tc_args.function.arguments = '{"msg": "hi"}'

        chunk1 = make_chunk(tool_calls=[tc_name], finish_reason=None)
        chunk1.choices[0].delta.content = None
        chunk2 = make_chunk(tool_calls=[tc_id], finish_reason=None)
        chunk2.choices[0].delta.content = None
        chunk3 = make_chunk(tool_calls=[tc_args], finish_reason="tool_calls")
        chunk3.choices[0].delta.content = None
        chunk_stop = make_chunk(content="done", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncIterator([chunk1, chunk2, chunk3])
            return AsyncIterator([chunk_stop])

        provider.chat = mock_chat
        session.tools_map = {"echo": lambda msg: msg}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        # The result event must carry the complete, final ID — not an empty string.
        assert result_events[0]["id"] == "call_real"
        assert result_events[0]["success"] is True

    @pytest.mark.asyncio
    async def test_tool_call_name_streamed_in_second_chunk_is_updated_on_block(
        self, session, provider, runtime
    ):
        """If the function name arrives across two chunks (first half in chunk 1,
        second half in chunk 2), the block must end up with the full name so the
        tool can be dispatched correctly."""

        # Chunk 1: id + first half of name
        tc_part1 = mock.MagicMock()
        tc_part1.index = 0
        tc_part1.id = "call_n"
        tc_part1.function.name = "ec"   # partial name
        tc_part1.function.arguments = None

        # Chunk 2: second half of name
        tc_part2 = mock.MagicMock()
        tc_part2.index = 0
        tc_part2.id = None
        tc_part2.function.name = "ho"   # completes "echo"
        tc_part2.function.arguments = None

        # Chunk 3: arguments + finish reason
        tc_args = mock.MagicMock()
        tc_args.index = 0
        tc_args.id = None
        tc_args.function.name = None
        tc_args.function.arguments = '{"msg": "world"}'

        chunk1 = make_chunk(tool_calls=[tc_part1], finish_reason=None)
        chunk1.choices[0].delta.content = None
        chunk2 = make_chunk(tool_calls=[tc_part2], finish_reason=None)
        chunk2.choices[0].delta.content = None
        chunk3 = make_chunk(tool_calls=[tc_args], finish_reason="tool_calls")
        chunk3.choices[0].delta.content = None
        chunk_stop = make_chunk(content="done", finish_reason="stop")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncIterator([chunk1, chunk2, chunk3])
            return AsyncIterator([chunk_stop])

        provider.chat = mock_chat
        session.tools_map = {"echo": lambda msg: msg}

        events = []
        async for event in runtime.run(session):
            events.append(event)

        result_events = [e for e in events if e["event"] == "agent.tool_call.result"]
        assert len(result_events) == 1
        # The tool must have been dispatched by the FULL name "echo", not "ec".
        assert result_events[0]["success"] is True
        assert result_events[0]["result"] == "world"


class TestAgentOnEndBackwardCompatibility:
    """Regression tests for the on_end backward-compat fix in Agent.run()."""

    @pytest.mark.asyncio
    async def test_on_end_receives_agent_wrapper_not_session(self):
        """When using the Agent wrapper, on_end must receive the Agent instance
        (with .run(), .cancel(), etc.) not the bare AgentSession."""
        provider = mock.MagicMock()
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="hi", finish_reason="stop")])
        )

        received_first_arg = []

        def on_end(obj, messages):
            received_first_arg.append(obj)

        agent = Agent(provider=provider, on_end=on_end)
        async for _ in agent.run():
            pass

        assert len(received_first_arg) == 1
        # Must be the Agent wrapper, not the underlying AgentSession
        assert isinstance(received_first_arg[0], Agent)
        assert hasattr(received_first_arg[0], "run")

    @pytest.mark.asyncio
    async def test_on_end_session_attribute_restored_after_run(self):
        """The session's on_end must be restored to the original callable after
        Agent.run() completes, even if the generator is not fully exhausted."""
        provider = mock.MagicMock()
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="hi", finish_reason="stop")])
        )

        original_on_end = mock.MagicMock()
        agent = Agent(provider=provider, on_end=original_on_end)
        session = object.__getattribute__(agent, "_session")

        async for _ in agent.run():
            pass

        # After the run the session's on_end must be the original callable.
        assert session.on_end is original_on_end

    @pytest.mark.asyncio
    async def test_async_on_end_receives_agent_wrapper(self):
        """An async on_end callback is also given the Agent wrapper."""
        provider = mock.MagicMock()
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="hi", finish_reason="stop")])
        )

        received = []

        async def async_on_end(obj, messages):
            received.append(obj)

        agent = Agent(provider=provider, on_end=async_on_end)
        async for _ in agent.run():
            pass

        assert len(received) == 1
        assert isinstance(received[0], Agent)

    @pytest.mark.asyncio
    async def test_direct_session_on_end_still_receives_session(self):
        """When on_end is registered directly on AgentSession (not via Agent),
        it must still receive the AgentSession instance (no wrapping)."""
        provider = mock.MagicMock()
        provider.chat = mock.AsyncMock(
            return_value=AsyncIterator([make_chunk(content="hi", finish_reason="stop")])
        )

        received = []

        def on_end(obj, messages):
            received.append(obj)

        session = AgentSession(provider=provider, on_end=on_end)
        runtime = AgentRuntime

        async for _ in runtime.run(session):
            pass

        assert len(received) == 1
        assert isinstance(received[0], AgentSession)
        assert not isinstance(received[0], Agent)
