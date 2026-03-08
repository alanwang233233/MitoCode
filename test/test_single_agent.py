import sys
import os
import types
import unittest.mock as mock

# Mock openai and loguru before importing _single_agent (they are not installed in the test environment)
_mock_openai = types.ModuleType("openai")
_mock_openai.OpenAI = mock.MagicMock()
sys.modules.setdefault("openai", _mock_openai)

_mock_loguru = types.ModuleType("loguru")
_mock_loguru.logger = mock.MagicMock()
sys.modules.setdefault("loguru", _mock_loguru)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

from _single_agent import _SingleAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(prompt=None, **kwargs):
    """Create a _SingleAgent with a fresh mocked OpenAI client."""
    if prompt is None:
        prompt = []
    agent = _SingleAgent(prompt, **kwargs)
    # Replace the OpenAI client with a plain MagicMock so tests control responses.
    agent.client = mock.MagicMock()
    return agent


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestSingleAgentInit:
    def test_default_settings(self):
        agent = _make_agent()
        assert agent.model == "gpt-5"
        assert agent.tools == []
        assert agent.stream is False
        assert agent.prompt == []

    def test_custom_settings(self):
        on_token = mock.MagicMock()
        on_end = mock.MagicMock()
        prompt = [{"role": "system", "content": "You are helpful."}]
        agent = _make_agent(
            prompt=prompt,
            base_url="https://custom.api/v1",
            api_key="test-key",
            model="gpt-4",
            tools=["tool1"],
            stream=True,
            on_token=on_token,
            on_end=on_end,
        )
        assert agent.model == "gpt-4"
        assert agent.tools == ["tool1"]
        assert agent.stream is True
        assert agent.on_token is on_token
        assert agent.on_end is on_end
        assert agent.prompt == prompt

    def test_prompt_stored_by_reference(self):
        """The agent should hold the same prompt list that was passed in."""
        prompt = [{"role": "user", "content": "hello"}]
        agent = _make_agent(prompt=prompt)
        assert agent.prompt is prompt


# ---------------------------------------------------------------------------
# append_raw_message tests
# ---------------------------------------------------------------------------

class TestAppendRawMessage:
    def test_appends_dict_message(self):
        agent = _make_agent()
        msg = {"role": "user", "content": "hi"}
        agent.append_raw_message(msg)
        assert agent.prompt == [msg]

    def test_appends_multiple_messages(self):
        agent = _make_agent()
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        for m in msgs:
            agent.append_raw_message(m)
        assert agent.prompt == msgs


# ---------------------------------------------------------------------------
# append_str_message tests
# ---------------------------------------------------------------------------

class TestAppendStrMessage:
    def test_user_message(self):
        agent = _make_agent()
        agent.append_str_message("hello", is_user=True)
        assert agent.prompt == [{"role": "user", "content": "hello"}]

    def test_assistant_message(self):
        agent = _make_agent()
        agent.append_str_message("world", is_user=False)
        assert agent.prompt == [{"role": "assistant", "content": "world"}]

    def test_default_is_user(self):
        """is_user defaults to True."""
        agent = _make_agent()
        agent.append_str_message("hey")
        assert agent.prompt[0]["role"] == "user"


# ---------------------------------------------------------------------------
# run() – non-stream mode
# ---------------------------------------------------------------------------

class TestRunNonStream:
    def _build_response(self, content):
        response = mock.MagicMock()
        response.choices[0].message.content = content
        return response

    def test_run_returns_response(self):
        agent = _make_agent()
        fake_response = self._build_response("answer")
        agent.client.chat.completions.create.return_value = fake_response
        result = agent.run()
        assert result is fake_response

    def test_run_appends_assistant_message(self):
        agent = _make_agent(prompt=[{"role": "user", "content": "q"}])
        fake_response = self._build_response("answer")
        agent.client.chat.completions.create.return_value = fake_response
        agent.run()
        assert agent.prompt[-1] == {"role": "assistant", "content": "answer"}

    def test_run_calls_create_with_correct_args(self):
        tools = [{"type": "function", "function": {"name": "my_tool"}}]
        agent = _make_agent(model="gpt-4", tools=tools)
        fake_response = self._build_response("ok")

        # The prompt list is passed by reference, so capture a snapshot via side_effect
        # before run() mutates it with the appended assistant message.
        captured_kwargs = {}

        def _capture(**kwargs):
            captured_kwargs.update({k: list(v) if isinstance(v, list) else v for k, v in kwargs.items()})
            return fake_response

        agent.client.chat.completions.create.side_effect = _capture
        agent.run()

        assert captured_kwargs["model"] == "gpt-4"
        assert captured_kwargs["messages"] == []
        assert captured_kwargs["tools"] == tools
        assert captured_kwargs["stream"] is False

    def test_run_passes_none_when_no_tools(self):
        """When tools list is empty, None should be passed to the API."""
        agent = _make_agent()
        fake_response = self._build_response("ok")
        agent.client.chat.completions.create.return_value = fake_response
        agent.run()
        _, kwargs = agent.client.chat.completions.create.call_args
        assert kwargs["tools"] is None


# ---------------------------------------------------------------------------
# run() – stream mode
# ---------------------------------------------------------------------------

def _make_chunk(content):
    chunk = mock.MagicMock()
    chunk.choices = [mock.MagicMock()]
    chunk.choices[0].delta.content = content
    return chunk


def _make_empty_chunk():
    """A chunk with no choices (should be skipped)."""
    chunk = mock.MagicMock()
    chunk.choices = []
    return chunk


class TestRunStream:
    def test_stream_calls_on_token_for_each_chunk(self):
        on_token = mock.MagicMock()
        on_end = mock.MagicMock()
        agent = _make_agent(stream=True, on_token=on_token, on_end=on_end)

        chunks = [_make_chunk("Hello"), _make_chunk(" world")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent.run()

        assert on_token.call_count == 2
        # First call: full_message="Hello", delta="Hello"
        on_token.assert_any_call(agent, "Hello", "Hello")
        # Second call: full_message="Hello world", delta=" world"
        on_token.assert_any_call(agent, "Hello world", " world")

    def test_stream_calls_on_end_with_full_message(self):
        on_end = mock.MagicMock()
        agent = _make_agent(stream=True, on_end=on_end)

        chunks = [_make_chunk("Hi"), _make_chunk("!")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent.run()

        on_end.assert_called_once_with(agent, "Hi!")

    def test_stream_appends_full_message_to_prompt(self):
        agent = _make_agent(stream=True)
        chunks = [_make_chunk("foo"), _make_chunk("bar")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent.run()

        assert agent.prompt[-1] == {"role": "assistant", "content": "foobar"}

    def test_stream_skips_chunks_without_choices(self):
        on_token = mock.MagicMock()
        agent = _make_agent(stream=True, on_token=on_token)

        chunks = [_make_empty_chunk(), _make_chunk("ok")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent.run()

        # on_token should only be called for the chunk that has content
        assert on_token.call_count == 1

    def test_stream_skips_none_delta_content(self):
        """Chunks whose delta.content is None should not contribute to the message."""
        on_token = mock.MagicMock()
        agent = _make_agent(stream=True, on_token=on_token)

        chunk_none = mock.MagicMock()
        chunk_none.choices = [mock.MagicMock()]
        chunk_none.choices[0].delta.content = None

        chunks = [chunk_none, _make_chunk("text")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent.run()

        assert on_token.call_count == 1
        assert agent.prompt[-1]["content"] == "text"
