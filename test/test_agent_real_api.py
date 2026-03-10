"""Integration tests that exercise _SingleAgent against a real OpenAI-compatible API.

All tests in this module are skipped automatically when the OPENAI_API_KEY
environment variable is not set, so the CI suite can run without credentials.
Set the optional OPENAI_BASE_URL and TEST_MODEL variables to target a different
endpoint or model (defaults: https://api.openai.com/v1, gpt-4o-mini).
"""

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module when no API key is available.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set – skipping real API tests",
)

# ---------------------------------------------------------------------------
# Import the real _SingleAgent (no mocking).
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

from _single_agent import _SingleAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test configuration (read from environment).
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("TEST_MODEL", "gpt-4o-mini")


def _make_real_agent(prompt=None, **kwargs):
    """Return a _SingleAgent configured to call the real API."""
    if prompt is None:
        prompt = []
    return _SingleAgent(
        prompt,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestRealAPIInit:
    def test_agent_stores_model(self):
        agent = _make_real_agent()
        assert agent.model == MODEL

    def test_agent_client_is_not_none(self):
        agent = _make_real_agent()
        assert agent.client is not None

    def test_agent_stores_prompt_by_reference(self):
        prompt = [{"role": "user", "content": "hi"}]
        agent = _make_real_agent(prompt=prompt)
        assert agent.prompt is prompt


# ---------------------------------------------------------------------------
# Non-streaming run()
# ---------------------------------------------------------------------------


class TestRealAPINonStream:
    def test_run_returns_response(self):
        prompt = [{"role": "user", "content": "Reply with only the word 'yes'."}]
        agent = _make_real_agent(prompt=prompt)
        response = agent.run()
        assert response is not None
        assert response.choices[0].message.content is not None

    def test_run_response_contains_text(self):
        prompt = [{"role": "user", "content": "Reply with only the word 'yes'."}]
        agent = _make_real_agent(prompt=prompt)
        agent.run()
        content = agent.prompt[-1]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_run_appends_assistant_message(self):
        prompt = [{"role": "user", "content": "Say 'ok'."}]
        agent = _make_real_agent(prompt=prompt)
        agent.run()
        assert len(agent.prompt) == 2
        assert agent.prompt[1]["role"] == "assistant"

    def test_run_with_empty_tools_passes_none(self):
        """An empty tools list should send None to the API (no tool calls requested)."""
        prompt = [{"role": "user", "content": "Say 'hello'."}]
        agent = _make_real_agent(prompt=prompt)
        assert agent.tools == []
        response = agent.run()
        # The API should not return tool_calls when no tools are provided.
        assert response.choices[0].message.tool_calls is None


# ---------------------------------------------------------------------------
# Streaming run()
# ---------------------------------------------------------------------------


class TestRealAPIStream:
    def test_stream_on_token_called(self):
        tokens = []

        def on_token(agent, full_message, delta):
            tokens.append(delta)

        prompt = [{"role": "user", "content": "Say 'hi'."}]
        agent = _make_real_agent(
            prompt=prompt,
            stream=True,
            on_token=on_token,
            on_end=lambda a, m: None,
        )
        agent.run()
        assert len(tokens) > 0

    def test_stream_on_end_receives_full_message(self):
        final = []

        def on_end(agent, full_message):
            final.append(full_message)

        prompt = [{"role": "user", "content": "Say 'done'."}]
        agent = _make_real_agent(
            prompt=prompt,
            stream=True,
            on_token=lambda a, f, d: None,
            on_end=on_end,
        )
        agent.run()
        assert len(final) == 1
        assert len(final[0]) > 0

    def test_stream_full_message_equals_concatenated_tokens(self):
        tokens = []
        final = []

        def on_token(agent, full_message, delta):
            tokens.append(delta)

        def on_end(agent, full_message):
            final.append(full_message)

        prompt = [{"role": "user", "content": "Count: 1 2 3."}]
        agent = _make_real_agent(
            prompt=prompt,
            stream=True,
            on_token=on_token,
            on_end=on_end,
        )
        agent.run()
        assert final[0] == "".join(tokens)

    def test_stream_appends_assistant_message(self):
        prompt = [{"role": "user", "content": "Say 'stream'."}]
        agent = _make_real_agent(
            prompt=prompt,
            stream=True,
            on_token=lambda a, f, d: None,
            on_end=lambda a, m: None,
        )
        agent.run()
        assert len(agent.prompt) == 2
        assert agent.prompt[1]["role"] == "assistant"
        assert len(agent.prompt[1]["content"]) > 0


# ---------------------------------------------------------------------------
# Message helpers with a live agent instance
# ---------------------------------------------------------------------------


class TestRealAPIMessageHelpers:
    def test_append_raw_message_then_run(self):
        agent = _make_real_agent()
        agent.append_raw_message({"role": "user", "content": "Say 'raw'."})
        response = agent.run()
        assert response is not None
        assert agent.prompt[-1]["role"] == "assistant"

    def test_append_str_message_user(self):
        agent = _make_real_agent()
        agent.append_str_message("Say 'str'.")
        assert agent.prompt[-1] == {"role": "user", "content": "Say 'str'."}

    def test_append_str_message_assistant(self):
        agent = _make_real_agent()
        agent.append_str_message("I am an assistant.", is_user=False)
        assert agent.prompt[-1] == {"role": "assistant", "content": "I am an assistant."}

    def test_multi_turn_conversation(self):
        """A two-turn conversation should accumulate messages in the prompt."""
        prompt = [{"role": "user", "content": "Remember the number 42."}]
        agent = _make_real_agent(prompt=prompt)
        agent.run()
        agent.append_str_message("What number did I ask you to remember?")
        agent.run()
        # prompt: user → assistant → user → assistant (4 entries)
        assert len(agent.prompt) == 4
        roles = [m["role"] for m in agent.prompt]
        assert roles == ["user", "assistant", "user", "assistant"]
