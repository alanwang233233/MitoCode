"""Tests for ChatProviderOpenAI's reasoning_effort request parameter."""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.chat_provider.chat_provider_openai import (
    ChatProviderOpenAI,
    normalize_reasoning_effort,
)


class _FakeCompletions:
    """Capture every ``create(**req)`` payload; never touches the network.

    Attributes:
        requests: One entry per ``create`` call, in call order.
    """

    def __init__(self) -> None:
        """Initialise an empty request log."""
        self.requests: List[Dict[str, Any]] = []

    async def create(self, **req: Any) -> Any:
        """Record one request and return a dumpable completion stand-in.

        Args:
            **req: The exact keyword arguments ``chat()`` assembled.

        Returns:
            An object exposing ``model_dump()`` for the non-streaming path.
        """
        self.requests.append(req)
        return SimpleNamespace(model_dump=lambda: {"id": "x"})


def _provider(effort: Any) -> ChatProviderOpenAI:
    """Build an offline provider with a fake client and a preset effort.

    Args:
        effort: Value assigned to ``reasoning_effort`` before the call.

    Returns:
        A provider whose client records requests instead of sending them.
    """
    provider = ChatProviderOpenAI(
        default_model="m", default_model_list=["m"], settings={"api_key": "k"}
    )
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions())
    )
    provider.reasoning_effort = effort
    return provider


def test_normalize_strips_lowercases_and_rejects_blank():
    """Valid text is stripped and lowercased; blank and non-text become None."""
    assert normalize_reasoning_effort(" HIGH ") == "high"
    assert normalize_reasoning_effort("Low") == "low"
    assert normalize_reasoning_effort("") is None
    assert normalize_reasoning_effort("   ") is None
    assert normalize_reasoning_effort(None) is None
    assert normalize_reasoning_effort(7) is None


@pytest.mark.asyncio
async def test_chat_includes_reasoning_effort_when_set():
    """A configured effort reaches the request payload verbatim."""
    provider = _provider("high")
    await provider.chat(model="m", messages=[], stream=False)
    assert provider.client.chat.completions.requests[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_chat_omits_reasoning_effort_when_unset():
    """Without configuration the request carries no reasoning_effort key."""
    provider = _provider(None)
    await provider.chat(model="m", messages=[], stream=False)
    assert "reasoning_effort" not in provider.client.chat.completions.requests[0]


def test_settings_schema_documents_reasoning_effort():
    """The settings schema advertises the optional reasoning_effort string."""
    schema = ChatProviderOpenAI.get_settings_schema()
    assert schema["properties"]["reasoning_effort"]["type"] == "string"
