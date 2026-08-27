"""Scripted mock provider replaying a fixed multi-turn tool-call demo.

Used for offline demos and TUI development: each ``chat`` call plays back
the next pre-recorded turn of a bubble-sort coding session.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, AsyncGenerator

from mitocode.backend.chat_provider.sample_chat_provider import _SampleChatProvider


def _make_chunk(
    content: Optional[str] = None,
    tool_calls: Optional[List[Any]] = None,
    finish_reason: Optional[str] = None,
):
    """Create a simulated OpenAI streaming chunk with optional content and tool calls."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _make_tool_call(index: int, call_id: str, name: Optional[str], arguments: str):
    """Create a simulated tool-call object matching the OpenAI streaming format."""
    func = SimpleNamespace(name=name or "", arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=func)


def _finish(reason: str):
    """Return a chunk with only a finish reason to signal end of stream."""
    return _make_chunk(finish_reason=reason)


TURN_0_MARKDOWN = """\
Let me help you create a bubble sort implementation in Python.

## Plan

1. Check the current working directory
2. Write `bubble_sort.py` with the implementation
3. Verify the file was written correctly

First, let me see what's in the current directory:"""

TURN_1_MARKDOWN = """\
Good, the directory is clean. Now let me write the bubble sort implementation.

I'll include an **early exit optimization** that stops the sort if no swaps occur:"""

BUBBLE_SORT_CODE = """\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_arr}")
    sorted_arr = bubble_sort(test_arr)
    print(f"Sorted:   {sorted_arr}")
"""

TURN_2_MARKDOWN = """\
Let me verify the file was written correctly by reading it back:"""

TURN_3_MARKDOWN = """\
The bubble sort implementation is complete. Here's a breakdown:

| Aspect | Detail |
|--------|--------|
| **Time Complexity** | O(n²) worst / average, O(n) best |
| **Space Complexity** | O(1) — in-place |
| **Stability** | Stable |

The optimization uses a `swapped` flag — if no elements are swapped during a full pass,
the array is already sorted and we exit early.

To run it:

```bash
python bubble_sort.py
```
"""

BASH_ARGS = json.dumps({"command": "ls -la"})

WRITE_FILE_ARGS = json.dumps(
    {
        "file_path": "bubble_sort.py",
        "content": BUBBLE_SORT_CODE,
    },
    ensure_ascii=False,
)

READ_FILE_ARGS = json.dumps({"file_path": "bubble_sort.py"})


def _chunk_text(text: str, size: int = 4) -> List[Any]:
    """Split Markdown text into streaming chunks."""
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(_make_chunk(content=text[i : i + size]))
    return chunks


def _chunk_tool_call(
    index: int, call_id: str, name: str, arguments: str, arg_chunk_size: int = 8
) -> List[Any]:
    """Simulate OpenAI streaming tool call: name chunk + argument chunks."""
    chunks = []
    # Tool call header: id + name
    chunks.append(
        _make_chunk(
            tool_calls=[_make_tool_call(index, call_id, name, "")],
        )
    )
    # Streaming arguments
    for i in range(0, len(arguments), arg_chunk_size):
        chunks.append(
            _make_chunk(
                tool_calls=[
                    _make_tool_call(
                        index, call_id, None, arguments[i : i + arg_chunk_size]
                    )
                ],
            )
        )
    return chunks


class ScenarioProvider(_SampleChatProvider):
    """Mock provider that plays back a multi-turn tool-call scenario.

    Each call to :meth:`chat` advances an internal turn counter and returns
    the next pre-recorded stream: markdown text, bash / write_file / read_file
    tool calls with bubble sort code.
    """

    def __init__(
        self,
        default_model: str = "scenario-model",
        default_model_list: Optional[List[str]] = None,
        default_messages: Optional[List[Dict]] = None,
        default_tools: Optional[List[Dict]] = None,
        stream: bool = True,
        settings: Optional[Dict[str, Any]] = None,
        delay: float = 0.03,
    ):
        """Initialize the provider with an inter-chunk streaming delay.

        Args:
            delay: Seconds to sleep between yielded chunks (default 0.03).
        """
        if default_model_list is None:
            default_model_list = ["scenario-model"]
        super().__init__(
            default_model=default_model,
            default_model_list=default_model_list,
            default_messages=default_messages,
            default_tools=default_tools,
            stream=stream,
            settings=settings,
        )
        self.turn = 0
        self._delay = delay

    @classmethod
    async def factory(cls, *args, **kwargs) -> "ScenarioProvider":
        """Create and initialize a provider instance asynchronously."""
        ins = cls(*args, **kwargs)
        await ins.init()
        return ins

    @classmethod
    def get_settings_schema(cls) -> Dict[str, Any]:
        """Return the JSON Schema describing accepted settings."""
        return {
            "type": "object",
            "properties": {
                "delay": {"type": "number", "minimum": 0, "default": 0.03},
            },
            "additionalProperties": True,
        }

    @classmethod
    def check_settings_schema(cls, settings: Any) -> bool:
        """Validate that *settings* is a dict with optional delay."""
        if not isinstance(settings, dict):
            return False
        delay = settings.get("delay", 0.03)
        return isinstance(delay, (int, float)) and delay >= 0

    async def get_model_list(self) -> List[str]:
        """Return the model identifiers accepted by this provider."""
        return ["scenario-model"]

    async def init(self) -> None:
        """No-op initialization hook kept for provider interface parity."""
        delay = self.settings.get("delay", 0.03) if self.settings else 0.03
        self._delay = delay

    async def chat(
        self,
        model: str = "",
        tools: Optional[List[Dict]] = None,
        messages: Optional[List[Dict]] = None,
        stream: bool = True,
    ) -> AsyncGenerator | Dict:
        """Return an async stream playing back the next scripted turn.

        Each invocation advances the internal turn counter: turns 0-2 emit
        Markdown text followed by bash / write_file / read_file tool calls;
        later turns emit the final summary with finish reason ``stop``.
        Chunks are yielded with the configured inter-chunk delay.
        """
        turn = self.turn
        self.turn += 1

        chunks: List[Any] = []

        if turn == 0:
            chunks.extend(_chunk_text(TURN_0_MARKDOWN, size=5))
            chunks.extend(
                _chunk_tool_call(0, "call_bash", "bash", BASH_ARGS, arg_chunk_size=6)
            )
            chunks.append(_finish("tool_calls"))

        elif turn == 1:
            chunks.extend(_chunk_text(TURN_1_MARKDOWN, size=5))
            chunks.extend(
                _chunk_tool_call(
                    0, "call_write", "write_file", WRITE_FILE_ARGS, arg_chunk_size=12
                )
            )
            chunks.append(_finish("tool_calls"))

        elif turn == 2:
            chunks.extend(_chunk_text(TURN_2_MARKDOWN, size=5))
            chunks.extend(
                _chunk_tool_call(
                    0, "call_read", "read_file", READ_FILE_ARGS, arg_chunk_size=6
                )
            )
            chunks.append(_finish("tool_calls"))

        else:
            chunks.extend(_chunk_text(TURN_3_MARKDOWN, size=5))
            chunks.append(_finish("stop"))

        async def _stream():
            for c in chunks:
                yield c
                await asyncio.sleep(self._delay)

        if stream:
            return _stream()

        full_content = ""
        for c in chunks:
            if c.choices[0].delta.content:
                full_content += c.choices[0].delta.content
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "model": model or self.default_model,
        }
