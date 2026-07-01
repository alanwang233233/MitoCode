import asyncio
import json
from typing import Any, Dict, List, Optional


def _make_chunk(
    content: Optional[str] = None,
    tool_calls: Optional[List[Any]] = None,
    finish_reason: Optional[str] = None,
):
    class _D:
        pass

    delta = _D()
    delta.content = content
    delta.tool_calls = tool_calls

    choice = _D()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = _D()
    chunk.choices = [choice]
    return chunk


def _make_tool_call(index: int, call_id: str, name: Optional[str], arguments: str):
    tc = type("_TC", (), {})()
    tc.index = index
    tc.id = call_id
    tc.function = type("_F", (), {})()
    tc.function.name = name or ""
    tc.function.arguments = arguments
    return tc


def _finish(reason: str):
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

WRITE_FILE_ARGS = json.dumps({
    "file_path": "bubble_sort.py",
    "content": BUBBLE_SORT_CODE,
}, ensure_ascii=False)

READ_FILE_ARGS = json.dumps({"file_path": "bubble_sort.py"})


def _chunk_text(text: str, size: int = 4) -> List[Any]:
    """Split markdown text into streaming chunks."""
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


class ScenarioProvider:
    """Mock provider that plays back a multi-turn tool-call scenario.

    Each call to :meth:`chat` advances an internal turn counter and returns
    the next pre-recorded stream: markdown text, bash, write_file, read_file
    tool calls with bubble sort code.
    """

    def __init__(self, delay: float = 0.03):
        self._turn = 0
        self._delay = delay

    @staticmethod
    def get_model_list():
        return ["scenario-model"]

    @classmethod
    def get_settings_schema(cls):
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    @staticmethod
    def check_settings_schema(settings):
        return isinstance(settings, dict)

    def init(self):
        pass

    async def chat(
        self,
        tools: Optional[list] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        turn = self._turn
        self._turn += 1

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

        return _stream()
