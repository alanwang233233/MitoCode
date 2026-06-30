"""Default mock chat provider that can also serve as a provider base class."""

from typing import List, Dict, Any, Optional
import asyncio


class _MockFunction:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class _MockToolCallDelta:
    def __init__(self, index: int, tool_call_id: str, name: str, arguments: str):
        self.index = index
        self.id = tool_call_id
        self.function = _MockFunction(name=name, arguments=arguments)


class _MockDelta:
    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[_MockToolCallDelta]] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls


class _MockChoice:
    def __init__(self, delta: _MockDelta, finish_reason: Optional[str] = None):
        self.delta = delta
        self.finish_reason = finish_reason


class _MockChunk:
    def __init__(self, delta: _MockDelta, finish_reason: Optional[str] = None):
        self.choices = [_MockChoice(delta=delta, finish_reason=finish_reason)]


class _SampleChatProvider:
    """Default provider: offline mock server, overridable for real API backends."""

    def __init__(
        self,
        default_model: str = "mock-model",
        default_messages: Optional[List[Dict[str, Any]]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        if default_messages is None:
            default_messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        if settings is None:
            settings = {}

        self.model_list = self.get_model_list()
        if default_model not in self.model_list:
            raise ValueError(f"Unknown model: {default_model}")
        if not self.check_settings_schema(settings):
            raise ValueError("Invalid settings schema")

        self.default_model: str = default_model
        self.default_messages: List[Dict[str, Any]] = default_messages
        self.settings: Dict[str, Any] = settings
        self.init()

    @staticmethod
    def get_model_list() -> List[str]:
        return ["mock-model"]

    def init(self) -> None:
        return None

    @classmethod
    def get_settings_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mock_reply_prefix": {"type": "string"},
                "chunk_size": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        }

    @staticmethod
    def check_settings_schema(settings) -> bool:
        return isinstance(settings, dict)

    @staticmethod
    def _merge_messages(
        default_messages: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, Any]]],
        user_messages: str,
    ) -> List[Dict[str, Any]]:
        current_messages: List[Dict[str, Any]] = list(default_messages)
        if messages:
            current_messages.extend(messages)
        if user_messages:
            current_messages.append({"role": "user", "content": user_messages})
        return current_messages

    async def _wrap_stream(
        self,
        stream,
        current_messages: List[Dict[str, Any]],
        on_token,
        on_stop,
    ):
        finish_reason = "unknown"
        try:
            async for chunk in stream:
                if getattr(chunk, "choices", None):
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None):
                        on_token(delta.content, current_messages)
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                yield chunk
            on_stop(finish_reason, current_messages)
        except Exception as e:
            on_stop("network-error", current_messages)
            raise

    async def chat(
        self,
        user_messages: str = "",
        model: str = "",
        tools: Optional[list] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        on_start=lambda _, __: None,
        on_token=lambda _, __: None,
        tool_call=lambda _, __: None,
        on_tool_call=lambda _, __: None,
        on_stop=lambda _, __: None,
    ):
        # TODO: Add tool call mock
        del tools, tool_call, on_tool_call

        current_model = model or self.default_model
        if current_model not in self.model_list:
            raise ValueError(f"Unknown model: {current_model}")

        current_messages = self._merge_messages(
            self.default_messages, messages, user_messages
        )
        on_start(user_messages, current_messages)

        last_user = ""
        for item in reversed(current_messages):
            if item.get("role") == "user":
                last_user = str(item.get("content", ""))
                break

        reply_prefix = str(self.settings.get("mock_reply_prefix", "[mock]"))
        full_text = f"{reply_prefix} {last_user}".strip()
        chunk_size = int(self.settings.get("chunk_size", 16))
        if chunk_size <= 0:
            chunk_size = 16

        async def _mock_stream():
            for i in range(0, len(full_text), chunk_size):
                delta_text = full_text[i : i + chunk_size]
                on_token(delta_text, current_messages)
                yield _MockChunk(
                    delta=_MockDelta(content=delta_text), finish_reason=None
                )
                await asyncio.sleep(0)
            yield _MockChunk(delta=_MockDelta(content=None), finish_reason="stop")

        async def _result_stream():
            async for chunk in _mock_stream():
                yield chunk
            on_stop("stop", current_messages)

        return _result_stream()
