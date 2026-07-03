from typing import Any, Dict, List, Optional

from ._sample_chat_provider import SampleChatProvider

from openai import AsyncOpenAI


class ChatProviderOpenAI(SampleChatProvider):
    """Provider subclass backed by OpenAI Chat Completions streaming API."""

    @staticmethod
    def get_model_list() -> List[str]:
        # Common defaults; callers can override this in a custom subclass.
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5-mini",
            "gpt-5",
        ]

    @classmethod
    def get_settings_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "api_key": {"type": "string"},
                "base_url": {"type": "string"},
                "timeout": {"type": "number", "minimum": 1},
            },
            "required": ["api_key"],
            "additionalProperties": True,
        }

    @staticmethod
    def check_settings_schema(settings) -> bool:
        if not isinstance(settings, dict):
            return False
        api_key = settings.get("api_key", "")
        return isinstance(api_key, str) and len(api_key.strip()) > 0

    def init(self) -> None:
        self.client = AsyncOpenAI(
            api_key=self.settings.get("api_key", ""),
            base_url=self.settings.get("base_url", None),
        )
        timeout = self.settings.get("timeout", None)
        self.timeout = timeout if timeout is not None else 60

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
        # TODO: Add tool call callback function

        current_model = model or self.default_model
        if current_model not in self.model_list:
            raise ValueError(f"Unknown model: {current_model}")

        req_messages = messages or []
        current_messages = self._merge_messages(
            self.default_messages, req_messages, user_messages
        )
        on_start(user_messages, current_messages)

        req = {
            "model": current_model,
            "messages": current_messages,
            "stream": True,
            "timeout": self.timeout,
        }
        if tools:
            req["tools"] = tools

        stream = await self.client.chat.completions.create(**req)
        return self._wrap_stream(stream, current_messages, on_token, on_stop)
