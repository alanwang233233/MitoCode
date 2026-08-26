"""旧实现，与现有ABC不符，已弃用，请勿参考！"""
"""OpenAI Chat Completions implementation of the chat provider interface."""

from typing import Any, Dict, List, Optional

from .sample_chat_provider import _SampleChatProvider

from openai import AsyncOpenAI


class ChatProviderOpenAI(_SampleChatProvider):
    """Provider subclass backed by OpenAI Chat Completions streaming API."""
    @staticmethod
    def _merge_messages(
            default_messages: List[Dict[str, Any]],
            messages: Optional[List[Dict[str, Any]]],
            user_messages: str,
    ) -> List[Dict[str, Any]]:
        """Merge default, additional, and user messages into a single list.

        The resulting list preserves ordering: default messages first,
        then any additional *messages*, and finally the user's text
        appended as a ``"user"`` role entry.

        Args:
            default_messages: Base conversation (typically system messages).
            messages: Optional extra messages to append after defaults.
            user_messages: The user's latest text input.

        Returns:
            A new list of message dicts ready to send to the LLM.
        """
        current_messages: List[Dict[str, Any]] = list(default_messages)
        if messages:
            current_messages.extend(messages)
        if user_messages:
            current_messages.append({"role": "user", "content": user_messages})
        return current_messages

    @staticmethod
    async def _wrap_stream(
            stream,
            current_messages: List[Dict[str, Any]],
            on_token,
            on_stop,
    ):
        """Iterate over *stream* chunks and fire callbacks.

        For each chunk that contains a text delta, *on_token* is called.
        When the stream finishes (or errors), *on_stop* is called with
        the final finish reason.

        Args:
            stream: An async iterable of chunk objects (e.g. OpenAI
                streaming response chunks).
            current_messages: The full message history; passed through
                to callbacks for context.
            on_token: Callable ``(content: str, messages: List[Dict]) -> None``
                invoked for every text delta.
            on_stop: Callable ``(finish_reason: str, messages: List[Dict]) -> None``
                invoked when the stream ends or on error.

        Yields:
            Each chunk from the underlying *stream*, allowing callers
            to consume the raw data as well.
        """
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
        except Exception:
            on_stop("network-error", current_messages)
            raise

    def get_model_list() -> List[str]:
        """Return a list of supported OpenAI model identifiers.

        Returns a curated set of GPT models. Callers can override this
        method in a custom subclass to provide a different set.
        """
        # TODO: Consider fetching the model list dynamically from OpenAI's API.
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5-mini",
            "gpt-5",
        ]

    @classmethod
    def get_settings_schema(cls) -> Dict[str, Any]:
        """Return a JSON Schema describing the required provider settings.

        The schema mandates ``api_key`` (string) as a required field,
        with optional ``base_url`` and ``timeout`` properties.
        """
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
        """Validate that *settings* contains a non-empty ``api_key`` string.

        Returns ``True`` only when *settings* is a dict and its
        ``api_key`` value is a non-blank string; ``False`` otherwise.
        """
        if not isinstance(settings, dict):
            return False
        api_key = settings.get("api_key", "")
        return isinstance(api_key, str) and len(api_key.strip()) > 0

    def init(self) -> None:
        """Initialise the ``AsyncOpenAI`` client from the current settings.

        Reads ``api_key``, ``base_url`` and ``timeout`` from
        ``self.settings`` and stores the resulting client on
        ``self.client``.
        """
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
        """Send a chat request to OpenAI and return a streaming response.

        Merges *messages* with the provider defaults and invokes
        ``on_start`` before the request. Tokens are delivered via
        ``on_token`` callbacks; ``on_stop`` is called when the stream
        finishes.  Raises ``ValueError`` if *model* is unknown.
        """
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
