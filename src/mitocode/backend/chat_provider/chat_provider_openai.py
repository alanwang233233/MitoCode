"""OpenAI Chat Completions implementation of the chat provider interface."""

from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from mitocode.backend.exception.mitocode_exception import (
    MitoCodeChatProviderModelError,
)

from .sample_chat_provider import _SampleChatProvider


class ChatProviderOpenAI(_SampleChatProvider):
    """Provider subclass backed by the OpenAI Chat Completions API.

    Conforms to the current ``_SampleChatProvider`` contract: schema-driven
    settings validation performed by the base class, asynchronous client
    initialization via :meth:`init`, and a :meth:`chat` that either streams
    chunks through an ``AsyncGenerator`` or returns the full completion as
    a dict in OpenAI Chat Completions format, depending on the ``stream``
    flag.
    """

    def get_model_list(self) -> List[str]:
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

    @classmethod
    def check_settings_schema(cls, settings: Any) -> bool:
        """Validate that *settings* contains a non-empty ``api_key`` string.

        Returns ``True`` only when *settings* is a dict and its
        ``api_key`` value is a non-blank string; ``False`` otherwise.
        """
        if not isinstance(settings, dict):
            return False
        api_key = settings.get("api_key", "")
        return isinstance(api_key, str) and len(api_key.strip()) > 0

    async def init(self) -> None:
        """Initialise the ``AsyncOpenAI`` client from the current settings.

        Reads ``api_key``, ``base_url`` and ``timeout`` from
        ``self.settings`` and stores the resulting client on
        ``self.client``.
        """
        settings: Dict[str, Any] = self.settings or {}
        self.client = AsyncOpenAI(
            api_key=settings.get("api_key", ""),
            base_url=settings.get("base_url", None),
            timeout=settings.get("timeout", 60),
        )

    @staticmethod
    async def _wrap_stream(stream: AsyncIterator[Any]) -> AsyncGenerator:
        """Yield each chunk from an OpenAI streaming response unchanged.

        Args:
            stream: The async iterator returned by the OpenAI SDK when
                the request is made with ``stream=True``.

        Yields:
            Each chunk object from *stream*, preserving the raw SDK types.
        """
        async for chunk in stream:
            yield chunk

    async def chat(
        self,
        model: str = "",
        tools: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
        stream: bool = True,
    ) -> AsyncGenerator | Dict:
        """Send a chat request to OpenAI and return the completion result.

        Merges *messages* with the provider defaults, falls back to
        ``default_tools`` when *tools* is omitted, and issues a Chat
        Completions request. Raises ``MitoCodeChatProviderModelError`` if
        *model* is unknown.

        Args:
            model: Model name to use; falls back to ``default_model`` if empty.
            tools: Tool definitions in OpenAI Chat Completions API format.
            messages: Full conversation history in OpenAI Chat Completions
                API format.
            stream: Whether to return a streaming response (True) or a single
                completion (False).

        Returns:
            When *stream* is true, an ``AsyncGenerator`` yielding chunks in
            OpenAI streaming format; otherwise the full completion response
            as a dict in OpenAI Chat Completions format.
        """
        current_model = model or self.default_model
        if current_model not in self.model_list:
            raise MitoCodeChatProviderModelError(
                f"Model '{current_model}' not in available models: {self.model_list}"
            )

        current_messages: List[Dict[str, Any]] = messages

        req: Dict[str, Any] = {
            "model": current_model,
            "messages": current_messages,
        }
        effective_tools = tools if tools is not None else self.default_tools
        if effective_tools:
            req["tools"] = effective_tools

        if stream:
            req["stream"] = True
            api_stream = await self.client.chat.completions.create(**req)
            return self._wrap_stream(api_stream)

        completion = await self.client.chat.completions.create(**req)
        return completion.model_dump()
