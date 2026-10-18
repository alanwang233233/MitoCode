"""Abstract base class for chat providers.

This module defines the interface that all chat provider implementations must follow.
Subclasses implement provider-specific logic for connecting to LLM APIs.
"""

from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any, AsyncGenerator

from mitocode.core.exception.mitocode_exception import (
    MitoCodeChatProviderConfigError,
)


class _SampleChatProvider(ABC):
    """Abstract base class for chat providers.

    Defines the common lifecycle and contract shared by all provider
    implementations: configuration validation, model listing, initialization,
    and chat completion.
    """

    @classmethod
    async def factory(cls, *args, **kwargs) -> "_SampleChatProvider":
        """Create and initialize a provider instance asynchronously.

        Args:
            *args: Positional arguments passed to the constructor.
            **kwargs: Keyword arguments passed to the constructor.

        Returns:
            A fully initialized provider instance.
        """
        ins = cls(*args, **kwargs)
        await ins.init()
        return ins

    def __init__(
        self,
        default_model: str = "",
        default_model_list: Optional[List[str]] = None,
        default_messages: Optional[List[Dict]] = None,
        default_tools: Optional[List[Dict]] = None,
        stream: bool = True,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the provider with model, default messages, and settings.

        Args:
            default_model: Name of the default model; must not be empty.
            default_messages: Initial conversation messages in OpenAI Chat
                Completions format. Defaults to a single system message.
            settings: Provider-specific configuration dict validated against
                the schema returned by ``get_settings_schema``.

        Raises:
            MitoCodeChatProviderConfigError: If the model name is empty or the
                settings fail schema validation.
            MitoCodeChatProviderModelError: If the default model is not listed
                among available models.
        """
        if settings is None:
            settings = {}
        if default_messages is None:
            default_messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        if default_model == "":
            raise MitoCodeChatProviderConfigError("default_model cannot be empty")
        if not type(self).check_settings_schema(settings):
            raise MitoCodeChatProviderConfigError("settings schema validation failed")
        self.stream = stream
        self.default_tools = default_tools
        self.default_model = default_model
        self.model_list = default_model_list
        self.default_messages = default_messages
        self.settings = settings

    @abstractmethod
    async def get_model_list(self) -> List[str]:
        """Return the list of model names supported by this provider."""

    async def set_model_list(self, model_list: List[str]) -> None:
        """Set the list of model names supported by this provider.

        Args:
            model_list: List of model names to set.
        """
        self.model_list = model_list

    @abstractmethod
    async def init(self) -> None:
        """Perform asynchronous initialization (e.g. connect to the API)."""

    @classmethod
    @abstractmethod
    def get_settings_schema(cls) -> Dict:
        """Return the schema describing the expected structure of ``settings``."""

    @classmethod
    @abstractmethod
    def check_settings_schema(cls, settings) -> bool:
        """Validate ``settings`` against the provider's schema.

        Args:
            settings: The settings dict to validate.

        Returns:
            True if the settings conform to the schema, False otherwise.
        """

    @abstractmethod
    async def chat(
        self,
        model: str = "",
        tools: Optional[List[Dict]] = None,
        messages: Optional[List[Dict]] = None,
        stream: bool = True,
    ) -> AsyncGenerator | Dict:
        """Send messages to the model and return the completion result.

        Args:
            model: Model name to use; falls back to ``default_model`` if empty.
            tools: Tool definitions in OpenAI Chat Completions API format.
            messages: Full conversation history in OpenAI Chat Completions
                API format.
            stream: Whether to return a streaming response (True) or a single
                completion (False).

        Returns:
            The completion response as a dict or an AsyncGenerator in OpenAI Chat Completions format.
        """


SampleChatProvider = _SampleChatProvider
