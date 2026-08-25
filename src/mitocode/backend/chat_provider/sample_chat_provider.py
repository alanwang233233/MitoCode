"""Abstract base class for chat providers.

This module defines the interface that all chat provider implementations must follow.
Subclasses implement provider-specific logic for connecting to LLM APIs.
"""
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Callable, Any

from mitocode.backend.exception.mitocode_exception import (
    MitoCodeChatProviderConfigError,
    MitoCodeChatProviderModelError,
)


class _SampleChatProvider(ABC):
    @classmethod
    async def factory(cls, *args, **kwargs) -> "_SampleChatProvider":
        ins = cls(*args, **kwargs)
        await ins.init()
        return ins

    def __init__(
        self,
        default_model: str = "",
        default_messages=None,
        settings=None
    ):
        if settings is None:
            settings = []
        if default_messages is None:
            default_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]
        if default_model == "":
            raise MitoCodeChatProviderConfigError("default_model cannot be empty") from None
        if not type(self).check_settings_schema(settings):
            raise MitoCodeChatProviderConfigError("settings schema validation failed") from None
        self.model_list = self.get_model_list()
        if default_model not in self.model_list:
            raise MitoCodeChatProviderModelError(f"Model '{default_model}' not in available models: {self.model_list}") from None
        self.default_model = default_model
        self.default_messages = default_messages
        self.settings = settings

    @abstractmethod
    def get_model_list(self) -> List[str]: pass

    @abstractmethod
    async def init(self) -> None: pass

    @classmethod
    @abstractmethod
    def get_settings_schema(cls) -> Dict: pass

    @classmethod
    @abstractmethod
    def check_settings_schema(cls, settings) -> bool: pass

    @abstractmethod
    async def chat(
        self,
        user_messages: str = '',
        model: str = '',
        tools: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
        on_start: Callable[[str, List[Dict]], None] = lambda _, __: None,
        on_stop: Callable[[str, List[Dict]], None] = lambda _, __: None,
    ) -> Dict: pass


SampleChatProvider = _SampleChatProvider