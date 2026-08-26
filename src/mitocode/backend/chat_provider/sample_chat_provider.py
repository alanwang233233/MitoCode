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
        default_messages: Optional[List[Dict]]=None,
        settings: Optional[Dict[str, Any]]=None # settings是一个字典，格式为{"key": "value"}
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
        if default_model == "": raise MitoCodeChatProviderConfigError("default_model cannot be empty")
        if not type(self).check_settings_schema(settings):
            raise MitoCodeChatProviderConfigError("settings schema validation failed")
        self.default_model = default_model
        self.default_messages = default_messages
        self.settings = settings
        self.model_list = self.get_model_list()
        if default_model not in self.model_list:
            raise MitoCodeChatProviderModelError(f"Model '{default_model}' not in available models: {self.model_list}")

    @abstractmethod
    def get_model_list(self) -> List[str]: ...

    @abstractmethod
    async def init(self) -> None: ...

    @classmethod
    @abstractmethod
    def get_settings_schema(cls) -> Dict: ...

    @classmethod
    @abstractmethod
    def check_settings_schema(cls, settings) -> bool: ...

    @abstractmethod
    async def chat(
        self,
        user_messages: str = '',
        model: str = '',
        tools: Optional[Dict] = None, # 符合OpenAI Chat Completions API的tools参数格式
        messages: Optional[List[Dict]] = None, # 符合OpenAI Chat Completions API的messages参数格式
    ) -> Dict: ...


SampleChatProvider = _SampleChatProvider