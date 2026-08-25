"""
用于对接AI平台API的接口，其他类需实现这个接口
"""
from abc import abstractmethod, ABC
from typing import List, Dict


class _SampleChatProvider(ABC) :
    @classmethod
    def factory(cls, *args, **kwargs) -> "_SampleChatProvider" :
        return cls(*args, **kwargs)

    def __init__(
        self,
        default_model : str = "",
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
        if default_model == "" : raise Exception
        if not self.check_settings_schema(settings) : raise Exception
        self.model_list = self.get_model_list()
        if default_model not in self.model_list :
            raise Exception
        self.default_model = default_model
        self.default_messages = default_messages
        self.settings = settings

    @staticmethod
    @abstractmethod
    def get_model_list() -> List[str] : pass

    @abstractmethod
    def init(self) -> None : pass

    @classmethod
    @abstractmethod
    def get_settings_schema(cls) -> Dict : pass

    @staticmethod
    @abstractmethod
    def check_settings_schema(settings) -> bool: pass

    @abstractmethod
    async def chat(
        self,
        user_messages : str = '',
        model : str = '',
        tools : dict = None, # 符合OpenAI规范的工具列表
        messages : List[Dict] = None, # 符合OpenAI规范的上下文
        # 开始前运行的回调函数，传入以下参数：
        # on_start(user_messages : str, current_messages : List[Dict])
        # user_messages为用户输入的消息，current_messages为当前的完整上下文
        on_start = lambda _,__ : None,
        # 输出完成时调用的回调函数
        # on_stop(reason : str = 'finish', current_messages)
        # reason有stop, length, content_filter, tool_calls,
        on_stop = lambda _,__ : None
    ) -> Dict : pass