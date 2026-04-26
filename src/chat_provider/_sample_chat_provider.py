"""
用于对接AI平台API的接口，其他类需实现这个接口
"""
from abc import abstractmethod, ABC
from typing import List, Dict


class _SampleChatProvider(ABC) :
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
        self.init()

    @staticmethod
    @abstractmethod
    def get_model_list() -> List[str] : pass
        
    @abstractmethod
    async def init(self) -> None : pass

    @classmethod
    @abstractmethod
    async def get_settings_schema(cls) -> Dict : pass
    
    @staticmethod
    @abstractmethod
    def check_settings_schema(settings) -> bool: pass
    
    @abstractmethod
    async def chat(
        self,
        user_messages : str = '',
        model : str = '',
        tools : dict = {}, # 符合OpenAI规范的工具列表
        messages : List[Dict] = [
            {
                "role" : "system",
                "content" : "You are a helpful assistant."
            }
        ],
        # 开始前运行的回调函数，传入以下参数：
        # on_start(user_messages : str, current_messages : List[Dict])
        # current_messages为当前的完整上下文
        on_start = lambda _,__ : None,
        # 生成100 token时调用的回调参数
        # 开始前运行的回调函数，传入以下参数：
        # on_start(user_messages : str, current_messages : List[Dict])
        # current_messages为当前的完整上下文
        on_token = lambda _,__ : None,
        tool_call = lambda _,__ : None, # 工具调用时调用的回调函数
                 # tool_call(tool_name, params)
                 # 返回值传给AI
        on_tool_call = lambda _,__ : None, # 工具调用时调用的回调函数
                    # on_tool_call(tool_name, params)
                    # 返回值不用管
        on_stop = lambda _,__ : None, # 输出完成时调用的回调函数
                # on_stop(reason : str = 'finish', current_messages)
                # reason有finish, network-error, api-error, unknown四种形式
    ) -> Dict : pass