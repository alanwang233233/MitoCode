from openai import OpenAI

class SingleAgent(object):
    # 基础定义
    system_prompt = ""
    model = ""
    messages = []
    tools = []
    stream = False
    client: OpenAI = None

    #两个回调函数，仅当stream=true时被调用，分别对应流式返回一个token和流式响应结束
    on_token = None
    on_end = None

    def __init__(self, system_prompt, api_base, api_key, tools=None, stream=False, on_token=None, on_end=None):
        self.system_prompt = system_prompt
        self.tools = tools
        self.stream = stream
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.on_token = on_token
        self.on_end = on_end

    def append_raw_message(self, message):
        self.messages.append(message)

    def append_str_message(self, message: str, is_user: bool = True):
        self.messages.append({"role": "user" if is_user else "assistant", "message": message})

    def act(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            stream=self.stream
        )
        if self.stream:
            full_message = ""
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    delta_message = delta["content"]
                    full_message += delta_message
                    self.on_token(self, full_message, delta_message)
            self.on_end(self, full_message)
            self.append_str_message(full_message)
        else:
            self.append_str_message(str(response.choices[0].message))
        return response
