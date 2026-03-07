from openai import OpenAI
import loguru

type Prompt = list[dict]


class SingleAgent(object):
    def __init__(self, prompt: Prompt, **settings):
        base_url = settings.get("base_url", "https://api.openai.com/v1")
        api_key = settings.get("api_key", "")
        model = settings.get("model", "gpt-5")
        tools = settings.get("tools", [])
        stream = settings.get("stream", False)
        on_token = settings.get("on_token", lambda agent, full_message, delta_message: None)
        on_end = settings.get("on_end", lambda agent, full_message: None)
        loguru.logger.debug("Initializing Agent with base_url: {}, model: {}, tools: {}, stream: {}".format(base_url, model, tools, stream))
        self.prompt = prompt
        self.model = model
        self.tools = tools
        self.stream = stream
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.on_token = on_token
        self.on_end = on_end

    def append_raw_message(self, message):
        self.prompt.append(message)

    def append_str_message(self, message: str, is_user: bool = True):
        self.prompt.append({"role": "user" if is_user else "assistant", "message": message})

    def run(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.prompt,
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
