import _single_agent
import mcpmanager


class Agent(_single_agent.SingleAgent):
    def __init__(self, prompt: list[dict], globalmcpmanager : mcpmanager.MCPManager, **settings):
        tools = globalmcpmanager.get_tools()
        single_settings = {
            "base_url": settings.get("base_url", "https://api.openai.com/v1"),
            "api_key": settings.get("api_key", ""),
            "model": settings.get("model", "gpt-5"),
            "stream": settings.get("stream", False),
            "on_token": settings.get("on_token", lambda agent, full_message, delta_message: None),
            "on_end": settings.get("on_end", lambda agent, full_message:None),
            "tools": tools
        }
        super().__init__(prompt, **single_settings)