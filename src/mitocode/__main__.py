"""WIP"""

import argparse
import asyncio
from pyexpat.errors import messages

from httpx import stream

from mitocode.backend.agent.agent_orchestrator import AgentOrchestrator
from mitocode.backend.chat_provider.scenario_provider import ScenarioProvider
from mitocode.backend.chat_provider.chat_provider_openai import ChatProviderOpenAI
from mitocode.backend.tool.tool_manager import ToolManager
from mitocode.tui.ui_main import VibeFrontend


def _register_tools(tool_manager: ToolManager):
    """Register built-in demo tools (bash, read_file, write_file) on the manager."""

    @tool_manager.tool(
        human_doc="Run a shell command",
        agent_doc="Execute a bash command and return its output",
        display_action="bash",
    )
    async def bash(command: str) -> str:
        return f"$ {command}\n[bash] command executed successfully"

    @tool_manager.tool(
        human_doc="Read file content",
        agent_doc="Read the full content of a file at the given path",
        display_action="file.read",
        display_param="file_path",
    )
    async def read_file(file_path: str) -> str:
        return (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        swapped = False\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "                swapped = True\n"
            "        if not swapped:\n"
            "            break\n"
            "    return arr\n"
        )

    @tool_manager.tool(
        human_doc="Write content to a file",
        agent_doc="Create or overwrite a file with the given content",
        display_action="file.write",
        display_param="file_path",
    )
    async def write_file(file_path: str, content: str) -> str:
        return f"Successfully wrote {len(content)} bytes to {file_path}"


def main():
    """Parse CLI arguments and launch the MitoCode TUI application."""
    parser = argparse.ArgumentParser(description="MitoCode")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.03,
        help="Delay between chunks in seconds (default: 0.03)",
    )
    args = parser.parse_args()

    tool_manager = ToolManager()
    _register_tools(tool_manager)

    provider = asyncio.run(
        ChatProviderOpenAI.factory(
            default_model="nvidia/nemotron-3.5-lightning:free",
            default_model_list=None,
            default_messages=None,
            default_tools=None,
            stream=True,
            settings={
                "api_key": "sk-别看了这次我不会再把key提交到仓库了！",
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": 60,
            },
        )
    )

    # print(asyncio.run(provider.chat(
    #     model ="nvidia/nemotron-3.5-lightning:free",
    #     tools =None,
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Hello!"}
    #     ],
    #     stream = False
    # )))

    orchestrator = AgentOrchestrator(
        tool_manager=tool_manager, default_provider=provider
    )
    asyncio.run(
        orchestrator.create_agent(
            name="Main",
            description="Chief orchestrator",
        )
    )

    app = VibeFrontend(orchestrator=orchestrator)
    app.run()


if __name__ == "__main__":
    main()
