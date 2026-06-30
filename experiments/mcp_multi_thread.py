import subprocess
import sys
import os
import asyncio
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from openai import OpenAI

# Configuration
OPENAI_API_KEY = ""

# 增加超时时间（从30秒改为120秒）
COMMAND_TIMEOUT = 120


def ensure_dependencies():
    print("Checking and installing dependencies...")
    try:
        # Check Node.js and npm
        subprocess.run(
            ["npm", "--version"], check=True, capture_output=True, timeout=10
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    try:
        # Check uv
        subprocess.run(["uv", "--version"], check=True, capture_output=True, timeout=10)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        print("Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv", "mcp", "openai"],
            check=True,
            timeout=300,
        )


def build_server_params():
    # 设置环境变量，解决编码和 Git 分页器问题
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["GIT_PAGER"] = "cat"  # 禁用 Git 分页器，防止等待用户输入
    env["PAGER"] = "cat"

    server_params_list = [
        # Git server parameters
        StdioServerParameters(
            command=sys.executable,
            args=["-m", "uv", "tool", "run", "-q", "mcp-server-git", "-r", os.getcwd()],
            env=env,
        ),
        # Terminal controller MCP server parameters
        StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "uv",
                "tool",
                "run",
                "-q",
                "--from",
                "terminal-controller",
                "terminal_controller",
            ],
            env=env,
        ),
    ]

    return server_params_list


class MCPSessionManager:
    """Manages MCP server connections in a dedicated background thread.

    Since the MCP library uses async I/O internally, a private asyncio event
    loop is kept alive on a daemon thread.  All async operations are submitted
    to that loop via asyncio.run_coroutine_threadsafe, which returns a
    concurrent.futures.Future that the calling thread can wait on.  This lets
    the rest of the program stay fully synchronous / thread-based.
    """

    def __init__(self, server_params_list: list[StdioServerParameters]):
        self.server_params_list = server_params_list
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-event-loop"
        )
        self._stack = AsyncExitStack()
        self.sessions: list[ClientSession] = []
        self.all_tools: list = []
        self.tool_to_session_map: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro, timeout=None):
        """Submit a coroutine to the background loop and block until done."""
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _connect_all(self):
        for params in self.server_params_list:
            print(f"Connecting to server: {params.command} {' '.join(params.args)}")
            transport_manager = stdio_client(params)
            read_stream, write_stream = await self._stack.enter_async_context(
                transport_manager
            )
            session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self.sessions.append(session)
            print(
                f"Connected to {params.command} {params.args[1] if len(params.args) > 1 else ''} MCP Server."
            )

        for session in self.sessions:
            response = await session.list_tools()
            for tool in response.tools:
                self.all_tools.append(tool)
                self.tool_to_session_map[tool.name] = session
                print(f"Loaded tool: {tool.name}")

    async def _call_tool_async(self, name: str, args: dict):
        session = self.tool_to_session_map[name]
        return await session.call_tool(name, arguments=args)

    async def _cleanup_async(self):
        await self._stack.aclose()

    # ------------------------------------------------------------------
    # Public interface (synchronous)
    # ------------------------------------------------------------------

    def start(self):
        """Start the background event loop and connect to all MCP servers."""
        self._loop_thread.start()
        self._run_coro(self._connect_all())

    def call_tool(self, name: str, args: dict, timeout=COMMAND_TIMEOUT):
        """Call an MCP tool synchronously, with an optional timeout (seconds)."""
        return self._run_coro(self._call_tool_async(name, args), timeout=timeout)

    def cleanup(self):
        """Disconnect from all servers and stop the background event loop."""
        if self._loop.is_running():
            self._run_coro(self._cleanup_async())
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()


def call_tool_in_thread(
    manager: MCPSessionManager,
    tool_name: str,
    tool_args: dict,
    result_holder: list,
    error_holder: list,
):
    """Worker function executed in a ThreadPoolExecutor thread to call an MCP tool."""
    try:
        result = manager.call_tool(tool_name, tool_args, timeout=COMMAND_TIMEOUT)
        result_holder.append(result)
    except TimeoutError as exc:
        error_holder.append(exc)
    except Exception as exc:
        error_holder.append(exc)


def main():
    if not OPENAI_API_KEY:
        print(
            "WARNING: OPENAI_API_KEY environment variable is not set. The OpenAI API call will fail unless you provide one."
        )

    ensure_dependencies()

    print("\nStarting MCP servers...")
    server_params_list = build_server_params()

    manager = MCPSessionManager(server_params_list)
    manager.start()

    try:
        # Prepare tools for OpenAI
        openai_tools = []
        for t in manager.all_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
            )

        print("\nAll MCP servers initialized and tools loaded.")
        print(f"Total tools available to AI: {len(openai_tools)}")

        if not OPENAI_API_KEY:
            return

        # Synchronous OpenAI client — no async needed
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://app.nextchat.club/api/siliconflow/v1",
        )
        messages = [{"role": "user", "content": "run winver"}]

        print("\nSending prompt to OpenAI:")
        print(messages[0]["content"])

        # Use a thread pool for parallel tool calls when the model requests multiple at once
        with ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="tool-worker"
        ) as executor:
            while True:
                print("\nWaiting for OpenAI model response...")
                response = client.chat.completions.create(
                    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    timeout=COMMAND_TIMEOUT,  # 添加 API 请求超时
                )

                response_message = response.choices[0].message
                messages.append(response_message)

                if response_message.tool_calls:
                    # Dispatch all requested tool calls concurrently via the thread pool
                    futures = {}
                    result_holders = {}
                    error_holders = {}
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        print(
                            f"\nModel wants to call tool '{tool_name}' with args {tool_args}"
                        )

                        if tool_name in manager.tool_to_session_map:
                            result_holders[tool_call.id] = []
                            error_holders[tool_call.id] = []
                            futures[tool_call.id] = executor.submit(
                                call_tool_in_thread,
                                manager,
                                tool_name,
                                tool_args,
                                result_holders[tool_call.id],
                                error_holders[tool_call.id],
                            )
                        else:
                            print(f"Unknown tool requested: {tool_name}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": f"Error: unknown tool '{tool_name}'",
                                }
                            )

                    # Collect results in original order
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        if tool_call.id not in futures:
                            continue
                        # Wait for the worker to finish. Add a small grace period beyond the
                        # internal timeout so we never block here indefinitely.
                        futures[tool_call.id].result(timeout=COMMAND_TIMEOUT + 5)

                        if error_holders[tool_call.id]:
                            exc = error_holders[tool_call.id][0]
                            if isinstance(exc, TimeoutError):
                                print(
                                    f"Error: Tool '{tool_name}' execution timed out after {COMMAND_TIMEOUT} seconds"
                                )
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_name,
                                        "content": f"Error: Command timed out after {COMMAND_TIMEOUT} seconds. Try breaking the command into smaller steps.",
                                    }
                                )
                            else:
                                print(f"Error calling tool {tool_name}: {exc}")
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_name,
                                        "content": f"Error: {str(exc)}",
                                    }
                                )
                        else:
                            result = result_holders[tool_call.id][0]
                            tool_result_content = ""
                            if hasattr(result, "content"):
                                for content_item in result.content:
                                    if hasattr(content_item, "text"):
                                        tool_result_content += content_item.text + "\n"
                            else:
                                tool_result_content = str(result)

                            print(
                                f"Tool execution result: {tool_result_content[:500]}..."
                            )  # show first 500 chars
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": tool_result_content,
                                }
                            )
                else:
                    # No more tool calls, AI is providing final text response
                    print("\nFinal response from AI:")
                    print(response_message.content)
                    break
    finally:
        # Ensure all sessions are closed properly
        print("\nCleaning up sessions...")
        manager.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
