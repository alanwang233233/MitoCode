import subprocess
import sys
import os
import asyncio
import json
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from openai import AsyncOpenAI

# Configuration
OPENAI_API_KEY = ""

# 增加超时时间（从30秒改为120秒）
COMMAND_TIMEOUT = 120


async def ensure_dependencies():
    print("Checking and installing dependencies...")
    try:
        # Check Node.js and npm
        subprocess.run(["npm", "--version"], check=True, capture_output=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Check uv
        subprocess.run(["uv", "--version"], check=True, capture_output=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv", "mcp", "openai"], check=True, timeout=300)


async def start_mcp_servers():
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
            env=env
        ),
        # Terminal controller MCP server parameters
        StdioServerParameters(
            command=sys.executable,
            args=["-m", "uv", "tool", "run", "-q", "--from", "terminal-controller", "terminal_controller"],
            env=env
        )
    ]

    return server_params_list


async def main():
    if not OPENAI_API_KEY:
        print(
            "WARNING: OPENAI_API_KEY environment variable is not set. The OpenAI API call will fail unless you provide one.")

    await ensure_dependencies()

    print("\nStarting MCP servers...")
    server_params_list = await start_mcp_servers()

    sessions = []
    transport_managers = []

    # We use AsyncExitStack to manage multiple server connections easily
    async with AsyncExitStack() as stack:
        for params in server_params_list:
            print(f"Connecting to server: {params.command} {' '.join(params.args)}")
            transport_manager = stdio_client(params)
            read_stream, write_stream = await stack.enter_async_context(transport_manager)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            sessions.append(session)
            transport_managers.append(transport_manager)
            print(f"Connected to {params.command} {params.args[1] if len(params.args) > 1 else ''} MCP Server.")

        # Aggregate all tools from all sessions
        all_tools = []
        tool_to_session_map = {}
        for session in sessions:
            response = await session.list_tools()
            for tool in response.tools:
                all_tools.append(tool)
                tool_to_session_map[tool.name] = session
                print(f"Loaded tool: {tool.name}")

        # Prepare tools for OpenAI mapping
        openai_tools = []
        for t in all_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            })

        print("\nAll MCP servers initialized and tools loaded.")
        print(f"Total tools available to AI: {len(openai_tools)}")  # 修复 f-string

        if not OPENAI_API_KEY:
            return

        # Example interactions with AI
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://app.nextchat.club/api/siliconflow/v1"  # 修复 URL 多余空格
        )
        messages = [
            {
                "role": "user",
                "content": "run winver"
            }
        ]

        print("\nSending prompt to OpenAI:")
        print(messages[0]["content"])

        try:
            while True:
                print("\nWaiting for OpenAI model response...")
                response = await client.chat.completions.create(
                    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    timeout=COMMAND_TIMEOUT  # 添加 API 请求超时
                )

                response_message = response.choices[0].message
                messages.append(response_message)

                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        print(f"\nModel wants to call tool '{tool_name}' with args {tool_args}")

                        if tool_name in tool_to_session_map:
                            session = tool_to_session_map[tool_name]
                            try:
                                # 添加超时控制
                                result = await asyncio.wait_for(
                                    session.call_tool(tool_name, arguments=tool_args),
                                    timeout=COMMAND_TIMEOUT
                                )
                                tool_result_content = ""
                                if hasattr(result, 'content'):
                                    for content_item in result.content:
                                        if hasattr(content_item, 'text'):
                                            tool_result_content += content_item.text + "\n"
                                else:
                                    tool_result_content = str(result)

                                print(f"Tool execution result: {tool_result_content[:500]}...")  # 增加显示长度
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": tool_result_content
                                })
                            except asyncio.TimeoutError:
                                print(f"Error: Tool '{tool_name}' execution timed out after {COMMAND_TIMEOUT} seconds")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": f"Error: Command timed out after {COMMAND_TIMEOUT} seconds. Try breaking the command into smaller steps."
                                })
                            except Exception as e:
                                print(f"Error calling tool {tool_name}: {e}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": f"Error: {str(e)}"
                                })
                        else:
                            print(f"Unknown tool requested: {tool_name}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": f"Error: unknown tool '{tool_name}'"
                            })
                else:
                    # No more tool calls, AI is providing final text response
                    print("\nFinal response from AI:")
                    print(response_message.content)
                    break
        finally:
            # 确保所有会话正确关闭
            print("\nCleaning up sessions...")
            for session in sessions:
                try:
                    await session.close()
                except Exception as e:
                    print(f"Warning: Error closing session: {e}")


if __name__ == "__main__":
    # Windows 上设置事件循环策略，避免子进程问题
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise