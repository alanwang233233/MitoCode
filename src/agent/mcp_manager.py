import asyncio
import threading
from contextlib import AsyncExitStack
import loguru
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# noinspection DuplicatedCode,PyAbstractClass
class MCPManager:
    @staticmethod
    def start_mcp_servers() -> list[StdioServerParameters]:
        server_list : list[StdioServerParameters] = []
        # TODO:启动mcp服务器
        return server_list

    def __init__(self):
        self.server_params_list = []
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.all_tools = []
        self.tool_to_session_map = {}
        self.sessions = []
        self.stack = AsyncExitStack()
        try:
            self.server_params_list = self.start_mcp_servers()
        except Exception as e:
            loguru.logger.error(f"Error starting MCP servers: {e}")

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_async(self):
        for params in self.server_params_list:
            loguru.logger.debug(f"Connecting to server: {params.command} {' '.join(params.args)}")
            transport_manager = stdio_client(params)
            read_stream, write_stream = await self.stack.enter_async_context(transport_manager)
            session = await self.stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self.sessions.append(session)
            loguru.logger.debug(f"Connected to {params.command} {params.args[1] if params.args else ''} MCP Server.")

        for session in self.sessions:
            response = await session.list_tools()
            for tool in response.tools:
                self.all_tools.append(tool)
                self.tool_to_session_map[tool.name] = session
                loguru.logger.debug(f"Loaded tool: {tool.name}")

    def start(self):
        self.thread.start()
        future = asyncio.run_coroutine_threadsafe(self._init_async(), self.loop)
        future.result()

    def get_tools(self):
        return self.all_tools

    async def _call_tool_async(self, name, args):
        if name in self.tool_to_session_map:
            session = self.tool_to_session_map[name]
            result = await session.call_tool(name, arguments=args)
            return result
        raise ValueError(f"Unknown tool: {name}")

    def call_tool(self, name, args):
        future = asyncio.run_coroutine_threadsafe(self._call_tool_async(name, args), self.loop)
        return future.result()

    def cleanup(self):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.stack.aclose(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()