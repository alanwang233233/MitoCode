"""Connection lifecycle for a single MCP server.

:class:`McpClient` owns one server connection: it opens the transport, caches
the tool catalogue the server advertises, and invokes tools by name.  It
deliberately owns *only* the connection — assembling a set of servers and
deciding which ones to start is a later layer's job.

The asymmetry that shapes this module is between :meth:`McpClient.connect` and
:meth:`McpClient.call_tool`.  Connecting is a *best-effort* operation: a server
that is down, misconfigured or simply slow must not abort the startup of the
servers that do work, so :meth:`connect` records the failure into
:attr:`McpClient.state` and :attr:`McpClient.last_error` and returns ``False``
instead of raising.  Invoking a tool, by contrast, is a direct answer to a
direct request from the model, so :meth:`call_tool` raises
:class:`MitoCodeMcpToolCallError` and lets the caller decide what to do.
"""

import asyncio
import json
from contextlib import AsyncExitStack
from enum import Enum
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpToolCallError
from mitocode.backend.mcp.mcp_config import McpServerConfig


class McpConnectionState(str, Enum):
    """Connection state of a single :class:`McpClient`."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    FAILED = "failed"


class McpClient:
    """One MCP server connection plus its cached tool catalogue.

    Args:
        config: The validated server definition to connect to.
    """

    def __init__(self, config: McpServerConfig):
        """Store the server definition; the connection opens on :meth:`connect`.

        Args:
            config: The validated server definition to connect to.
        """
        self._config = config
        self._client: Optional[Client] = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools: List[Any] = []
        self._state = McpConnectionState.DISCONNECTED
        self._last_error: str = ""

    def _build_fastmcp_client(self) -> Client:
        """Construct the FastMCP client for this server.

        Exists as a seam so tests can substitute an in-memory client.
        """
        return Client(
            self._config.build_transport(),
            name=self._config.name,
            timeout=self._config.timeout,
        )

    @property
    def server_name(self) -> str:
        """Name this server's tools are namespaced under."""
        return self._config.name

    @property
    def state(self) -> McpConnectionState:
        """Current connection state, as of the last :meth:`connect` or :meth:`disconnect`."""
        return self._state

    @property
    def last_error(self) -> str:
        """Description of the most recent failure, or ``""`` when there is none."""
        return self._last_error

    @property
    def tools(self) -> List[Any]:
        """Tool objects cached by :meth:`connect`; empty while not connected."""
        return self._tools

    async def connect(self) -> bool:
        """Open the connection and cache the server's tool catalogue.

        Failures are recorded rather than raised: an unreachable server must not
        take down the servers that are reachable.  Callers read :attr:`state` or
        the return value to find out what happened.

        The timeout path deliberately closes the client rather than merely
        abandoning the await.  FastMCP's ``StdioTransport`` spawns its subprocess
        inside a *detached* task and the calling coroutine only waits on that
        task's ready event, so cancelling our await — which is all
        ``asyncio.wait_for`` does — leaves the transport's connect task running
        and the child process alive as an orphan.  Only ``Client.close()`` sets
        the transport's stop event and tears down the process group.

        Returns:
            ``True`` when the client is connected and its tools are cached,
            ``False`` otherwise.
        """
        if self._state is McpConnectionState.CONNECTED:
            return True

        timeout = self._config.connect_timeout
        try:
            # Assigned before entering the context so that a handshake which
            # never completes still leaves close() reachable from _force_close.
            client = self._build_fastmcp_client()
            self._client = client
            stack = AsyncExitStack()
            self._stack = stack
            await asyncio.wait_for(stack.enter_async_context(client), timeout=timeout)
            # Discovery is a second round trip and needs the same deadline;
            # without it a server that accepts connections but never answers
            # would hang forever.
            self._tools = list(
                await asyncio.wait_for(client.list_tools(), timeout=timeout)
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Cleanup runs outside the timed-out await, or the teardown itself
            # would be cancelled and the subprocess orphaned after all.
            await self._force_close()
            self._state = McpConnectionState.FAILED
            self._last_error = f"{type(exc).__name__}: {exc}"
            return False

        self._state = McpConnectionState.CONNECTED
        self._last_error = ""
        return True

    async def _force_close(self) -> None:
        """Tear the connection down, swallowing and recording cleanup errors.

        Shared by the failure path of :meth:`connect` and by
        :meth:`disconnect` so there is exactly one way to release a connection.
        The handles are dropped before the awaits so that a transport which
        raises while closing still leaves the object in a clean state.
        """
        client = self._client
        stack = self._stack
        self._client = None
        self._stack = None
        self._tools = []

        errors = []
        if client is not None:
            try:
                await client.close()
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"close failed: {type(exc).__name__}: {exc}")
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"stack close failed: {type(exc).__name__}: {exc}")
        if errors:
            self._last_error = "; ".join(errors)

    async def disconnect(self) -> None:
        """Close the connection if one is open.  Idempotent.

        Errors raised while closing are swallowed and recorded in
        :attr:`last_error` rather than propagated, because a failing teardown
        must not prevent the remaining servers from being shut down.
        """
        if self._stack is None and self._client is None:
            return
        await self._force_close()
        self._state = McpConnectionState.DISCONNECTED

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke *tool_name* on the connected server and render the result as text.

        Args:
            tool_name: The tool's name **as advertised by the server**, without
                the ``mcp__<server>__`` namespace prefix.
            arguments: Keyword arguments to forward to the tool.

        Returns:
            The tool result as text.

        Raises:
            MitoCodeMcpToolCallError: If the client is not connected, the call
                times out, or the server reports a tool error.
        """
        server = self._config.name
        client = self._client
        if client is None:
            raise MitoCodeMcpToolCallError(
                f"MCP server '{server}' is not connected, "
                f"cannot call tool '{tool_name}'"
            )

        timeout = self._config.timeout
        try:
            result = await asyncio.wait_for(
                client.call_tool(tool_name, arguments, timeout=timeout),
                timeout=timeout,
            )
        except ToolError as exc:
            raise MitoCodeMcpToolCallError(
                f"MCP tool '{tool_name}' on server '{server}' failed: {exc}"
            ) from exc
        except asyncio.TimeoutError as exc:
            raise MitoCodeMcpToolCallError(
                f"MCP tool '{tool_name}' on server '{server}' timed out "
                f"after {timeout} seconds"
            ) from exc
        except MitoCodeMcpToolCallError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            # Arguments are never echoed: they routinely carry file contents and
            # credentials, and this text reaches the TUI error pane and logs.
            raise MitoCodeMcpToolCallError(
                f"MCP tool '{tool_name}' on server '{server}' raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return self._render_result(result)

    @staticmethod
    def _render_result(result: Any) -> str:
        """Render a fastmcp ``CallToolResult`` as plain text.

        Structured results are JSON-encoded so the model sees data rather than a
        Python repr.  A result carrying only non-text blocks — images, audio,
        embedded resources — is reported as such instead of collapsing to an
        empty string, which would read to the model as a successful empty answer.

        Args:
            result: The ``CallToolResult`` returned by fastmcp.

        Returns:
            The rendered text, possibly the empty string.
        """
        data = getattr(result, "data", None)
        if data is not None:
            if isinstance(data, str):
                return data
            return json.dumps(data, ensure_ascii=False, default=str)

        content = getattr(result, "content", None) or []
        texts = [block.text for block in content if hasattr(block, "text")]
        joined = "".join(texts)
        if joined:
            return joined
        if content:
            return (
                f"[MCP server returned {len(content)} non-text content block(s), "
                "which are not supported here]"
            )
        return ""
