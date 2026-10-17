"""Lifecycle management for the set of configured MCP servers.

:class:`McpManager` owns one :class:`~mitocode.backend.mcp.mcp_client.McpClient`
per configured server and decides when each of them starts, refreshes and
stops.  It takes no interest in what those tools look like: registration is
delegated to :meth:`ToolManager.register_mcp_client`, which namespaces each
tool under the server that advertises it.

Two decisions shape the module.  Connections are driven one server at a time
rather than through a fastmcp ``ClientGroup``, because a group connects its
members as a unit and one unreachable server would take down every tool of
every other server.  And no operation here is allowed to raise on account of
a single server: :meth:`McpManager.start` runs once during startup, and one
broken entry in a shared configuration file must not stop the harness from
starting with the tools that do work.  A server that fails costs you that
server's tools and — because :meth:`McpManager.status` reports the reason —
nothing else.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

from loguru import logger as loguru_logger

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.backend.mcp.mcp_client import McpClient
from mitocode.backend.mcp.mcp_config import McpServerConfig
from mitocode.backend.tool.tool_manager import ToolManager

logger = loguru_logger.bind(name=__name__)


def _describe(exc: Exception) -> str:
    """Render an exception as its type plus its message.

    Args:
        exc: The exception to describe.

    Returns:
        ``"<TypeName>: <message>"``.
    """
    return f"{type(exc).__name__}: {exc}"


def _normalise_servers(
    config: Union[Dict[str, Any], List[McpServerConfig]],
) -> List[McpServerConfig]:
    """Turn whatever the caller passed as configuration into validated entries.

    A raw ``{"mcpServers": {...}}`` document is handed to
    :meth:`McpServerConfig.parse_config`; a list of already-validated entries
    — from another layer that loaded them — is taken as it is.

    Args:
        config: The MCP configuration, in either accepted shape.

    Returns:
        One validated configuration per entry, in configuration order.

    Raises:
        MitoCodeMcpConfigError: If *config* is neither shape, if a list element
            is not an :class:`McpServerConfig`, or if a raw document holds an
            invalid entry.  Only an element's position and type are ever
            reported, never a value, because ``env``, ``headers`` and ``args``
            carry credentials.
    """
    if isinstance(config, dict):
        return McpServerConfig.parse_config(config)

    if isinstance(config, list):
        servers: List[McpServerConfig] = []
        for index, entry in enumerate(config):
            if not isinstance(entry, McpServerConfig):
                raise MitoCodeMcpConfigError(
                    f"MCP server entry {index} must be an McpServerConfig, "
                    f"got {type(entry).__name__}"
                )
            servers.append(entry)
        return servers

    raise MitoCodeMcpConfigError(
        "MCP configuration must be an object holding an 'mcpServers' mapping or "
        f"a list of McpServerConfig, got {type(config).__name__}"
    )


@dataclass(frozen=True)
class McpServerStatus:
    """A snapshot of one MCP server's state.

    Attributes:
        state: ``"connected"``, ``"disconnected"``, ``"failed"`` or ``"disabled"``.
        error: The last connection error, empty when there is none.
        tool_count: How many tools this server contributed to the registry.
    """

    state: str
    error: str
    tool_count: int


class McpManager:
    """Owns the lifecycle of every configured MCP server.

    One client is built per enabled server up front, so a configuration
    mistake is reported by the constructor rather than by the first connect
    attempt.  Registration goes through :meth:`ToolManager.register_mcp_client`
    and is undone by name, which is what lets one server be refreshed or
    stopped without disturbing another's tools.

    Args:
        config: The MCP configuration: a ``{"mcpServers": {...}}`` document,
            or a list of :class:`McpServerConfig` entries.
        tool_manager: Registry the servers' tools are published into.
        client_factory: Builds the client for one server.  Exists as a seam so
            tests can drive the lifecycle without a subprocess; the default is
            the real :class:`McpClient`.

    Note:
        ``AgentSession`` snapshots the tool registry at construction time, so
        :meth:`start` must complete before any agent session is created,
        otherwise existing sessions will not see the MCP tools.

    Note:
        The connection deadline is not a manager setting: it comes from each
        server's own ``connect_timeout`` (30 seconds by default), because a
        cold ``npx -y`` start can take tens of seconds and a single short
        global timeout would report a healthy server as failed.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], List[McpServerConfig]],
        *,
        tool_manager: ToolManager,
        client_factory: Callable[[McpServerConfig], McpClient] = McpClient,
    ) -> None:
        """Validate the configuration and build a client for each enabled server.

        Args:
            config: The MCP configuration, in either accepted shape.
            tool_manager: Registry the servers' tools are published into.
            client_factory: Builds the client for one server.

        Raises:
            MitoCodeMcpConfigError: If the configuration is invalid in either
                shape.  Raised before any client is built, so a bad document
                cannot leave a half-built manager behind.
        """
        servers = _normalise_servers(config)
        self._tool_manager = tool_manager
        self._client_factory = client_factory
        self._clients: Dict[str, McpClient] = {}
        self._registered: Dict[str, List[str]] = {}
        self._disabled: Set[str] = set()
        self._errors: Dict[str, str] = {}

        for server in servers:
            if server.enabled:
                self._clients[server.name] = client_factory(server)
            else:
                self._disabled.add(server.name)

    async def start(self) -> None:
        """Connect every enabled server and register the tools they advertise.

        Servers are connected concurrently and independently: one that is
        unreachable, slow or broken costs its own tools and is described by
        :meth:`status`, while every other server still comes up.  This method
        never raises.

        A server's tools are registered only after its ``connect`` has
        reported success, because a disconnected client has no catalogue —
        registering then would yield an empty list that reads exactly like a
        server that genuinely advertises nothing.
        """
        await asyncio.gather(
            *(self._connect_one(name) for name in list(self._clients)),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        """Unregister every MCP tool and close every connection.  Idempotent.

        Tools are removed before the connections are closed, so no tool can be
        dispatched to a client that is going away.  One connection that
        refuses to close is recorded against its server in :meth:`status`
        instead of stranding the others.
        """
        for name in list(self._clients):
            await self._unregister_server(name)

        await asyncio.gather(
            *(self._disconnect_one(name) for name in list(self._clients)),
            return_exceptions=True,
        )

    async def refresh(self, server_name: Optional[str] = None) -> None:
        """Reconnect servers so a changed tool catalogue is picked up.

        Every target is unregistered *before* it is reconnected, so the
        registry ends up holding the catalogue the server advertises now —
        including the tools it has dropped.  A reconnect that fails therefore
        leaves that server with no tools rather than with tools that dispatch
        to a client that is not connected.

        Args:
            server_name: The server to refresh; ``None`` refreshes every
                enabled server.

        Raises:
            MitoCodeMcpConfigError: If *server_name* is neither an enabled nor
                a disabled configured server.  Never raised for a server that
                merely fails to reconnect.
        """
        targets = self._refresh_targets(server_name)
        await asyncio.gather(
            *(self._reconnect_one(name) for name in targets),
            return_exceptions=True,
        )

    def status(self) -> Dict[str, McpServerStatus]:
        """Describe every configured server, connected or not.

        Returns:
            One snapshot per server, keyed by server name.  Enabled servers are
            reported first, each with its client's own state and error text,
            then the disabled ones — which have no client and no tools.
        """
        snapshot = {
            name: McpServerStatus(
                state=client.state.value,
                error=client.last_error or self._errors.get(name, ""),
                tool_count=len(self._registered.get(name, [])),
            )
            for name, client in self._clients.items()
        }
        for name in self._disabled:
            snapshot[name] = McpServerStatus(state="disabled", error="", tool_count=0)
        return snapshot

    async def _connect_one(self, name: str) -> None:
        """Connect one server and register the tools it advertises.

        Never raises.  A failure the client reported itself is already in
        :attr:`McpClient.last_error`; one it raised instead is recorded here,
        because a server that failed with no stated reason is exactly the case
        that is hard to diagnose from the outside.

        Args:
            name: Name of the server to connect, as keyed in ``_clients``.
        """
        client = self._clients[name]
        try:
            connected = await client.connect()
        except Exception as exc:  # pylint: disable=broad-except
            self._record_failure(name, exc, "connect")
            return

        if not connected:
            self._errors.pop(name, None)
            logger.warning(
                f"MCP server '{name}' is unavailable; its tools are not registered"
            )
            return

        # Whatever this server registered last time is released first, so a
        # repeated start or refresh cannot leave the previous names behind as
        # tools nobody owns and stop() cannot unregister.
        await self._unregister_server(name)
        try:
            specs = self._tool_manager.register_mcp_client(client)
        except Exception as exc:  # pylint: disable=broad-except
            self._record_failure(name, exc, "register its tools")
            return

        self._registered[name] = [spec.name for spec in specs]
        self._errors.pop(name, None)

    async def _reconnect_one(self, name: str) -> None:
        """Unregister, close and reconnect one server.

        Args:
            name: Name of the server to reconnect, as keyed in ``_clients``.
        """
        await self._unregister_server(name)
        await self._disconnect_one(name)
        await self._connect_one(name)

    async def _disconnect_one(self, name: str) -> None:
        """Close one server's connection, recording a failure instead of raising.

        A teardown that raises is recorded against the server, so a
        connection left open is visible in :meth:`status` rather than silently
        swallowed.

        Args:
            name: Name of the server to close, as keyed in ``_clients``.
        """
        try:
            await self._clients[name].disconnect()
        except Exception as exc:  # pylint: disable=broad-except
            self._record_failure(name, exc, "disconnect")

    async def _unregister_server(self, name: str) -> None:
        """Remove every tool one server registered from the registry.

        Args:
            name: Name of the server whose tools are removed.
        """
        for tool_name in self._registered.pop(name, []):
            self._tool_manager.unregister(tool_name)

    def _record_failure(self, name: str, exc: Exception, action: str) -> None:
        """Remember a failure the client did not report itself, and log it.

        Args:
            name: Name of the MCP server that failed.
            exc: The exception that ended the operation.
            action: What was being attempted, phrased to read as
                ``failed to <action>``.
        """
        self._errors[name] = _describe(exc)
        logger.error(f"MCP server '{name}': failed to {action}: {self._errors[name]}")

    def _refresh_targets(self, server_name: Optional[str]) -> List[str]:
        """Resolve the ``refresh`` argument into the servers to reconnect.

        Args:
            server_name: The requested server, or ``None`` for all of them.

        Returns:
            The names to reconnect.

        Raises:
            MitoCodeMcpConfigError: If *server_name* is not configured.
        """
        if server_name is None:
            return list(self._clients)
        if server_name in self._clients:
            return [server_name]
        if server_name in self._disabled:
            raise MitoCodeMcpConfigError(
                f"MCP server '{server_name}' is disabled and cannot be refreshed"
            )
        raise MitoCodeMcpConfigError(
            f"MCP server '{server_name}' is not configured; the configured "
            f"servers are {sorted(self._clients)}"
        )
