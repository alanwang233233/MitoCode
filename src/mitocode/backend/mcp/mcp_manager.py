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
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
)

from loguru import logger as loguru_logger

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.backend.mcp.mcp_client import McpClient
from mitocode.backend.mcp.mcp_config import McpServerConfig

if TYPE_CHECKING:
    # The tool layer is reached only through the object handed in, never by
    # importing it: the dependency runs one way, tool_manager -> mcp.  A runtime
    # import here would make mitocode.backend.mcp unimportable on its own and
    # would break the symmetry with tool_manager's own TYPE_CHECKING guard for
    # McpClient.  The annotation is quoted because this module has no
    # `from __future__ import annotations`, so annotations are evaluated.
    from mitocode.backend.tool.tool_manager import ToolManager

logger = loguru_logger.bind(name=__name__)


def _describe(exc: BaseException) -> str:
    """Render an exception as its type plus its message.

    Args:
        exc: The exception to describe.

    Returns:
        ``"<TypeName>: <message>"``, or just the type when the exception
        carries no message — a bare ``CancelledError`` has none.
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _normalise_servers(
    config: Union[Dict[str, Any], List[McpServerConfig]],
) -> List[McpServerConfig]:
    """Turn whatever the caller passed as configuration into validated entries.

    A raw ``{"mcpServers": {...}}`` document is handed to
    :meth:`McpServerConfig.parse_config`, whose keys are unique by
    construction; a list of already-validated entries — from another layer that
    loaded them — is taken as it is, subject to the same uniqueness rule.

    A repeated name is refused rather than resolved.  The name is both the key
    a client is stored under and the ``mcp__<name>__`` namespace its tools are
    published in, so letting two entries share one would leave a tool that is
    registered and dispatchable while :meth:`McpManager.status` describes it as
    the *other* entry's — ``disabled`` with ``tool_count=0``, say.  Whichever
    entry happened to come last would silently win, and the tools of the one
    that lost would be unreachable with no way to tell that is what happened.

    Args:
        config: The MCP configuration, in either accepted shape.

    Returns:
        One validated configuration per entry, in configuration order, with
        every name distinct.

    Raises:
        MitoCodeMcpConfigError: If *config* is neither shape, if a list element
            is not an :class:`McpServerConfig`, if two entries share a name, or
            if a raw document holds an invalid entry.  Only an element's
            position, its type and the duplicated name are ever reported, never
            a configuration value, because ``env``, ``headers`` and ``args``
            carry credentials.
    """
    if isinstance(config, dict):
        return McpServerConfig.parse_config(config)

    if isinstance(config, list):
        servers: List[McpServerConfig] = []
        seen: Dict[str, int] = {}
        for index, entry in enumerate(config):
            if not isinstance(entry, McpServerConfig):
                raise MitoCodeMcpConfigError(
                    f"MCP server entry {index} must be an McpServerConfig, "
                    f"got {type(entry).__name__}"
                )
            if entry.name in seen:
                raise MitoCodeMcpConfigError(
                    f"MCP server entry {index} reuses the name {entry.name!r} "
                    f"already taken by entry {seen[entry.name]}; server names "
                    "must be unique because they namespace that server's tools"
                )
            seen[entry.name] = index
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
        tool_manager: "ToolManager",
        client_factory: Callable[[McpServerConfig], McpClient] = McpClient,
    ) -> None:
        """Validate the configuration and build a client for each enabled server.

        The tool registry is held as a duck-typed object: the module never
        imports the tool layer, so any object exposing
        :meth:`~mitocode.backend.tool.tool_manager.ToolManager.register_mcp_client`
        and :meth:`~mitocode.backend.tool.tool_manager.ToolManager.unregister`
        will do.

        Args:
            config: The MCP configuration, in either accepted shape.  Every
                server name must be distinct.
            tool_manager: Registry the servers' tools are published into.
            client_factory: Builds the client for one server.

        Raises:
            MitoCodeMcpConfigError: If the configuration is invalid in either
                shape, or repeats a server name.  Raised before any client is
                built, so a bad document cannot leave a half-built manager
                behind.
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
        never raises for a server that fails.

        A server's tools are registered only after its ``connect`` has
        reported success, because a disconnected client has no catalogue —
        registering then would yield an empty list that reads exactly like a
        server that genuinely advertises nothing.

        Raises:
            asyncio.CancelledError: If a server's connect is cancelled.
                ``McpClient.connect`` re-raises cancellation by contract, so it
                is recorded against that server in :meth:`status` and then
                propagated: a cancellation is the caller's decision to unwind,
                not a server that failed.
        """
        await self._run_each(list(self._clients), self._connect_one)

    async def stop(self) -> None:
        """Unregister every MCP tool and close every connection.  Idempotent.

        Tools are removed before the connections are closed, so no tool can be
        dispatched to a client that is going away — the opposite order would
        leave a registered tool pointing at a dead client, which fails at
        call time in the model's hands rather than here.

        Both phases are isolated per server.  A registry that refuses to give a
        tool back must not skip the teardown that releases the subprocesses,
        and a connection that refuses to close must not strand the others;
        either is recorded against its server in :meth:`status` instead.

        Raises:
            asyncio.CancelledError: If one server's unregistration or
                disconnection is cancelled.
        """
        await self._run_each(list(self._clients), self._unregister_one)
        await self._run_each(list(self._clients), self._disconnect_one)

    async def refresh(self, server_name: Optional[str] = None) -> None:
        """Reconnect servers so a changed tool catalogue is picked up.

        Every target is unregistered *before* it is reconnected, so the
        registry ends up holding the catalogue the server advertises now —
        including the tools it has dropped.  A reconnect that fails therefore
        leaves that server with no tools rather than with tools that dispatch
        to a client that is not connected.

        A server that has not been started yet is connected by this call.
        Refusing that would make a manager being recovered after a crash reject
        the one call meant to recover it.

        Args:
            server_name: The server to refresh; ``None`` refreshes every
                enabled server.

        Raises:
            MitoCodeMcpConfigError: If *server_name* is neither an enabled nor
                a disabled configured server.  Never raised for a server that
                merely fails to reconnect.
            asyncio.CancelledError: If one target's work is cancelled.
        """
        await self._run_each(self._refresh_targets(server_name), self._reconnect_one)

    def status(self) -> Dict[str, McpServerStatus]:
        """Describe every configured server, connected or not.

        A server name resolves to exactly one entry: :func:`_normalise_servers`
        refuses a repeated name, so a server is never both enabled and
        disabled.

        Returns:
            One snapshot per server, keyed by server name.  Enabled servers are
            reported first, each with its client's own state and its most recent
            error, then the disabled ones — which have no client and no tools.
        """
        snapshot = {
            name: McpServerStatus(
                state=client.state.value,
                # A failure the manager recorded is the more recent of the two:
                # the client's own text describes an earlier attempt, and the
                # recorded one describes the step that just failed.
                error=self._errors.get(name, "") or client.last_error,
                tool_count=len(self._registered.get(name, [])),
            )
            for name, client in self._clients.items()
        }
        for name in self._disabled:
            snapshot[name] = McpServerStatus(state="disabled", error="", tool_count=0)
        return snapshot

    async def _run_each(
        self, names: List[str], operation: Callable[[str], Awaitable[None]]
    ) -> None:
        """Run *operation* once per name, concurrently, isolating failures.

        Every result is awaited and every name is visited, so one server that
        raises cannot cost any other server its turn — which is the whole point
        of the gather and the reason it is not a plain loop.

        The results are then inspected rather than discarded.
        ``return_exceptions=True`` hands back a child's ``CancelledError`` as a
        *result* instead of raising it, and ``except Exception`` cannot catch
        one, so a cancelled connect would otherwise be dropped with no trace
        and the server would report a failure with no reason at all.  It is
        recorded against that server and re-raised, because a cancellation is
        an instruction to unwind, not a server that failed.

        Args:
            names: Servers to run *operation* for, each visited exactly once.
            operation: One server's worth of work, e.g. :meth:`_connect_one`.

        Raises:
            asyncio.CancelledError: If *operation* was cancelled for one server.
        """
        results = await asyncio.gather(
            *(operation(name) for name in names),
            return_exceptions=True,
        )
        for name, result in zip(names, results):
            if isinstance(result, asyncio.CancelledError):
                self._record_failure(name, result, "cancelled before it finished")
                raise result

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
            self._record_failure(name, exc, "failed to connect")
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
        await self._unregister_one(name)
        try:
            specs = self._tool_manager.register_mcp_client(client)
        except Exception as exc:  # pylint: disable=broad-except
            self._record_failure(name, exc, "failed to register its tools")
            return

        self._registered[name] = [spec.name for spec in specs]
        self._errors.pop(name, None)

    async def _reconnect_one(self, name: str) -> None:
        """Unregister, close and reconnect one server.

        Args:
            name: Name of the server to reconnect, as keyed in ``_clients``.
        """
        await self._unregister_one(name)
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
            self._record_failure(name, exc, "failed to disconnect")

    async def _unregister_one(self, name: str) -> None:
        """Remove every tool one server registered from the registry.

        The ownership record is updated per tool and only once the registry has
        confirmed the outcome, which is the reason ``unregister``'s ``bool``
        return is not discarded.  Three outcomes are possible and only two let
        go of the name:

        * removed — the tool is gone, so drop it from the record;
        * already absent — somebody removed it behind the manager's back, so
          there is nothing left to retry and it is dropped too;
        * raised — the outcome is unknown, so the name stays owned and a later
          :meth:`stop` can finish the job.  Forgetting it here would leave a
          tool in the registry that no one will ever remove.

        A failure here is recorded and never raised, so it cannot skip the
        teardown that releases this and every other server's resources.

        Args:
            name: Name of the server whose tools are removed.
        """
        remaining: List[str] = []
        for tool_name in self._registered.get(name, []):
            try:
                removed = self._tool_manager.unregister(tool_name)
            except Exception as exc:  # pylint: disable=broad-except
                remaining.append(tool_name)
                self._record_failure(
                    name, exc, f"failed to unregister the tool {tool_name!r}"
                )
                continue
            if not removed:
                logger.warning(
                    f"MCP tool '{tool_name}' was not in the registry; "
                    f"server '{name}' no longer owns it"
                )
        if remaining:
            self._registered[name] = remaining
            return
        self._registered.pop(name, None)
        # Nothing failed and nothing is left, so whatever this server last
        # complained about is history.  Leaving it in place would have status()
        # report a failure that no longer describes anything.
        self._errors.pop(name, None)

    def _record_failure(self, name: str, exc: BaseException, problem: str) -> None:
        """Remember a failure the client did not report itself, and log it.

        Args:
            name: Name of the MCP server that failed.
            exc: The exception that ended the operation.
            problem: A complete phrase saying what went wrong, e.g.
                ``"failed to connect"`` — it is logged verbatim and may name a
                tool, which is never a credential.
        """
        self._errors[name] = _describe(exc)
        logger.error(f"MCP server '{name}': {problem}: {self._errors[name]}")

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
