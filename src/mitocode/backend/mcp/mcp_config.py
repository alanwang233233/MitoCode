"""Normalisation and validation of MCP server configuration entries.

The Claude Desktop configuration shape — ``{"mcpServers": {"<name>": {...}}}``
— is the accepted input format.  An entry with a ``url`` describes a remote
server reached over Streamable HTTP; an entry with a ``command`` describes a
local server launched as a subprocess and spoken to over stdio.

This layer only normalises configuration.  It never touches the filesystem and
never opens a connection: :meth:`McpServerConfig.build_transport` merely
assembles the transport object that a later layer hands to a live client.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, NoReturn

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpConfigError

try:
    from fastmcp.client.transports import (
        ClientTransport,
        StdioTransport,
        StreamableHttpTransport,
    )
except ImportError as exc:  # pragma: no cover - fastmcp is a hard dependency
    raise ImportError(
        "MCP support requires the 'fastmcp' package; "
        "install it with 'pip install fastmcp>=4.1,<5'"
    ) from exc


class McpServerKind(str, Enum):
    """Transport used to reach an MCP server."""

    STDIO = "stdio"
    HTTP = "http"


@dataclass(frozen=True)
class McpServerConfig:  # pylint: disable=too-many-instance-attributes
    """A validated MCP server definition.

    Attributes:
        name: Unique server name used to namespace its tools.
        kind: Which transport this server is reached over.
        url: Streamable HTTP endpoint; empty for stdio servers.
        command: Executable to launch; empty for HTTP servers.
        args: Command-line arguments passed to ``command``.
        env: Extra environment variables for the subprocess.
        cwd: Working directory for the subprocess.
        headers: Extra HTTP headers, typically carrying a static token.
        timeout: Per-call timeout in seconds.
        connect_timeout: Seconds allowed for the connection handshake.
            Generous by default because a cold ``npx -y`` start can take
            tens of seconds.
        enabled: Whether the manager should attempt to connect.
    """

    name: str
    kind: McpServerKind
    url: str = ""
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 60.0
    connect_timeout: float = 30.0
    enabled: bool = True

    @classmethod
    def parse(cls, name: str, raw: dict) -> "McpServerConfig":
        """Normalise one raw server entry into a validated configuration.

        Every field is type-checked but only the ones that belong to the
        resolved transport are meaningful; entries that carry a stray field for
        the other transport are accepted and simply carried along, because
        being lenient about shape keeps hand-written configuration usable.

        Args:
            name: Unique server name, normally the key of the ``mcpServers``
                mapping.
            raw: One server entry.  It must carry exactly one of ``url`` (a
                remote server) or ``command`` (a local stdio server).

        Returns:
            A frozen configuration with every field populated.

        Raises:
            MitoCodeMcpConfigError: If ``name`` is blank or not a string, if
                ``raw`` is not a mapping, if the transport selector is missing,
                ambiguous or malformed, or if any optional field has the wrong
                type.
        """
        if not isinstance(name, str) or not name.strip():
            raise MitoCodeMcpConfigError(
                f"MCP server name must be a non-empty string, got {name!r}"
            )
        if not isinstance(raw, dict):
            raise MitoCodeMcpConfigError(
                f"MCP server '{name}': entry must be an object, "
                f"got {type(raw).__name__}"
            )

        has_url = "url" in raw
        has_command = "command" in raw
        if has_url and has_command:
            raise MitoCodeMcpConfigError(
                f"MCP server '{name}': 'url' and 'command' are mutually "
                "exclusive, provide exactly one"
            )
        if not has_url and not has_command:
            raise MitoCodeMcpConfigError(
                f"MCP server '{name}': either 'url' for a remote server or "
                f"'command' for a stdio server is required, got keys {sorted(map(str, raw))}"
            )

        url = ""
        command = ""
        if has_url:
            kind = McpServerKind.HTTP
            url = _read_http_url(name, raw["url"])
        else:
            kind = McpServerKind.STDIO
            command = _read_stdio_command(name, raw["command"])

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            _invalid(name, "enabled", "a boolean", enabled)

        return cls(
            name=name,
            kind=kind,
            url=url,
            command=command,
            args=_read_str_list(name, raw, "args"),
            env=_read_str_mapping(name, raw, "env"),
            cwd=_read_str(name, raw, "cwd"),
            headers=_read_str_mapping(name, raw, "headers"),
            timeout=_read_positive_number(name, raw, "timeout", 60.0),
            connect_timeout=_read_positive_number(name, raw, "connect_timeout", 30.0),
            enabled=enabled,
        )

    @classmethod
    def parse_config(cls, config: dict) -> List["McpServerConfig"]:
        """Normalise a whole ``{"mcpServers": {...}}`` document.

        Args:
            config: The full configuration mapping.

        Returns:
            One validated configuration per entry, in document order.

        Raises:
            MitoCodeMcpConfigError: If ``config`` is not a mapping, has no
                ``"mcpServers"`` key, or if that key does not hold a mapping.
                Individual entries are reported by :meth:`parse`.
        """
        if not isinstance(config, dict):
            raise MitoCodeMcpConfigError(
                "MCP configuration must be an object holding an 'mcpServers' "
                f"mapping, got {type(config).__name__}"
            )
        if "mcpServers" not in config:
            raise MitoCodeMcpConfigError(
                "MCP configuration is missing the 'mcpServers' key, "
                f"got keys {sorted(map(str, config))}"
            )
        servers = config["mcpServers"]
        if not isinstance(servers, dict):
            raise MitoCodeMcpConfigError(
                "MCP configuration 'mcpServers' must be an object, "
                f"got {type(servers).__name__}"
            )
        return [cls.parse(name, raw) for name, raw in servers.items()]

    def build_transport(self) -> ClientTransport:
        """Build the fastmcp transport object matching this server's kind.

        No I/O happens here — the transport only carries connection parameters
        until a client actually uses it.

        Returns:
            A ``StreamableHttpTransport`` for remote servers or a
            ``StdioTransport`` for local ones.  Empty optional mappings are
            passed as ``None`` because that is how fastmcp spells "unset".

        Raises:
            MitoCodeMcpConfigError: If ``kind`` is not a known transport kind.
        """
        if self.kind is McpServerKind.HTTP:
            return StreamableHttpTransport(url=self.url, headers=self.headers or None)
        if self.kind is McpServerKind.STDIO:
            return StdioTransport(
                command=self.command,
                args=self.args,
                env=self.env or None,
                cwd=self.cwd or None,
            )
        raise MitoCodeMcpConfigError(
            f"MCP server '{self.name}': unknown transport kind {self.kind!r}"
        )


def _invalid(name: str, key: str, expected: str, value: Any) -> NoReturn:
    """Report one configuration field that failed validation.

    Args:
        name: Name of the offending MCP server.
        key: The configuration key that failed validation.
        expected: Human-readable description of the accepted shape.
        value: The offending value.

    Raises:
        MitoCodeMcpConfigError: Always.
    """
    raise MitoCodeMcpConfigError(
        f"MCP server '{name}': '{key}' must be {expected}, got {value!r}"
    )


def _read_http_url(name: str, value: Any) -> str:
    """Validate the ``url`` field of a remote server entry.

    Args:
        name: Name of the MCP server, used in the error message.
        value: The raw ``url`` value.

    Returns:
        The validated endpoint.

    Raises:
        MitoCodeMcpConfigError: If ``value`` is not an http(s) endpoint.
    """
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise MitoCodeMcpConfigError(
            f"MCP server '{name}': 'url' must be an http:// or https:// "
            f"endpoint, got {value!r}"
        )
    return value


def _read_stdio_command(name: str, value: Any) -> str:
    """Validate the ``command`` field of a local server entry.

    Args:
        name: Name of the MCP server, used in the error message.
        value: The raw ``command`` value.

    Returns:
        The validated executable name.

    Raises:
        MitoCodeMcpConfigError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise MitoCodeMcpConfigError(
            f"MCP server '{name}': 'command' must be a non-empty string, "
            f"got {value!r}"
        )
    return value


def _read_str(name: str, raw: dict, key: str) -> str:
    """Read an optional string field, defaulting to the empty string.

    Args:
        name: Name of the MCP server, used in the error message.
        raw: The raw server entry.
        key: The configuration key to read.

    Returns:
        The field value, or ``""`` when the key is absent.

    Raises:
        MitoCodeMcpConfigError: If the value is present but not a string.
    """
    value = raw.get(key, "")
    if not isinstance(value, str):
        _invalid(name, key, "a string", value)
    return value


def _read_str_list(name: str, raw: dict, key: str) -> List[str]:
    """Read an optional list-of-strings field, defaulting to an empty list.

    Args:
        name: Name of the MCP server, used in the error message.
        raw: The raw server entry.
        key: The configuration key to read.

    Returns:
        A copy of the field value, so later mutation of the caller's list
        cannot reach the frozen configuration.

    Raises:
        MitoCodeMcpConfigError: If the value is not a list of strings.
    """
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _invalid(name, key, "a list of strings", value)
    return list(value)


def _read_str_mapping(name: str, raw: dict, key: str) -> Dict[str, str]:
    """Read an optional string-to-string mapping, defaulting to an empty dict.

    Args:
        name: Name of the MCP server, used in the error message.
        raw: The raw server entry.
        key: The configuration key to read.

    Returns:
        A copy of the field value, so later mutation of the caller's mapping
        cannot reach the frozen configuration.

    Raises:
        MitoCodeMcpConfigError: If the value is not a mapping of strings.
    """
    value = raw.get(key, {})
    if not isinstance(value, dict):
        _invalid(name, key, "an object mapping strings to strings", value)
    for entry_key, entry_value in value.items():
        if not isinstance(entry_key, str) or not isinstance(entry_value, str):
            _invalid(name, key, "an object mapping strings to strings", value)
    return dict(value)


def _read_positive_number(name: str, raw: dict, key: str, default: float) -> float:
    """Read an optional strictly positive number field.

    ``bool`` is rejected explicitly: it is a subclass of ``int`` in Python, and
    ``timeout: true`` is a configuration mistake rather than a timeout of 1.
    The comparison is spelled ``not value > 0`` rather than ``value <= 0`` so a
    NaN timeout is rejected too — it is not greater than zero.

    Args:
        name: Name of the MCP server, used in the error message.
        raw: The raw server entry.
        key: The configuration key to read.
        default: Value used when the key is absent.

    Returns:
        The field value as a float.

    Raises:
        MitoCodeMcpConfigError: If the value is a boolean, is not a number, or
            is not greater than zero.
    """
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        _invalid(name, key, "a positive number", value)
    return float(value)
