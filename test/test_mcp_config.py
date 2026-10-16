import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.backend.mcp.mcp_config import McpServerConfig, McpServerKind


def test_mcp_config_error_is_a_mitocode_base_error():
    from mitocode.backend.exception.mitocode_exception import MitoCodeMcpError

    assert issubclass(MitoCodeMcpConfigError, MitoCodeMcpError)


def test_parse_http_server_config():
    config = McpServerConfig.parse(
        "weather", {"url": "https://weather.example.com/mcp"}
    )
    assert config.name == "weather"
    assert config.kind is McpServerKind.HTTP
    assert config.url == "https://weather.example.com/mcp"
    assert config.timeout == 60.0
    assert config.enabled is True


def test_parse_stdio_server_config_with_env_and_cwd():
    config = McpServerConfig.parse(
        "files",
        {
            "command": "npx",
            "args": ["-y", "@mcp/files"],
            "env": {"TOKEN": "x"},
            "cwd": "/tmp",
        },
    )
    assert config.kind is McpServerKind.STDIO
    assert config.command == "npx"
    assert config.args == ["-y", "@mcp/files"]
    assert config.env == {"TOKEN": "x"}
    assert config.cwd == "/tmp"


def test_parse_rejects_both_url_and_command():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse("bad", {"url": "https://x/mcp", "command": "npx"})


def test_parse_rejects_neither_url_nor_command():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse("bad", {"env": {}})


def test_parse_rejects_non_http_url():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse("bad", {"url": "ftp://x/mcp"})


def test_parse_rejects_non_string_argv():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse("bad", {"command": "npx", "args": ["ok", 3]})


def test_parse_rejects_non_positive_timeout():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse("bad", {"url": "https://x/mcp", "timeout": 0})


def test_parse_defaults_timeout_and_connect_timeout():
    config = McpServerConfig.parse("a", {"url": "https://x/mcp"})
    assert config.timeout == 60.0
    assert config.connect_timeout == 30.0
    assert config.enabled is True


def test_parse_rejects_non_positive_connect_timeout():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse("a", {"url": "https://x/mcp", "connect_timeout": -1})


def test_parse_config_reads_mcp_servers_wrapper():
    configs = McpServerConfig.parse_config(
        {
            "mcpServers": {
                "weather": {"url": "https://weather.example.com/mcp"},
                "files": {"command": "npx", "args": ["-y", "files-mcp"]},
            }
        }
    )
    assert [c.name for c in configs] == ["weather", "files"]
    assert [c.kind for c in configs] == [McpServerKind.HTTP, McpServerKind.STDIO]


def test_parse_config_rejects_missing_mcp_servers_key():
    with pytest.raises(MitoCodeMcpConfigError):
        McpServerConfig.parse_config({"weather": {"url": "https://x/mcp"}})


def test_build_transport_returns_http_transport():
    from fastmcp.client.transports import StreamableHttpTransport

    config = McpServerConfig.parse(
        "weather", {"url": "https://weather.example.com/mcp", "headers": {"A": "b"}}
    )
    transport = config.build_transport()
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == "https://weather.example.com/mcp"


def test_build_transport_returns_stdio_transport():
    from fastmcp.client.transports import StdioTransport

    config = McpServerConfig.parse("files", {"command": "npx", "args": ["-y", "f"]})
    transport = config.build_transport()
    assert isinstance(transport, StdioTransport)
    assert transport.command == "npx"
    assert transport.args == ["-y", "f"]


def test_parse_error_message_never_echoes_header_values():
    secret = "Bearer sk-live-DO-NOT-LEAK"
    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpServerConfig.parse(
            "weather",
            {
                "url": "https://weather.example.com/mcp",
                "headers": {"Authorization": secret, "X-Retry": 5},
            },
        )
    message = str(excinfo.value)
    assert "X-Retry" in message
    assert secret not in message


def test_parse_error_message_never_echoes_env_values():
    secret = "env-token-DO-NOT-LEAK"
    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpServerConfig.parse(
            "files", {"command": "npx", "env": {"MCP_TOKEN": secret, "PORT": 8080}}
        )
    message = str(excinfo.value)
    assert "PORT" in message
    assert secret not in message


def test_parse_error_message_never_echoes_a_bare_header_string():
    secret = "sk-live-BARE-DO-NOT-LEAK"
    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpServerConfig.parse(
            "weather",
            {"url": "https://weather.example.com/mcp", "headers": secret},
        )
    message = str(excinfo.value)
    assert "headers" in message
    assert secret not in message


def test_parse_error_message_never_echoes_a_url_credential():
    secret = "url-password-DO-NOT-LEAK"
    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpServerConfig.parse("weather", {"url": f"ftp://user:{secret}@x/mcp"})
    message = str(excinfo.value)
    assert "ftp" in message
    assert secret not in message


def test_parse_error_message_never_echoes_argv_credentials():
    secret = "argv-token-DO-NOT-LEAK"
    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpServerConfig.parse(
            "files", {"command": "npx", "args": ["-y", "--token", secret, 3]}
        )
    message = str(excinfo.value)
    assert "args[3]" in message
    assert secret not in message
