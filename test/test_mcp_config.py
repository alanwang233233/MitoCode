"""Unit tests for :mod:`mitocode.backend.mcp.mcp_config`.

The module is pure: it normalises a ``{"mcpServers": {...}}`` document into
frozen :class:`McpServerConfig` values and assembles a fastmcp transport
object.  Nothing here touches the filesystem or a network, so the whole suite
runs offline and fast.

Two properties are load-bearing beyond "it raises on bad input":

* every optional field is *copied* out of the caller's raw mapping, so later
  mutation of the document cannot reach a config that has already been
  validated; and
* no exception message ever echoes a value from a secret-bearing field
  (``url``, ``args``, ``env``, ``headers``), because those messages end up in
  the TUI error pane and in logs.
"""

import os
import re
import sys
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.backend.mcp import mcp_config
from mitocode.backend.mcp.mcp_config import McpServerConfig, McpServerKind

# A valid https endpoint used wherever the transport selector is incidental.
HTTP_ENDPOINT = "https://weather.example.com/mcp"

# Distinct sentinel credentials.  None of these strings may ever appear inside
# an exception message raised while validating a secret-bearing field.
SECRET_HEADER = "Bearer sk-live-HEADER-DO-NOT-LEAK"
SECRET_ENV = "env-token-ENV-DO-NOT-LEAK"
SECRET_ARGV = "argv-token-ARGV-DO-NOT-LEAK"
SECRET_URL = "url-password-URL-DO-NOT-LEAK"


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


# --------------------------------------------------------------------------
# Enum surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected_token"),
    [(McpServerKind.STDIO, "stdio"), (McpServerKind.HTTP, "http")],
    ids=["stdio", "http"],
)
def test_mcp_server_kind_is_a_string_enum_token_used_in_config_documents(
    kind, expected_token
):
    """A kind doubles as the lowercase token written into a config document."""
    assert (
        kind.value == expected_token
    ), f"{kind!r} must serialise as {expected_token!r}"
    assert isinstance(
        kind, str
    ), "McpServerKind must subclass str so config values can be compared to it"


# --------------------------------------------------------------------------
# parse() — accepted shapes and defaulting
# --------------------------------------------------------------------------


def test_parse_leaves_the_mutable_optional_fields_empty_when_they_are_absent():
    """Absent optional fields must default to empty containers, never None."""
    config = McpServerConfig.parse("weather", {"url": HTTP_ENDPOINT})

    assert config.args == [], "args must default to an empty list"
    assert config.env == {}, "env must default to an empty dict"
    assert config.headers == {}, "headers must default to an empty dict"
    assert config.cwd == "", "cwd must default to the empty string"
    assert config.command == "", "an HTTP entry must not gain a command"


def test_parse_keeps_stray_fields_that_belong_to_the_other_transport():
    """Leniency contract: a stdio entry may carry headers/cwd and keeps them."""
    config = McpServerConfig.parse(
        "files",
        {"command": "npx", "cwd": "/tmp", "headers": {"X-Key": "value"}},
    )

    assert config.kind is McpServerKind.STDIO, "the entry is still a stdio server"
    assert config.headers == {
        "X-Key": "value"
    }, "a stray header mapping must be carried along, not dropped"
    assert config.cwd == "/tmp", "a stray cwd must be carried along, not dropped"


def test_parse_honours_enabled_false():
    """enabled=False must survive parsing, not be coerced to the True default."""
    config = McpServerConfig.parse("weather", {"url": HTTP_ENDPOINT, "enabled": False})

    assert (
        config.enabled is False
    ), "an explicitly disabled server must stay disabled after parsing"


def test_parse_copies_the_callers_args_list():
    """Mutation of the caller's argv after parse must not reach the config."""
    args = ["-y", "files-mcp"]
    config = McpServerConfig.parse("files", {"command": "npx", "args": args})

    args.append("--token")
    args[0] = "mutated"

    assert config.args == [
        "-y",
        "files-mcp",
    ], "the config holds its own copy of argv, so later mutation cannot reach it"
    assert config.args is not args, "parse must not alias the caller's list"


def test_parse_copies_the_callers_env_and_headers_mappings():
    """Mutation of the caller's mappings after parse must not reach the config."""
    env = {"MCP_TOKEN": "seed"}
    headers = {"Authorization": SECRET_HEADER}
    config = McpServerConfig.parse(
        "files", {"command": "npx", "env": env, "headers": headers}
    )

    env["INJECTED"] = "later"
    headers["X-Added"] = "later"

    assert config.env == {
        "MCP_TOKEN": "seed"
    }, "the config holds its own copy of env, so later mutation cannot reach it"
    assert config.headers == {
        "Authorization": SECRET_HEADER
    }, "the config holds its own copy of headers, so later mutation cannot reach it"
    assert config.env is not env, "parse must not alias the caller's env mapping"
    assert (
        config.headers is not headers
    ), "parse must not alias the caller's headers mapping"


def test_mcp_server_config_is_frozen():
    """A validated config is shared, so its fields must not be assignable."""
    config = McpServerConfig.parse("weather", {"url": HTTP_ENDPOINT})

    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'timeout'"):
        config.timeout = 1.0


# --------------------------------------------------------------------------
# parse() — the entry itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [None, 7, "", "   ", "\t\n"],
    ids=["none", "int", "empty", "spaces", "whitespace"],
)
def test_parse_rejects_a_blank_or_non_string_name(bad_name):
    """A blank name would namespace tools as '', so it must be refused."""
    with pytest.raises(MitoCodeMcpConfigError, match="name must be a non-empty string"):
        McpServerConfig.parse(bad_name, {"url": HTTP_ENDPOINT})


@pytest.mark.parametrize(
    ("bad_raw", "expected_type"),
    [
        (["https://x/mcp"], "list"),
        ("https://x/mcp", "str"),
        (None, "NoneType"),
        (7, "int"),
    ],
    ids=["list", "str", "none", "int"],
)
def test_parse_rejects_an_entry_that_is_not_a_mapping(bad_raw, expected_type):
    """A non-mapping entry must be refused, naming the type that arrived."""
    with pytest.raises(
        MitoCodeMcpConfigError, match=f"entry must be an object, got {expected_type}$"
    ):
        McpServerConfig.parse("bad", bad_raw)


def test_parse_error_message_lists_entry_keys_without_echoing_values():
    """'got keys [...]' may name keys but must never name their values."""
    with pytest.raises(
        MitoCodeMcpConfigError, match=r"got keys \['MCP_TOKEN'\]"
    ) as excinfo:
        McpServerConfig.parse("files", {"MCP_TOKEN": SECRET_ENV})

    message = str(excinfo.value)
    assert (
        SECRET_ENV not in message
    ), f"the missing-selector message leaked a credential: {message!r}"


# --------------------------------------------------------------------------
# parse() — the transport selector
# --------------------------------------------------------------------------


def test_parse_rejects_a_url_without_a_scheme():
    """A bare host has no scheme, so the detail line says so without the value."""
    with pytest.raises(
        MitoCodeMcpConfigError, match="got no http:// or https:// scheme"
    ):
        McpServerConfig.parse("bad", {"url": "weather.example.com/mcp"})


@pytest.mark.parametrize(
    ("bad_url", "expected_type"),
    [
        (42, "int"),
        (None, "NoneType"),
        ([HTTP_ENDPOINT], "list"),
        ({"url": HTTP_ENDPOINT}, "dict"),
    ],
    ids=["int", "none", "list", "dict"],
)
def test_parse_rejects_a_non_string_url(bad_url, expected_type):
    """A non-string url must be refused, naming the type that arrived."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match=(
            "'url' must be an http:// or https:// endpoint, " f"got {expected_type}$"
        ),
    ):
        McpServerConfig.parse("bad", {"url": bad_url})


@pytest.mark.parametrize(
    "bad_command",
    ["", "   ", None, 7, ["npx"]],
    ids=["empty", "blank", "none", "int", "list"],
)
def test_parse_rejects_a_blank_or_non_string_command(bad_command):
    """An empty command would build a transport that can never spawn."""
    with pytest.raises(
        MitoCodeMcpConfigError, match="'command' must be a non-empty string"
    ):
        McpServerConfig.parse("bad", {"command": bad_command})


# --------------------------------------------------------------------------
# parse() — the optional fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_enabled", "expected_repr"),
    [("yes", "'yes'"), (1, "1"), (0, "0"), (None, "None")],
    ids=["str", "int-one", "int-zero", "none"],
)
def test_parse_rejects_a_non_boolean_enabled(bad_enabled, expected_repr):
    """'enabled' is a flag; ints and strings are typos, not truthy values."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match=f"'enabled' must be a boolean, got {expected_repr}$",
    ):
        McpServerConfig.parse("weather", {"url": HTTP_ENDPOINT, "enabled": bad_enabled})


@pytest.mark.parametrize(
    ("bad_cwd", "expected_repr"),
    [
        (5, "5"),
        (None, "None"),
        (["/tmp"], "['/tmp']"),
        ({"dir": "/tmp"}, "{'dir': '/tmp'}"),
    ],
    ids=["int", "none", "list", "dict"],
)
def test_parse_rejects_a_non_string_cwd(bad_cwd, expected_repr):
    """A non-string cwd must be refused, echoing the value that arrived.

    ``cwd`` is not secret-bearing, so unlike ``env``/``headers`` its value is
    rendered into the message on purpose: a wrong path is what makes the typo
    findable.
    """
    with pytest.raises(
        MitoCodeMcpConfigError,
        match=f"'cwd' must be a string, got {re.escape(expected_repr)}$",
    ):
        McpServerConfig.parse("files", {"command": "npx", "cwd": bad_cwd})


@pytest.mark.parametrize("field", ["timeout", "connect_timeout"])
def test_parse_rejects_a_boolean_timeout(field):
    """bool subclasses int, so 'timeout: true' must not become a 1s timeout."""
    with pytest.raises(
        MitoCodeMcpConfigError, match=rf"'{field}' must be a positive number, got True"
    ):
        McpServerConfig.parse("files", {"command": "npx", field: True})


@pytest.mark.parametrize("field", ["timeout", "connect_timeout"])
def test_parse_rejects_a_nan_timeout(field):
    """NaN is not greater than zero, so it must be refused rather than stored."""
    with pytest.raises(
        MitoCodeMcpConfigError, match=rf"'{field}' must be a positive number, got nan"
    ):
        McpServerConfig.parse("files", {"command": "npx", field: float("nan")})


def test_parse_rejects_a_non_numeric_timeout():
    """A quoted timeout is a typo, not a string that coerces later."""
    with pytest.raises(
        MitoCodeMcpConfigError, match="'timeout' must be a positive number, got '30'"
    ):
        McpServerConfig.parse("files", {"command": "npx", "timeout": "30"})


def test_parse_coerces_an_integer_timeout_to_a_float():
    """An int timeout is accepted, but the field is documented to hold a float."""
    config = McpServerConfig.parse("files", {"command": "npx", "timeout": 90})

    assert config.timeout == 90.0, "the integer 90 must read back as 90.0"
    assert isinstance(
        config.timeout, float
    ), "an int timeout must be normalised to float so callers need no int check"


# --------------------------------------------------------------------------
# Secret-bearing fields — nothing may reach the exception message
# --------------------------------------------------------------------------


def test_parse_error_message_never_echoes_argv_supplied_as_a_bare_string():
    """'args: "--token <secret>"' is refused by type alone, value untouched."""
    with pytest.raises(
        MitoCodeMcpConfigError, match="'args' must be a list of strings, got str"
    ) as excinfo:
        McpServerConfig.parse(
            "files", {"command": "npx", "args": f"--token {SECRET_ARGV}"}
        )

    message = str(excinfo.value)
    assert (
        SECRET_ARGV not in message
    ), f"the not-a-list argv message leaked a credential: {message!r}"


def test_parse_error_message_never_echoes_env_supplied_as_a_bare_string():
    """'env: "TOKEN=<secret>"' is refused by type alone, value untouched."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match="'env' must be an object mapping strings to strings, got str",
    ) as excinfo:
        McpServerConfig.parse(
            "files", {"command": "npx", "env": f"MCP_TOKEN={SECRET_ENV}"}
        )

    message = str(excinfo.value)
    assert (
        SECRET_ENV not in message
    ), f"the not-a-mapping env message leaked a credential: {message!r}"


@pytest.mark.parametrize("field", ["env", "headers"])
def test_parse_error_message_names_a_non_string_entry_key_but_not_its_value(field):
    """The offending key is named so the typo is fixable; the value is not."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match=rf"'{field}' entry 5 must map a string to a string",
    ) as excinfo:
        McpServerConfig.parse("files", {"command": "npx", field: {5: SECRET_ENV}})

    message = str(excinfo.value)
    assert (
        SECRET_ENV not in message
    ), f"the bad-entry-key message for '{field}' leaked a credential: {message!r}"


def test_parse_error_message_never_echoes_a_scheme_less_url():
    """The no-scheme detail is a fixed sentence, so the url cannot leak."""
    with pytest.raises(
        MitoCodeMcpConfigError, match="got no http:// or https:// scheme"
    ) as excinfo:
        McpServerConfig.parse("bad", {"url": f"//user:{SECRET_URL}@x/mcp"})

    message = str(excinfo.value)
    assert (
        SECRET_URL not in message
    ), f"the no-scheme url message leaked a credential: {message!r}"


def test_parse_config_error_message_lists_top_level_keys_without_echoing_values():
    """The missing-key detail names keys only, never the document's values."""
    with pytest.raises(
        MitoCodeMcpConfigError, match=r"got keys \['mcpToken'\]"
    ) as excinfo:
        McpServerConfig.parse_config({"mcpToken": SECRET_HEADER})

    message = str(excinfo.value)
    assert (
        SECRET_HEADER not in message
    ), f"the missing-'mcpServers' message leaked a credential: {message!r}"


# --------------------------------------------------------------------------
# parse_config()
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_config", "expected_type"),
    [([], "list"), ("mcpServers", "str"), (None, "NoneType"), (7, "int")],
    ids=["list", "str", "none", "int"],
)
def test_parse_config_rejects_a_document_that_is_not_a_mapping(
    bad_config, expected_type
):
    """The whole document must be a mapping before its keys mean anything."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match=(
            "must be an object holding an 'mcpServers' mapping, "
            f"got {expected_type}$"
        ),
    ):
        McpServerConfig.parse_config(bad_config)


@pytest.mark.parametrize(
    ("bad_servers", "expected_type"),
    [([], "list"), ("weather", "str"), (None, "NoneType")],
    ids=["list", "str", "none"],
)
def test_parse_config_rejects_mcp_servers_that_is_not_a_mapping(
    bad_servers, expected_type
):
    """'mcpServers' must itself be a mapping of entries."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match=f"'mcpServers' must be an object, got {expected_type}$",
    ):
        McpServerConfig.parse_config({"mcpServers": bad_servers})


def test_parse_config_returns_an_empty_list_for_an_empty_servers_mapping():
    """A document with no servers is valid, not an error."""
    assert McpServerConfig.parse_config({"mcpServers": {}}) == []


def test_parse_config_surfaces_the_failing_entry_name():
    """An entry that parse() rejects must fail the whole document, by name."""
    with pytest.raises(
        MitoCodeMcpConfigError,
        match="MCP server 'broken': 'url' and 'command' are mutually exclusive",
    ):
        McpServerConfig.parse_config(
            {
                "mcpServers": {
                    "weather": {"url": HTTP_ENDPOINT},
                    "broken": {"url": HTTP_ENDPOINT, "command": "npx"},
                }
            }
        )


# --------------------------------------------------------------------------
# build_transport()
# --------------------------------------------------------------------------


def test_build_transport_rejects_an_unknown_transport_kind():
    """Only direct construction can reach the guard, since parse sets the kind."""
    config = McpServerConfig(name="legacy", kind="sse", url=HTTP_ENDPOINT)

    with pytest.raises(
        MitoCodeMcpConfigError,
        match="MCP server 'legacy': unknown transport kind 'sse'",
    ):
        config.build_transport()


def test_build_transport_passes_none_for_an_unset_env_and_cwd():
    """Unset stdio options must reach fastmcp as None, not as {} or ''."""
    config = McpServerConfig.parse("files", {"command": "npx", "args": []})

    transport = config.build_transport()

    assert (
        transport.env is None
    ), "an empty env must be handed over as 'unset' (None), not as {}"
    assert (
        transport.cwd is None
    ), "an empty cwd must be handed over as 'unset' (None), not as ''"


def test_build_transport_carries_populated_env_and_cwd_into_the_stdio_transport():
    """Configured stdio options must survive the hand-off to fastmcp."""
    config = McpServerConfig.parse(
        "files",
        {
            "command": "npx",
            "args": ["-y", "f"],
            "env": {"MCP_TOKEN": "seed"},
            "cwd": "/tmp",
        },
    )

    transport = config.build_transport()

    assert transport.command == "npx", "the command must reach the transport"
    assert transport.args == ["-y", "f"], "the argv must reach the transport"
    assert transport.env == {"MCP_TOKEN": "seed"}, "env must reach the transport"
    assert transport.cwd == "/tmp", "cwd must reach the transport"


def test_build_transport_passes_none_for_unset_headers():
    """fastmcp stores headers as {}, so only the call site shows 'unset'."""
    config = McpServerConfig.parse("weather", {"url": HTTP_ENDPOINT})

    with patch.object(mcp_config, "StreamableHttpTransport") as transport_cls:
        config.build_transport()

    transport_cls.assert_called_once_with(url=HTTP_ENDPOINT, headers=None)
