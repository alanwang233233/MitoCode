import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.backend.mcp.mcp_client import McpClient, McpConnectionState
from mitocode.backend.mcp.mcp_config import McpServerConfig
from mitocode.backend.mcp.mcp_manager import McpManager, McpServerStatus
from mitocode.backend.tool.tool_manager import ToolManager


class FakeClient:
    """Stand-in for McpClient whose connect outcome is scripted.

    ``state`` tracks the real client's three states rather than inferring one
    from ``connected``, so a client that has just been stopped stays
    distinguishable from one that never came up.  ``status()`` reports that
    difference and the manager's contract rests on it.
    """

    def __init__(self, config, succeed=True, tools=()):
        """Script one connect outcome and the tool catalogue it advertises.

        Args:
            config: The ``McpServerConfig`` this stand-in represents.
            succeed: Value :meth:`connect` reports back.
            tools: Names of the tools this stand-in advertises.
        """
        self._config = config
        self._succeed = succeed
        self.server_name = config.name
        self.tools = [
            SimpleNamespace(
                name=name,
                description=f"{name} on {config.name}",
                input_schema={"type": "object", "properties": {}},
            )
            for name in tools
        ]
        self.state = McpConnectionState.DISCONNECTED
        self.last_error = ""
        self.connected = False
        self.calls = 0

    async def connect(self):
        """Move to CONNECTED or FAILED according to the scripted outcome."""
        self.connected = self._succeed
        if self._succeed:
            self.state = McpConnectionState.CONNECTED
            self.last_error = ""
        else:
            self.state = McpConnectionState.FAILED
            self.last_error = "connection refused"
        return self._succeed

    async def disconnect(self):
        """Close the connection and go back to DISCONNECTED."""
        self.connected = False
        self.state = McpConnectionState.DISCONNECTED

    def go_offline(self):
        """Make every later connect attempt fail, as a server going away would."""
        self._succeed = False

    async def call_tool(self, tool_name, arguments):
        """Record the call and render it, as the real client would."""
        self.calls += 1
        return f"{tool_name}:{arguments}"


class RudeClient(FakeClient):
    """FakeClient that raises from one chosen method.

    The real :class:`McpClient` swallows its own failures and reports them, so
    this stand-in is how a test reaches the manager's own guard against a
    client that breaks its contract instead.
    """

    def __init__(self, config, fail_on=""):
        """Arm one method to raise.

        Args:
            config: The ``McpServerConfig`` this stand-in represents.
            fail_on: ``"connect"``, ``"disconnect"`` or ``""`` for neither.
        """
        super().__init__(config, succeed=fail_on != "connect")
        self._fail_on = fail_on

    async def connect(self):
        """Raise instead of connecting when this client is armed for it."""
        if self._fail_on == "connect":
            raise RuntimeError("handshake exploded")
        return await super().connect()

    async def disconnect(self):
        """Raise instead of closing when this client is armed for it."""
        if self._fail_on == "disconnect":
            raise RuntimeError("teardown exploded")
        await super().disconnect()


def _factory(script):
    """Build a client_factory driven by a {server_name: (succeed, tools)} map.

    Args:
        script: Mapping from server name to its scripted outcome.  A server
            the script does not mention connects successfully with no tools.

    Returns:
        The factory, with every client it built exposed as ``made``.
    """
    made = {}

    def _make(config):
        succeed, tools = script.get(config.name, (True, ()))
        client = FakeClient(config, succeed=succeed, tools=tools)
        made[config.name] = client
        return client

    _make.made = made
    return _make


def _rude_factory(script):
    """Build a client_factory whose clients raise from a scripted method.

    Args:
        script: Mapping from server name to the method it raises from, or
            ``""`` for a well-behaved client.

    Returns:
        The factory, with every client it built exposed as ``made``.
    """
    made = {}

    def _make(config):
        client = RudeClient(config, fail_on=script.get(config.name, ""))
        made[config.name] = client
        return client

    _make.made = made
    return _make


def _config(*entries):
    """Build a ``{"mcpServers": {...}}`` document from (name, spec) pairs."""
    return {"mcpServers": {name: spec for name, spec in entries}}


@pytest.mark.asyncio
async def test_start_registers_tools_from_every_server():
    tool_manager = ToolManager()
    factory = _factory(
        {
            "weather": (True, ("get_forecast",)),
            "files": (True, ("read",)),
        }
    )
    manager = McpManager(
        _config(
            ("weather", {"url": "https://w.invalid/mcp"}),
            ("files", {"command": "npx", "args": ["files-mcp"]}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert "mcp__weather__get_forecast" in tool_manager.tools_map()
    assert "mcp__files__read" in tool_manager.tools_map()


@pytest.mark.asyncio
async def test_start_isolates_failures_across_servers():
    tool_manager = ToolManager()
    factory = _factory(
        {
            "broken": (False, ()),
            "healthy": (True, ("ping",)),
        }
    )
    manager = McpManager(
        _config(
            ("broken", {"url": "https://broken.invalid/mcp"}),
            ("healthy", {"url": "https://healthy.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert "mcp__healthy__ping" in tool_manager.tools_map()
    assert tool_manager.tools_map().get("mcp__broken__ping") is None
    assert manager.status()["broken"].error != ""
    assert manager.status()["broken"].state == "failed"
    assert manager.status()["healthy"].state == "connected"


@pytest.mark.asyncio
async def test_status_reports_state_error_and_tool_count():
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a", "b"))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    status = manager.status()["weather"]
    assert isinstance(status, McpServerStatus)
    assert status.state == "connected"
    assert status.error == ""
    assert status.tool_count == 2


@pytest.mark.asyncio
async def test_stop_unregisters_tools_and_disconnects():
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("get_forecast",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()
    await manager.stop()

    assert "mcp__weather__get_forecast" not in tool_manager.tools_map()
    assert factory.made["weather"].connected is False
    assert manager.status()["weather"].state == "disconnected"
    assert manager.status()["weather"].tool_count == 0


@pytest.mark.asyncio
async def test_refresh_picks_up_new_tools():
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()
    assert "mcp__weather__a" in tool_manager.tools_map()

    factory.made["weather"].tools.append(
        SimpleNamespace(
            name="b",
            description="b",
            input_schema={"type": "object", "properties": {}},
        )
    )
    await manager.refresh("weather")

    assert "mcp__weather__a" in tool_manager.tools_map()
    assert "mcp__weather__b" in tool_manager.tools_map()


@pytest.mark.asyncio
async def test_refresh_rejects_unknown_server_name():
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=ToolManager(),
        client_factory=_factory({}),
    )

    with pytest.raises(MitoCodeMcpConfigError):
        await manager.refresh("nope")


@pytest.mark.asyncio
async def test_disabled_server_is_not_connected():
    tool_manager = ToolManager()
    factory = _factory({"off": (True, ("x",))})
    manager = McpManager(
        _config(("off", {"url": "https://off.invalid/mcp", "enabled": False})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert "off" not in factory.made
    assert tool_manager.tools_map() == {}
    assert manager.status()["off"].state == "disabled"
    assert manager.status()["off"].tool_count == 0
    assert manager.status()["off"].error == ""


@pytest.mark.asyncio
async def test_start_does_not_raise_when_a_server_connect_raises():
    """A client that breaks its own contract must not stop the other servers.

    The real ``McpClient`` swallows its own failures and returns ``False``, so
    this reaches the manager's own guard against a client that raises instead
    of reporting.  The caller still gets a return, and the healthy server still
    publishes its tools.
    """
    tool_manager = ToolManager()
    factory = _rude_factory({"bad": "connect", "good": ""})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert manager.status()["good"].state == "connected"
    assert "RuntimeError" in manager.status()["bad"].error
    assert "handshake exploded" in manager.status()["bad"].error
    assert manager.status()["bad"].state != "connected"
    assert (
        manager.status()["bad"].tool_count == 0
    ), "a server that raised must not look like one that advertised no tools"


@pytest.mark.asyncio
async def test_start_does_not_raise_when_a_registration_raises(monkeypatch):
    """A catalogue the registry cannot digest must cost only its own server.

    This is the seam the manager cannot otherwise reach: the real client
    reports its own failures, and a registration is the manager's own code
    raising underneath it.
    """
    tool_manager = ToolManager()
    real_register = tool_manager.register_mcp_client

    def _register(client):
        if client.server_name == "bad":
            raise ValueError("catalogue exploded")
        return real_register(client)

    monkeypatch.setattr(tool_manager, "register_mcp_client", _register)
    factory = _factory({"bad": (True, ("a",)), "good": (True, ("g",))})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert set(tool_manager.tools_map()) == {
        "mcp__good__g"
    }, "the healthy server still publishes its tools"
    assert "catalogue exploded" in manager.status()["bad"].error
    assert manager.status()["bad"].tool_count == 0


@pytest.mark.asyncio
async def test_stop_disconnects_every_server_when_one_teardown_raises():
    """One failing teardown must not strand the other connections open."""
    tool_manager = ToolManager()
    factory = _rude_factory({"bad": "disconnect", "good": ""})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    await manager.stop()

    assert factory.made["good"].state is McpConnectionState.DISCONNECTED
    assert tool_manager.tools_map() == {}
    assert "RuntimeError" in manager.status()["bad"].error


@pytest.mark.asyncio
async def test_start_twice_then_stop_leaves_no_registrations():
    """A repeated start must not orphan the tools the first one registered.

    ``ToolManager.register_mcp_client`` never overwrites an existing name, so
    a second registration pass registers nothing and empties the manager's
    record of what it owns.  ``stop()`` would then have no name left to
    unregister and the first pass's tools would outlive the session.
    """
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()
    await manager.start()
    assert list(tool_manager.tools_map()) == ["mcp__weather__a"]

    await manager.stop()

    assert (
        tool_manager.tools_map() == {}
    ), "every tool a repeated start registered must still be unregistered"
    assert manager.status()["weather"].tool_count == 0


@pytest.mark.asyncio
async def test_refresh_drops_tools_the_server_no_longer_advertises():
    """Refreshing must pick up the whole catalogue, not only its additions."""
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a", "b"))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    assert set(tool_manager.tools_map()) == {"mcp__weather__a", "mcp__weather__b"}

    factory.made["weather"].tools.pop()
    await manager.refresh("weather")

    assert set(tool_manager.tools_map()) == {
        "mcp__weather__a"
    }, "a tool the server dropped must not stay dispatchable to a dead client"


@pytest.mark.asyncio
async def test_refresh_drops_stale_tools_when_the_reconnect_fails():
    """A failed refresh leaves that server with no tools, not with dead ones.

    The tools must be unregistered *before* the reconnect is attempted: kept
    across a failure they would advertise a capability no model can call, and
    would dispatch into a client that is not connected.
    """
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a", "b"))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    assert set(tool_manager.tools_map()) == {"mcp__weather__a", "mcp__weather__b"}

    factory.made["weather"].go_offline()
    await manager.refresh("weather")

    assert (
        tool_manager.tools_map() == {}
    ), "a server that could not be reached must publish nothing"
    assert manager.status()["weather"].state == "failed"


@pytest.mark.asyncio
async def test_refresh_without_a_name_reconnects_every_server():
    tool_manager = ToolManager()
    factory = _factory({"one": (True, ("a",)), "two": (True, ("b",))})
    manager = McpManager(
        _config(
            ("one", {"url": "https://one.invalid/mcp"}),
            ("two", {"url": "https://two.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    await manager.refresh()

    assert factory.made["one"].state is McpConnectionState.CONNECTED
    assert factory.made["two"].state is McpConnectionState.CONNECTED
    assert set(tool_manager.tools_map()) == {"mcp__one__a", "mcp__two__b"}


@pytest.mark.asyncio
async def test_refresh_of_a_disabled_server_is_rejected_by_name():
    """A disabled server has no client, so the rejection must say why."""
    manager = McpManager(
        _config(("off", {"url": "https://off.invalid/mcp", "enabled": False})),
        tool_manager=ToolManager(),
        client_factory=_factory({}),
    )

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        await manager.refresh("off")

    assert "off" in str(excinfo.value)
    assert "disabled" in str(excinfo.value)


@pytest.mark.asyncio
async def test_refresh_error_names_the_configured_servers():
    manager = McpManager(
        _config(
            ("weather", {"url": "https://w.invalid/mcp"}),
            ("files", {"url": "https://f.invalid/mcp"}),
        ),
        tool_manager=ToolManager(),
        client_factory=_factory({}),
    )

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        await manager.refresh("nope")

    message = str(excinfo.value)
    assert "nope" in message
    assert "weather" in message and "files" in message


@pytest.mark.asyncio
async def test_refresh_error_does_not_echo_server_credentials():
    """The message names a server, never the configuration behind it."""
    manager = McpManager(
        _config(
            (
                "files",
                {
                    "url": "https://files.invalid/mcp",
                    "headers": {"Authorization": "Bearer s3cr3t-token"},
                },
            )
        ),
        tool_manager=ToolManager(),
        client_factory=_factory({}),
    )

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        await manager.refresh("nope")

    assert "s3cr3t-token" not in str(excinfo.value)


def test_a_list_of_configs_is_accepted_as_the_configuration():
    """Configurations normalised elsewhere need no round trip through JSON."""
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a",))})
    manager = McpManager(
        [McpServerConfig.parse("weather", {"url": "https://w.invalid/mcp"})],
        tool_manager=tool_manager,
        client_factory=factory,
    )

    assert "weather" in factory.made
    assert manager.status()["weather"].state == "disconnected"
    assert manager.status()["weather"].tool_count == 0


def test_a_configuration_that_is_neither_a_document_nor_a_list_is_rejected():
    with pytest.raises(MitoCodeMcpConfigError):
        McpManager(
            "mcpServers", tool_manager=ToolManager(), client_factory=_factory({})
        )


def test_a_list_entry_that_is_not_a_config_is_rejected():
    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpManager(
            [{"url": "https://w.invalid/mcp"}],
            tool_manager=ToolManager(),
            client_factory=_factory({}),
        )

    assert "0" in str(excinfo.value), "the offending position is named"


def test_an_invalid_server_entry_is_reported_before_any_client_is_built():
    """Configuration is validated as a whole, so no half-built manager escapes."""
    factory = _factory({"good": (True, ("a",))})
    config = {
        "mcpServers": {
            "good": {"url": "https://good.invalid/mcp"},
            "bad": {"url": "not-a-url"},
        }
    }

    with pytest.raises(MitoCodeMcpConfigError):
        McpManager(config, tool_manager=ToolManager(), client_factory=factory)

    assert factory.made == {}, "a rejected configuration connects to nothing"


@pytest.mark.asyncio
async def test_start_after_stop_brings_the_tools_back():
    """Stopping is a pause, not a one-way door: the same manager restarts."""
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    await manager.stop()
    assert tool_manager.tools_map() == {}

    await manager.start()

    assert set(tool_manager.tools_map()) == {"mcp__weather__a"}
    assert manager.status()["weather"].state == "connected"
    assert manager.status()["weather"].tool_count == 1


@pytest.mark.asyncio
async def test_start_with_no_servers_at_all_does_nothing():
    tool_manager = ToolManager()
    manager = McpManager(
        {"mcpServers": {}}, tool_manager=tool_manager, client_factory=_factory({})
    )

    await manager.start()

    assert manager.status() == {}
    assert tool_manager.tools_map() == {}


@pytest.mark.asyncio
async def test_status_before_start_reports_a_disconnected_server():
    """A manager that has not started yet says so rather than looking idle."""
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=ToolManager(),
        client_factory=_factory({"weather": (True, ("a",))}),
    )

    status = manager.status()["weather"]

    assert status.state == "disconnected"
    assert status.tool_count == 0
    assert status.error == ""


def test_mcp_package_exports_the_manager_and_its_neighbours():
    from mitocode.backend import mcp

    assert mcp.McpManager is McpManager
    assert mcp.McpServerStatus is McpServerStatus
    assert mcp.McpClient is McpClient
    assert mcp.McpConnectionState is McpConnectionState
    assert mcp.McpServerConfig is McpServerConfig


@pytest.mark.asyncio
async def test_local_stdio_server_end_to_end(tmp_path):
    """The local-server path, proven against a real subprocess.

    No ``client_factory`` here: this is the only test in the feature that
    exercises the whole chain — a real ``McpClient``, a real stdio transport
    and a real ``sys.executable`` child serving over MCP — so it is what
    justifies the transport plumbing underneath it.
    """
    server_script = tmp_path / "srv.py"
    server_script.write_text(
        "from fastmcp import FastMCP\n"
        "app = FastMCP('local')\n"
        "@app.tool\n"
        "def shout(word: str) -> str:\n"
        "    '''Shout a word.'''\n"
        "    return word.upper()\n"
        "app.run()\n"
    )

    tool_manager = ToolManager()
    manager = McpManager(
        {
            "mcpServers": {
                "local": {
                    "command": sys.executable,
                    "args": [str(server_script)],
                    "connect_timeout": 60.0,
                }
            }
        },
        tool_manager=tool_manager,
    )

    await manager.start()
    try:
        status = manager.status()["local"]
        assert status.state == "connected", status.error
        assert status.tool_count == 1
        assert "mcp__local__shout" in tool_manager.tools_map()

        result = await tool_manager.tools_map()["mcp__local__shout"](word="hey")
        assert result == "HEY"
    finally:
        await manager.stop()

    assert tool_manager.tools_map() == {}
    assert manager.status()["local"].state == "disconnected"
