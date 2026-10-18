import asyncio
import contextlib
import dataclasses
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.core.exception.mitocode_exception import MitoCodeMcpConfigError
from mitocode.core.mcp.mcp_client import McpClient, McpConnectionState
from mitocode.core.mcp.mcp_config import McpServerConfig, McpServerKind
from mitocode.core.mcp.mcp_manager import McpManager, McpServerStatus
from mitocode.core.tool.tool_manager import ToolManager

#: Imports the ``mcp`` package in a process where the tool layer cannot be
#: resolved at all, so the one-way ``mcp`` -> (never) ``tool_manager``
#: dependency is checked for real rather than by reading the source.
#:
#: Both ``mitocode`` and ``mitocode.core`` are stubbed as bare packages:
#: their real ``__init__`` files re-export the agent layer, which imports
#: ``tool_manager`` itself, so a real import of either parent would prove
#: nothing about the module under test.  The stubs carry only ``__path__``,
#: which is what lets ``mitocode.core.mcp`` load from disk as usual.
_IMPORT_WITHOUT_TOOL_LAYER = """
import importlib.abc
import os
import sys
import types

SRC_DIR = os.environ["MITOCODE_SRC"]


class _BlockToolLayer(importlib.abc.MetaPathFinder):
    \"\"\"Make the whole tool layer unimportable, however it is reached.\"\"\"

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("mitocode.core.tool"):
            raise ImportError(f"{fullname} is not importable here")
        return None


def _stub_package(name, path):
    \"\"\"Register a package that never runs its real __init__.

    Args:
        name: Fully qualified package name.
        path: Directory its submodules live in.
    \"\"\"
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module


_stub_package("mitocode", SRC_DIR)
_stub_package("mitocode.core", os.path.join(SRC_DIR, "mitocode", "core"))
sys.meta_path.insert(0, _BlockToolLayer())

import mitocode.core.mcp as mcp


def leaked():
    \"\"\"List the tool-layer modules this import dragged in.\"\"\"
    return sorted(m for m in sys.modules if m.startswith("mitocode.core.tool"))


assert not leaked(), f"importing the mcp package pulled in the tool layer: {leaked()}"


class _StubRegistry:
    \"\"\"Duck-typed stand-in for ToolManager, to drive the manager offline.\"\"\"

    def register_mcp_client(self, client):
        \"\"\"Register nothing and report nothing registered.\"\"\"
        return []

    def unregister(self, name):
        \"\"\"Report a removal that never happened.\"\"\"
        return False


# The manager must work with anything exposing those two methods: that is what
# lets the mcp package stay unaware of the tool layer.
manager = mcp.McpManager(
    {"mcpServers": {"w": {"url": "https://w.invalid/mcp"}}},
    tool_manager=_StubRegistry(),
)
assert manager.status()["w"].state == "disconnected"
print("ok")
"""


class FakeClient:
    """Stand-in for McpClient whose connect outcome is scripted.

    ``state`` tracks the real client's three states rather than inferring one
    from ``connected``, so a client that has just been stopped stays
    distinguishable from one that never came up.  ``status()`` reports that
    difference and the manager's contract rests on it.

    The catalogue follows the real client's rule that a client which is not
    connected has no tools: it is emptied on construction when the scripted
    outcome is a failure, and again by every failed :meth:`connect`.
    """

    def __init__(self, config, succeed=True, tools=(), keep_tools_on_failure=False):
        """Script one connect outcome and the tool catalogue it advertises.

        Args:
            config: The ``McpServerConfig`` this stand-in represents.
            succeed: Value :meth:`connect` reports back.
            tools: Names of the tools this stand-in advertises once connected.
            keep_tools_on_failure: Leave the catalogue populated even while
                not connected.  Deliberately unfaithful: it is how a test
                reaches an implementation that registers before connecting,
                which the real client would not let it get away with.
        """
        self._config = config
        self._succeed = succeed
        self.keep_tools_on_failure = keep_tools_on_failure
        self._cancel_next = False
        self.server_name = config.name
        self.tools = [
            SimpleNamespace(
                name=name,
                description=f"{name} on {config.name}",
                input_schema={"type": "object", "properties": {}},
            )
            for name in tools
        ]
        if not succeed and not keep_tools_on_failure:
            self.tools.clear()
        self.state = McpConnectionState.DISCONNECTED
        self.last_error = ""
        self.connected = False
        self.calls = 0
        self.attempts = 0

    async def connect(self):
        """Move to CONNECTED or FAILED according to the scripted outcome.

        One-shot cancellation is honoured first, so a test can arm it *after*
        a successful ``start`` rather than making every later connect fail.
        ``attempts`` counts every entry, so a test can tell "never given a turn"
        from "given one and failed".

        Returns:
            The scripted outcome.

        Raises:
            asyncio.CancelledError: If :meth:`cancel_next_connect` armed this
                attempt, which is how ``McpClient.connect`` behaves when the
                caller cancels it.
        """
        self.attempts += 1
        if self._cancel_next:
            self._cancel_next = False
            raise asyncio.CancelledError()
        self.connected = self._succeed
        if self._succeed:
            self.state = McpConnectionState.CONNECTED
            self.last_error = ""
            return True
        # The real client drops its cached catalogue when a connect fails.
        if not self.keep_tools_on_failure:
            self.tools.clear()
        self.state = McpConnectionState.FAILED
        self.last_error = "connection refused"
        return False

    async def disconnect(self):
        """Close the connection and go back to DISCONNECTED."""
        self.connected = False
        self.state = McpConnectionState.DISCONNECTED

    def go_offline(self):
        """Make every later connect attempt fail, as a server going away would."""
        self._succeed = False

    def go_online(self):
        """Make every later connect attempt succeed, as a server coming back."""
        self._succeed = True

    def cancel_next_connect(self):
        """Cancel only the next connect, leaving any later one to succeed."""
        self._cancel_next = True

    async def call_tool(self, tool_name, arguments):
        """Record the call and render it, as the real client would."""
        self.calls += 1
        return f"{tool_name}:{arguments}"


class _ShutdownSignal(BaseException):
    """A shutdown request that is neither an ``Exception`` nor a cancellation.

    ``KeyboardInterrupt`` and ``SystemExit`` cannot be used to drive this
    because asyncio re-raises those out of the event loop itself rather than
    delivering them to the awaiting coroutine, which is the whole point here.
    """


class RudeClient(FakeClient):
    """FakeClient that raises from one chosen method.

    The real :class:`McpClient` swallows its own failures and reports them, so
    this stand-in is how a test reaches the manager's own guard against a
    client that breaks its contract instead.
    """

    def __init__(
        self,
        config,
        succeed=True,
        fail_on="",
        cancel_on="",
        signal_on="",
        message="",
        tools=(),
    ):
        """Arm one method to raise, independently of the scripted outcome.

        Args:
            config: The ``McpServerConfig`` this stand-in represents.
            succeed: Value :meth:`connect` reports back when it does not raise.
            fail_on: ``"connect"``, ``"disconnect"`` or ``""``.
            cancel_on: ``"connect"``, ``"disconnect"`` or ``""``, for a
                cancellation rather than a failure.  Checked first, because a
                cancelled teardown has to reach the caller while a failed one
                must be recorded and swallowed.
            signal_on: ``"connect"``, ``"disconnect"`` or ``""``, for a
                ``BaseException`` that is neither of those two.
            message: Message the raised failure carries, so a test can quote an
                endpoint the way a real handshake failure would.
            tools: Names this stand-in advertises once connected.
        """
        super().__init__(config, succeed=succeed, tools=tools)
        self._fail_on = fail_on
        self._cancel_on = cancel_on
        self._signal_on = signal_on
        self._message = message

    async def connect(self):
        """Raise instead of connecting when this client is armed for it.

        ``attempts`` counts every entry, whichever way this call ends, so a
        test can never read ``0`` from a client that was handed a turn.
        """
        if self._cancel_on == "connect":
            self.attempts += 1
            raise asyncio.CancelledError()
        if self._signal_on == "connect":
            self.attempts += 1
            raise _ShutdownSignal("application is shutting down")
        if self._fail_on == "connect":
            self.attempts += 1
            raise RuntimeError(self._message or "handshake exploded")
        return await super().connect()

    async def disconnect(self):
        """Raise instead of closing when this client is armed for it."""
        if self._cancel_on == "disconnect":
            raise asyncio.CancelledError()
        if self._signal_on == "disconnect":
            raise _ShutdownSignal("application is shutting down")
        if self._fail_on == "disconnect":
            raise RuntimeError(self._message or "teardown exploded")
        await super().disconnect()


class ExplodingCatalogueClient:
    """Stand-in whose tool catalogue cannot be read at all.

    The real :class:`McpClient` exposes a plain list, so the only way to reach
    the registry's ``client.tools`` access is to substitute a client whose
    attribute raises.  That is the seam between "the server said it was
    connected but its catalogue is unreadable" and a failure the manager can
    report against that server alone.
    """

    def __init__(self, config, succeed=True):
        """Connect on demand, with a catalogue that raises whenever it is read.

        Args:
            config: The ``McpServerConfig`` this stand-in represents.
            succeed: Value :meth:`connect` reports back.
        """
        self.server_name = config.name
        self._succeed = succeed
        self.state = McpConnectionState.DISCONNECTED
        self.last_error = ""
        self.connected = False

    @property
    def tools(self):
        """Raise instead of returning the catalogue.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Every time the registry asks for the catalogue.
        """
        raise RuntimeError("catalogue exploded on access")

    async def connect(self):
        """Connect without ever holding a readable catalogue."""
        self.connected = self._succeed
        if not self._succeed:
            self.state = McpConnectionState.FAILED
            self.last_error = "connection refused"
            return False
        self.state = McpConnectionState.CONNECTED
        return True

    async def disconnect(self):
        """Close the connection and go back to DISCONNECTED."""
        self.connected = False
        self.state = McpConnectionState.DISCONNECTED


def _mcp_tool(name, description):
    """Build one advertised tool of the shape the registry reads.

    Args:
        name: The tool name as advertised, without the namespace prefix.
        description: Whatever the server sent.  The malformed case sends a
            value that is not a string, which is what a server advertising a
            field of the wrong JSON type looks like from here.

    Returns:
        A ``SimpleNamespace`` carrying ``name``, ``description`` and
        ``input_schema``.
    """
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
    )


class MalformedCatalogueClient:
    """Stand-in advertising a healthy tool followed by an undescribable one.

    ``ToolManager.register_mcp_client`` reads every tool's description, so a
    catalogue whose *second* entry carries something that is not a string is
    what makes a registration fail part way through.  The first tool is
    entirely well formed, which is what gives the case its teeth: a registry
    that publishes as it reads leaves that tool behind, and a caller told
    "the registration failed" has no way to reclaim it.
    """

    #: A description type a well behaved server never sends.
    MALFORMED_DESCRIPTION = 123

    def __init__(self, config, healthy_tool="healthy"):
        """Advertise one readable tool and one that cannot be described.

        Args:
            config: The ``McpServerConfig`` this stand-in represents.
            healthy_tool: Name of the well formed tool.  A test sets it to a
                name another layer already owns to reach the collision path.
        """
        self.server_name = config.name
        self.tools = [
            _mcp_tool(healthy_tool, f"{healthy_tool} on {config.name}"),
            _mcp_tool("malformed", self.MALFORMED_DESCRIPTION),
        ]
        self.state = McpConnectionState.DISCONNECTED
        self.last_error = ""
        self.connected = False

    async def connect(self):
        """Connect; publishing is the registry's business, not the client's."""
        self.state = McpConnectionState.CONNECTED
        self.connected = True
        return True

    async def disconnect(self):
        """Close the connection and go back to DISCONNECTED."""
        self.connected = False
        self.state = McpConnectionState.DISCONNECTED


def _factory(script, keep_tools_on_failure=False):
    """Build a client_factory driven by a {server_name: (succeed, tools)} map.

    Args:
        script: Mapping from server name to its scripted outcome.  A server
            the script does not mention connects successfully with no tools.
        keep_tools_on_failure: Give every client a catalogue that survives a
            failed connect, for the registration-ordering probe.

    Returns:
        The factory, with every client it built exposed as ``made``.
    """
    made = {}

    def _make(config):
        succeed, tools = script.get(config.name, (True, ()))
        client = FakeClient(
            config,
            succeed=succeed,
            tools=tools,
            keep_tools_on_failure=keep_tools_on_failure,
        )
        made[config.name] = client
        return client

    _make.made = made
    return _make


def _rude_factory(script, succeed=None):
    """Build a client_factory whose clients raise from a scripted method.

    Args:
        script: Mapping from server name to the method it raises from, or
            ``""`` for a well-behaved client.
        succeed: Optional mapping from server name to the connect outcome used
            when that method does not raise, overriding the default of ``True``.

    Returns:
        The factory, with every client it built exposed as ``made``.
    """
    made = {}
    outcomes = succeed or {}

    def _make(config):
        client = RudeClient(
            config,
            succeed=outcomes.get(config.name, True),
            fail_on=script.get(config.name, ""),
        )
        made[config.name] = client
        return client

    _make.made = made
    return _make


def _hostile_factory(script):
    """Build a client_factory in which each server misbehaves in its own way.

    Args:
        script: Mapping from server name to its misbehaviour.  ``rude`` names the
            method that raises ``"connect"`` or ``"disconnect"``, ``cancel_on``
            names the one that raises :class:`asyncio.CancelledError` instead,
            ``catalogue_raises`` swaps in a client whose tool list cannot be
            read, ``malformed`` swaps in one whose second advertised tool
            cannot be described, ``signal_on`` names the one that raises a
            ``BaseException`` which is neither an ``Exception`` nor a
            cancellation, and ``succeed``/``tools``/``healthy_tool``/``message``
            configure the rest.  A server the script does not mention is well
            behaved.

    Returns:
        The factory, with every client it built exposed as ``made``.
    """
    made = {}

    def _make(config):
        spec = dict(script.get(config.name, {}))
        if spec.pop("malformed", False):
            client = MalformedCatalogueClient(
                config, healthy_tool=spec.pop("healthy_tool", "healthy")
            )
        elif spec.pop("catalogue_raises", False):
            client = ExplodingCatalogueClient(config, succeed=spec.pop("succeed", True))
        else:
            client = RudeClient(
                config,
                succeed=spec.pop("succeed", True),
                fail_on=spec.pop("rude", ""),
                cancel_on=spec.pop("cancel_on", ""),
                signal_on=spec.pop("signal_on", ""),
                message=spec.pop("message", ""),
                tools=spec.pop("tools", ()),
            )
        made[config.name] = client
        return client

    _make.made = made
    return _make


#: Values standing in for credentials, one per configuration field that can
#: carry one.  Each is unique so a failing assertion names *which* field leaked
#: rather than only that something did.
_SECRET_USERINFO = "pw-2d4f-secret"
_SECRET_HEADER = "Bearer 7c21-secret"
_SECRET_ENV = "env-5b8e-secret"
_SECRET_ARG = "--arg-9f3a-secret"
_SECRET_PATH = "/srv-6a0b-secret"

_SECRETS = (
    _SECRET_USERINFO,
    _SECRET_HEADER,
    _SECRET_ENV,
    _SECRET_ARG,
    _SECRET_PATH,
)


def _http_secret_spec(server_name):
    """A remote server entry carrying a secret in every field that takes one."""
    return {
        "url": f"https://user:{_SECRET_USERINFO}@{server_name}.invalid/mcp",
        "headers": {"Authorization": _SECRET_HEADER},
        "env": {"MCP_TOKEN": _SECRET_ENV},
        "args": [_SECRET_ARG],
        "cwd": _SECRET_PATH,
    }


def _stdio_secret_spec():
    """A local server entry carrying a secret in every field that takes one."""
    return {
        "command": _SECRET_PATH,
        "args": [_SECRET_ARG],
        "env": {"MCP_TOKEN": _SECRET_ENV},
        "cwd": _SECRET_PATH,
        "headers": {"Authorization": _SECRET_HEADER},
    }


def _assert_carries_no_secret(text, where):
    """Fail naming every credential value that reached *text*.

    Args:
        text: The message a caller can see -- the TUI pane or the log.
        where: What produced it, used only in the failure message.
    """
    leaked = [secret for secret in _SECRETS if secret in text]
    assert not leaked, f"{where} leaked {leaked}"


@contextlib.contextmanager
def _captured_logs():
    """Collect every loguru line emitted inside the block.

    The manager's own logging is half of what it promises an operator: a
    failure it records against a server is also named in the log, with the
    tool that could not be released.  ``status()`` deliberately carries less,
    so the two surfaces have to be checked separately.

    Yields:
        The list every emitted line is appended to.
    """
    lines = []
    sink_id = loguru_logger.add(lines.append, level="DEBUG")
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)


def _config(*entries):
    """Build a ``{"mcpServers": {...}}`` document from (name, spec) pairs."""
    return {"mcpServers": {name: spec for name, spec in entries}}


def _run_import_probe(script):
    """Run *script* in a child interpreter that cannot resolve the tool layer.

    ``MITOCODE_SRC`` is absolute and the child runs in an empty *cwd*, so
    ``python -c`` cannot pick up a stray ``mitocode/`` package that would
    shadow the module under test.

    Args:
        script: Source for the child interpreter.

    Returns:
        The completed ``subprocess.CompletedProcess``.
    """
    env = dict(os.environ)
    env["MITOCODE_SRC"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=env["MITOCODE_SRC"],
        check=False,
        # A wedged child must fail the test rather than hang the whole suite.
        timeout=120,
    )


@pytest.fixture
def stubborn_registry(monkeypatch):
    """Build a registry that can be told to refuse to release particular tools.

    ``unregister`` is the seam the manager's ownership record is written
    against, and "the removal raised" is the one outcome no client-side
    stand-in can produce, so every test that needs a tool to stay in the
    registry reaches it through here.

    Args:
        monkeypatch: Patched here and undone by pytest when the test ends.

    Returns:
        A factory taking the tool names to refuse.  It returns a
        ``(tool_manager, armed)`` pair; edit the ``armed`` set in place to arm
        or disarm the refusal at any point in the test.
    """

    def _build(*names):
        tool_manager = ToolManager()
        real_unregister = tool_manager.unregister
        armed = set(names)

        def _unregister(name):
            if name in armed:
                raise RuntimeError("registry exploded")
            return real_unregister(name)

        monkeypatch.setattr(tool_manager, "unregister", _unregister)
        return tool_manager, armed

    return _build


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
async def test_tools_are_registered_only_after_a_successful_connect():
    """A server that failed to connect must publish nothing.

    The client is armed with a catalogue it never connected well enough to
    discover, which the real client does not do — that is precisely the shape
    an implementation that registers *before* connecting would publish.
    """
    tool_manager = ToolManager()
    factory = _factory(
        {"broken": (False, ("ping",)), "healthy": (True, ("pong",))},
        keep_tools_on_failure=True,
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

    assert set(tool_manager.tools_map()) == {
        "mcp__healthy__pong"
    }, "only the connected server's tools may be registered"
    assert manager.status()["broken"].tool_count == 0
    assert (
        factory.made["broken"].tools != []
    ), "the probe catalogue survived, so it proves the point"


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
async def test_refresh_before_start_connects_the_server_it_names():
    """``refresh`` is not gated on ``start``: it brings up what it names.

    Documented rather than rejected, because refusing it would make a manager
    that is being reconnected after a crash refuse the one call meant to
    recover it.
    """
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.refresh("weather")

    assert factory.made["weather"].state is McpConnectionState.CONNECTED
    assert set(tool_manager.tools_map()) == {"mcp__weather__a"}
    assert manager.status()["weather"].tool_count == 1


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
async def test_start_records_a_cancelled_server_and_lets_the_cancellation_out():
    """A cancellation is recorded *and* propagated, never swallowed.

    ``return_exceptions=True`` hands a child's ``CancelledError`` back as a
    result rather than raising it, and ``except Exception`` cannot catch one,
    so without this the server would report a failure with no reason at all.
    """
    tool_manager = ToolManager()
    factory = _rude_factory({"bad": "", "good": ""})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    factory.made["bad"].cancel_next_connect()

    with pytest.raises(asyncio.CancelledError):
        await manager.start()

    assert manager.status()["bad"].error == "CancelledError"
    assert manager.status()["bad"].tool_count == 0
    assert (
        factory.made["good"].state is McpConnectionState.CONNECTED
    ), "the server that did connect still published its tools"


@pytest.mark.asyncio
async def test_start_keeps_owning_a_tool_it_could_not_unregister(stubborn_registry):
    """A second ``start`` must not forget a tool the registry refused to remove.

    ``register_mcp_client`` skips a name that is still registered, so its
    result omits the stuck tool.  Assigning that result to the ownership record
    would forget the tool while it is still in the registry — dispatchable,
    counted nowhere, and reclaimable by nobody.
    """
    tool_manager, armed = stubborn_registry()
    factory = _factory({"weather": (True, ("x",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    armed.add("mcp__weather__x")

    await manager.start()

    assert (
        manager.status()["weather"].tool_count == 1
    ), "the tool is still in the registry, so status() must still count it"
    assert "registry exploded" in manager.status()["weather"].error

    armed.clear()
    await manager.stop()

    assert (
        tool_manager.tools_map() == {}
    ), "a later stop() must reclaim the tool nobody forgot"


@pytest.mark.asyncio
async def test_start_owns_the_tools_a_refresh_could_not_unregister(stubborn_registry):
    """The same guarantee holds on ``refresh``, the other path through it."""
    tool_manager, armed = stubborn_registry()
    factory = _factory({"weather": (True, ("a", "b"))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    armed.add("mcp__weather__a")

    await manager.refresh("weather")

    assert (
        manager.status()["weather"].tool_count == 2
    ), "'a' is stuck in the registry and 'b' was registered, so both are owned"

    armed.clear()
    await manager.stop()

    assert tool_manager.tools_map() == {}


@pytest.mark.asyncio
async def test_stop_completes_both_phases_and_then_propagates_a_cancellation(
    monkeypatch,
):
    """A cancelled unregistration must not skip the disconnections.

    The two phases run one after the other, so re-raising as soon as the first
    one is cancelled would leave every server connected — the outcome the
    per-server isolation exists to prevent, reached through the propagation path.
    """
    tool_manager = ToolManager()
    factory = _factory({"bad": (True, ("x",)), "good": (True, ("y",))})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    real_unregister = tool_manager.unregister

    def _unregister(name):
        if name == "mcp__bad__x":
            raise asyncio.CancelledError()
        return real_unregister(name)

    monkeypatch.setattr(tool_manager, "unregister", _unregister)

    with pytest.raises(asyncio.CancelledError):
        await manager.stop()

    assert factory.made["good"].state is McpConnectionState.DISCONNECTED
    assert (
        factory.made["bad"].state is McpConnectionState.DISCONNECTED
    ), "every connection was closed before the cancellation was re-raised"
    assert manager.status()["bad"].error == "CancelledError"


@pytest.mark.asyncio
async def test_stop_completes_the_teardown_before_re_raising_a_shutdown_signal(
    monkeypatch,
):
    """A signal out of ``unregister`` must not skip the disconnections either.

    The phase loop deferred a cancellation and nothing else, but the work it
    drives re-raises the whole ``BaseException`` family once every server has
    been visited — so a signal out of the removal phase walked straight out of
    ``stop()``, leaving every connection open and ``status`` still reporting
    ``connected``.  Deferring the family as a whole is what ``stop``'s own
    docstring promises: neither phase can cost the other its work.
    """
    tool_manager = ToolManager()
    factory = _factory({"bad": (True, ("x",)), "good": (True, ("y",))})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    real_unregister = tool_manager.unregister

    def _unregister(name):
        if name == "mcp__bad__x":
            raise _ShutdownSignal("application is shutting down")
        return real_unregister(name)

    monkeypatch.setattr(tool_manager, "unregister", _unregister)

    with pytest.raises(_ShutdownSignal):
        await manager.stop()

    assert (
        factory.made["bad"].state is McpConnectionState.DISCONNECTED
    ), "the signal skipped the disconnection phase"
    assert factory.made["good"].state is McpConnectionState.DISCONNECTED
    assert "shutting down" in manager.status()["bad"].error, "and was not swallowed"
    # The removal was cut short, so the tool is still in the registry -- and
    # still claimed there, so status() and the registry agree.
    assert set(tool_manager.tools_map()) == {"mcp__bad__x"}
    assert manager.status()["bad"].tool_count == 1


@pytest.mark.asyncio
async def test_refresh_records_a_cancellation_and_lets_it_out():
    """``refresh`` propagates a cancellation for the same reason ``start`` does."""
    tool_manager = ToolManager()
    factory = _rude_factory({"bad": "", "good": ""})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    factory.made["bad"].cancel_next_connect()

    with pytest.raises(asyncio.CancelledError):
        await manager.refresh("bad")

    assert manager.status()["bad"].error == "CancelledError"
    assert (
        factory.made["good"].state is McpConnectionState.CONNECTED
    ), "refreshing one server left the others alone"


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
async def test_stop_completes_teardown_when_an_unregister_raises(stubborn_registry):
    """A registry that refuses to give a tool back must not strand a connection.

    Unregistration runs before the disconnects, so an unguarded failure here
    would skip the teardown entirely and leave every server connected.
    """
    tool_manager, armed = stubborn_registry()
    factory = _factory({"bad": (True, ("x",)), "good": (True, ("y",))})
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    armed.add("mcp__bad__x")

    await manager.stop()

    assert (
        factory.made["good"].state is McpConnectionState.DISCONNECTED
    ), "the healthy server was torn down"
    assert factory.made["bad"].state is McpConnectionState.DISCONNECTED
    assert set(tool_manager.tools_map()) == {
        "mcp__bad__x"
    }, "only the tool the registry refused to remove is left"
    assert "registry exploded" in manager.status()["bad"].error


@pytest.mark.asyncio
async def test_stop_retries_an_unregister_that_previously_failed(stubborn_registry):
    """A failed removal stays owned, so a later stop() can finish the job.

    The manager pops its ownership record only once a removal is confirmed;
    forgetting a tool it never removed would leave it dispatchable forever.
    """
    tool_manager, armed = stubborn_registry()
    factory = _factory({"bad": (True, ("x",))})
    manager = McpManager(
        _config(("bad", {"url": "https://bad.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    armed.add("mcp__bad__x")
    await manager.stop()
    assert manager.status()["bad"].tool_count == 1, "still owned, so still counted"

    armed.clear()
    await manager.stop()

    assert tool_manager.tools_map() == {}
    assert manager.status()["bad"].tool_count == 0
    assert (
        manager.status()["bad"].error == ""
    ), "a failure the retry resolved is history, not a current error"


@pytest.mark.asyncio
async def test_stop_forgets_a_tool_that_was_already_removed():
    """``unregister`` reporting "nothing was there" ends the manager's claim.

    A tool someone else removed from the registry cannot be removed again, so
    keeping it on the ownership record would make every later stop() retry a
    removal that can never succeed.
    """
    tool_manager = ToolManager()
    factory = _factory({"weather": (True, ("a",))})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    assert (
        tool_manager.unregister("mcp__weather__a") is True
    ), "removed behind the manager's back"

    await manager.stop()

    assert manager.status()["weather"].tool_count == 0
    assert tool_manager.tools_map() == {}


@pytest.mark.asyncio
async def test_status_prefers_a_recorded_failure_over_a_stale_client_message():
    """The newest reason wins when the manager and the client both have one.

    The client's message describes the connect that failed earlier; the
    recorded failure describes the teardown that just failed, which is the one
    an operator can act on.
    """
    tool_manager = ToolManager()
    factory = _rude_factory({"weather": "disconnect"}, succeed={"weather": False})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    assert manager.status()["weather"].error == "connection refused"

    await manager.stop()

    assert manager.status()["weather"].error == (
        "RuntimeError: teardown exploded"
    ), "the teardown failure is the most recent one"


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
    assert (
        manager.status()["weather"].tool_count == 1
    ), "the ownership record must shrink with the catalogue, or status() promises a tool that left"


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


def test_a_repeated_server_name_in_a_list_is_rejected():
    """Two entries under one name would make status() and the registry disagree.

    The name is both the client key and the ``mcp__<name>__`` tool namespace,
    so a duplicate leaves a tool registered, dispatchable and reported as
    ``disabled`` with ``tool_count=0``.  Refusing the configuration is the only
    way to keep those three facts in agreement.
    """
    factory = _factory({"dup": (True, ("a",))})
    config = [
        McpServerConfig.parse("dup", {"url": "https://one.invalid/mcp"}),
        McpServerConfig.parse("dup", {"url": "https://two.invalid/mcp"}),
    ]

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpManager(config, tool_manager=ToolManager(), client_factory=factory)

    assert "dup" in str(excinfo.value), "the duplicated name is named"
    assert "1" in str(excinfo.value), "and so is the offending position"
    assert factory.made == {}, "a rejected configuration connects to nothing"


def test_a_repeated_server_name_is_rejected_even_when_only_one_is_enabled():
    """Enabled-then-disabled is the pair that produced the worst mismatch."""
    factory = _factory({"dup": (True, ("a",))})
    config = [
        McpServerConfig.parse("dup", {"url": "https://one.invalid/mcp"}),
        McpServerConfig.parse(
            "dup", {"url": "https://two.invalid/mcp", "enabled": False}
        ),
    ]

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpManager(config, tool_manager=ToolManager(), client_factory=factory)

    assert "dup" in str(excinfo.value)
    assert factory.made == {}


@pytest.mark.asyncio
async def test_two_servers_offering_the_same_tool_name_both_register():
    """A duplicate *tool* name across servers is legal and must keep working.

    Only duplicate *server* names collide; the ``mcp__<server>__`` prefix keeps
    two servers' identically-named tools apart.
    """
    tool_manager = ToolManager()
    factory = _factory({"alpha": (True, ("search",)), "beta": (True, ("search",))})
    manager = McpManager(
        _config(
            ("alpha", {"url": "https://alpha.invalid/mcp"}),
            ("beta", {"url": "https://beta.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert set(tool_manager.tools_map()) == {
        "mcp__alpha__search",
        "mcp__beta__search",
    }
    assert manager.status()["alpha"].tool_count == 1
    assert manager.status()["beta"].tool_count == 1
    assert tool_manager.agent_docs()["mcp__alpha__search"] == "[alpha] search on alpha"


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
    """Every name in ``__all__`` resolves; a stale entry is a failed import."""
    from mitocode.core import mcp

    assert mcp.McpManager is McpManager
    assert mcp.McpServerStatus is McpServerStatus
    assert mcp.McpClient is McpClient
    assert mcp.McpConnectionState is McpConnectionState
    assert mcp.McpServerConfig is McpServerConfig
    assert mcp.McpServerKind is McpServerKind
    assert set(mcp.__all__) == {
        "McpClient",
        "McpConnectionState",
        "McpManager",
        "McpServerConfig",
        "McpServerKind",
        "McpServerStatus",
    }, "the package advertises exactly the six documented names"


def test_mcp_package_imports_without_the_tool_layer():
    """The ``mcp`` package must be importable with no tool layer at all.

    The plan fixes the dependency direction: ``mcp`` never imports
    ``tool_manager``.  A reference kept under ``TYPE_CHECKING`` is invisible to
    this probe, which is why the block covers the whole ``mitocode.core.tool``
    package rather than one name.
    """
    result = _run_import_probe(_IMPORT_WITHOUT_TOOL_LAYER)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


@pytest.mark.asyncio
async def test_local_stdio_server_end_to_end(tmp_path):
    """The local-server path, proven against a real subprocess.

    No ``client_factory`` here: this is the only test in the feature that
    exercises the whole chain — a real ``McpClient``, a real stdio transport
    and a real ``sys.executable`` child serving over MCP — so it is what
    justifies the transport plumbing underneath it.

    ``start`` is inside the ``try`` on purpose.  A child process outlives the
    call that spawned it unless something closes its transport, so a failure
    *during* the startup is exactly when the teardown matters most.
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

    try:
        await manager.start()
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


@pytest.mark.asyncio
async def test_one_servers_failure_never_lands_on_another_servers_status():
    """Four servers, three broken in different ways, one healthy.

    Each failure has to be visible against the server that caused it and
    invisible against every other: a shared error slot would report one reason
    four times and say nothing about which server is actually broken.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {
            "raises": {"rude": "connect"},
            "offline": {"succeed": False},
            "unreadable": {"catalogue_raises": True},
            "healthy": {"tools": ("ping",)},
        }
    )
    manager = McpManager(
        _config(
            ("raises", {"url": "https://raises.invalid/mcp"}),
            ("offline", {"url": "https://offline.invalid/mcp"}),
            ("unreadable", {"url": "https://unreadable.invalid/mcp"}),
            ("healthy", {"url": "https://healthy.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    status = manager.status()
    assert set(tool_manager.tools_map()) == {
        "mcp__healthy__ping"
    }, "only the healthy server publishes tools"
    assert (
        status["healthy"].state,
        status["healthy"].error,
        status["healthy"].tool_count,
    ) == ("connected", "", 1), "the healthy server is untouched by its neighbours"
    assert (
        status["raises"].state,
        status["raises"].tool_count,
    ) == ("disconnected", 0), "a connect that raised never reached the registry"
    assert status["raises"].error == "RuntimeError: handshake exploded"
    assert (
        status["offline"].state,
        status["offline"].error,
        status["offline"].tool_count,
    ) == ("failed", "connection refused", 0)
    assert (
        status["unreadable"].state,
        status["unreadable"].error,
        status["unreadable"].tool_count,
    ) == (
        "connected",
        "RuntimeError: catalogue exploded on access",
        0,
    ), "an unreadable catalogue fails that server's registration, not its connect"

    reasons = {status[name].error for name in ("raises", "offline", "unreadable")}
    for name, entry in status.items():
        if name in ("raises", "offline", "unreadable"):
            continue
        assert (
            entry.error not in reasons
        ), f"'{name}' was handed another server's failure: {entry.error!r}"


@pytest.mark.asyncio
async def test_stop_after_an_isolated_start_closes_the_connection_that_did_open():
    """A partial start must still be fully torn down.

    ``start`` returns once every server has had its turn, failed or not, so
    ``stop`` is the only thing left that can reclaim what did come up.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {"offline": {"succeed": False}, "healthy": {"tools": ("ping",)}}
    )
    manager = McpManager(
        _config(
            ("offline", {"url": "https://offline.invalid/mcp"}),
            ("healthy", {"url": "https://healthy.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    assert "mcp__healthy__ping" in tool_manager.tools_map()

    await manager.stop()

    assert tool_manager.tools_map() == {}, "no tool outlives stop()"
    assert factory.made["healthy"].connected is False
    assert factory.made["healthy"].state is McpConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_a_registry_failing_on_the_second_server_leaves_the_first_published(
    monkeypatch,
):
    """The seam between two servers is one call, not one shared outcome."""
    tool_manager = ToolManager()
    real_register = tool_manager.register_mcp_client

    def _register(client):
        if client.server_name == "second":
            raise ValueError("catalogue exploded")
        return real_register(client)

    monkeypatch.setattr(tool_manager, "register_mcp_client", _register)
    factory = _hostile_factory(
        {
            "first": {"tools": ("a",)},
            "second": {"tools": ("b",)},
            "third": {"tools": ("c",)},
        }
    )
    manager = McpManager(
        _config(
            ("first", {"url": "https://first.invalid/mcp"}),
            ("second", {"url": "https://second.invalid/mcp"}),
            ("third", {"url": "https://third.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert set(tool_manager.tools_map()) == {
        "mcp__first__a",
        "mcp__third__c",
    }, "the servers either side of the failing one still publish"
    assert manager.status()["first"].tool_count == 1
    assert manager.status()["third"].tool_count == 1
    assert manager.status()["second"].tool_count == 0
    assert manager.status()["second"].error == "ValueError: catalogue exploded"


@pytest.mark.asyncio
async def test_a_tool_the_registry_already_holds_is_neither_adopted_nor_removed():
    """A name that is already taken belongs to whoever put it there.

    ``register_mcp_client`` skips a name that is still registered, so that tool
    is absent from the result it returns.  Recording the catalogue rather than
    the registration result would make the manager its owner: it would count a
    tool it never registered, and its ``stop`` would delete another layer's.
    """
    tool_manager = ToolManager()

    async def pre_existing(**_kwargs):
        """Stand-in for a tool some other layer registered.

        Returns:
            A fixed string, so a dispatch proves the original survived.
        """
        return "not from MCP"

    tool_manager.register(pre_existing, name="mcp__weather__a", human_doc="mine")
    factory = _hostile_factory({"weather": {"tools": ("a", "b")}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert set(tool_manager.tools_map()) == {"mcp__weather__a", "mcp__weather__b"}
    assert (
        tool_manager.get_spec("mcp__weather__a").human_doc == "mine"
    ), "the name that was already taken was not overwritten"
    assert (
        manager.status()["weather"].tool_count == 1
    ), "only the tool the manager actually registered is counted"

    await manager.stop()

    assert set(tool_manager.tools_map()) == {
        "mcp__weather__a"
    }, "stop must not remove a tool the manager never owned"
    assert await tool_manager.tools_map()["mcp__weather__a"]() == "not from MCP"


@pytest.mark.asyncio
async def test_a_connected_server_that_advertises_nothing_is_distinguishable_from_a_failed_one():
    """Both contribute no tools; only one of them is broken.

    ``start`` registers only after a successful connect precisely so that an
    empty result means "this server offers nothing" and never "this server
    never came up".  ``status`` has to keep the two apart as well.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory({"empty": {"succeed": True}, "down": {"succeed": False}})
    manager = McpManager(
        _config(
            ("empty", {"url": "https://empty.invalid/mcp"}),
            ("down", {"url": "https://down.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert (
        manager.status()["empty"].state,
        manager.status()["empty"].error,
        manager.status()["empty"].tool_count,
    ) == ("connected", "", 0), "an empty catalogue is not a failure"
    assert (
        manager.status()["down"].state,
        manager.status()["down"].error,
        manager.status()["down"].tool_count,
    ) == ("failed", "connection refused", 0)
    assert tool_manager.tools_map() == {}


def test_status_lists_every_enabled_server_before_the_disabled_ones():
    """The documented order is the enabled servers first, then the disabled."""
    manager = McpManager(
        _config(
            ("first", {"url": "https://a.invalid/mcp"}),
            ("off", {"url": "https://b.invalid/mcp", "enabled": False}),
            ("second", {"url": "https://c.invalid/mcp"}),
        ),
        tool_manager=ToolManager(),
        client_factory=_hostile_factory({"first": {}, "second": {}}),
    )

    assert list(manager.status()) == ["first", "second", "off"]


def test_each_status_entry_is_a_frozen_snapshot():
    """A reader cannot rewrite what the next reader of ``status()`` will see."""
    status = McpServerStatus(state="connected", error="", tool_count=2)

    assert (status.state, status.error, status.tool_count) == ("connected", "", 2)

    with pytest.raises(dataclasses.FrozenInstanceError):
        status.tool_count = 99

    assert status.tool_count == 2, "the rejected assignment changed nothing"


@pytest.mark.asyncio
async def test_status_counts_exactly_the_tools_surviving_a_half_finished_stop(
    stubborn_registry,
):
    """One tool out of two refuses to leave; only that one is still owned.

    The ownership record is updated per tool and only once the registry has
    confirmed each removal, so it must name the tool that stayed and not the
    one that left.  Keeping both would make ``status`` promise a tool nobody
    could find in the registry.
    """
    tool_manager, armed = stubborn_registry()
    factory = _hostile_factory({"weather": {"tools": ("a", "b")}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    armed.add("mcp__weather__a")

    await manager.stop()

    assert set(tool_manager.tools_map()) == {
        "mcp__weather__a"
    }, "'b' was released, so it must not still be claimed"
    assert manager.status()["weather"].tool_count == 1
    assert manager.status()["weather"].error == "RuntimeError: registry exploded"


@pytest.mark.asyncio
async def test_stop_removes_the_tools_while_their_client_is_still_connected(
    monkeypatch,
):
    """The removal phase runs before the disconnection phase.

    The documented reason is the reverse failure: a tool still registered while
    its client is gone dispatches into a dead client at call time, in the
    model's hands rather than here.  The order is observable from the registry
    side -- a removal that runs after the connection has closed is removing a
    tool nobody could serve any more.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory({"weather": {"tools": ("get_forecast",)}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    live_at_removal = {}
    real_unregister = tool_manager.unregister

    def _unregister(name):
        live_at_removal[name] = factory.made["weather"].connected
        return real_unregister(name)

    monkeypatch.setattr(tool_manager, "unregister", _unregister)

    await manager.stop()

    assert live_at_removal == {
        "mcp__weather__get_forecast": True
    }, "the tool was unregistered after its client had already been disconnected"


@pytest.mark.asyncio
async def test_stop_propagates_a_cancellation_raised_by_a_disconnect():
    """A cancelled teardown is the caller unwinding, not a server that failed.

    Both phases still run and every tool is still released.  The connection
    that refused to close stays open, and ``status`` says so instead of
    claiming a teardown that did not happen.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {
            "stuck": {"tools": ("x",), "cancel_on": "disconnect"},
            "good": {"tools": ("y",)},
        }
    )
    manager = McpManager(
        _config(
            ("stuck", {"url": "https://stuck.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    with pytest.raises(asyncio.CancelledError):
        await manager.stop()

    assert tool_manager.tools_map() == {}, "the removal phase ran before the teardown"
    assert factory.made["good"].state is McpConnectionState.DISCONNECTED
    assert (
        factory.made["stuck"].state is McpConnectionState.CONNECTED
    ), "the connection whose close was cancelled is still open"
    assert manager.status()["stuck"].error == "CancelledError"
    assert manager.status()["stuck"].tool_count == 0


@pytest.mark.asyncio
async def test_a_cancelled_refresh_leaves_that_server_with_no_tools_and_the_others_alone():
    """A refresh cancelled mid-teardown publishes nothing.

    The tools go before the reconnect is attempted, so a cancellation between
    the two leaves the server with none rather than with tools pointing at a
    client that is not coming back.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {
            "bad": {"tools": ("x",), "cancel_on": "disconnect"},
            "good": {"tools": ("y",)},
        }
    )
    manager = McpManager(
        _config(
            ("bad", {"url": "https://bad.invalid/mcp"}),
            ("good", {"url": "https://good.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    with pytest.raises(asyncio.CancelledError):
        await manager.refresh("bad")

    assert set(tool_manager.tools_map()) == {
        "mcp__good__y"
    }, "the cancelled server published nothing and the untouched one kept its tools"
    assert manager.status()["bad"].tool_count == 0
    assert manager.status()["bad"].error == "CancelledError"
    assert (
        manager.status()["good"].state,
        manager.status()["good"].error,
        manager.status()["good"].tool_count,
    ) == ("connected", "", 1), "refreshing one server left the other alone"


@pytest.mark.asyncio
async def test_every_cancelled_server_is_visited_before_the_cancellation_is_raised():
    """One cancelled server must not cost the next one its turn.

    ``start`` runs the servers concurrently and inspects the results only
    afterwards, so both cancelled servers are connected before the
    cancellation is re-raised.  A sequential loop that re-raised on the spot
    would never reach the second one, and the healthy server would be left
    never tried at all.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {
            "first": {"tools": ("a",)},
            "second": {"tools": ("b",)},
            "healthy": {"tools": ("c",)},
        }
    )
    manager = McpManager(
        _config(
            ("first", {"url": "https://first.invalid/mcp"}),
            ("second", {"url": "https://second.invalid/mcp"}),
            ("healthy", {"url": "https://healthy.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    factory.made["first"].cancel_next_connect()
    factory.made["second"].cancel_next_connect()

    with pytest.raises(asyncio.CancelledError):
        await manager.start()

    assert factory.made["first"].attempts == 1
    assert (
        factory.made["second"].attempts == 1
    ), "the second cancelled server never got its turn"
    assert set(tool_manager.tools_map()) == {"mcp__healthy__c"}
    assert (
        manager.status()["first"].state,
        manager.status()["first"].tool_count,
    ) == ("disconnected", 0)
    assert (
        manager.status()["healthy"].state,
        manager.status()["healthy"].tool_count,
    ) == ("connected", 1)


@pytest.mark.asyncio
async def test_status_error_is_the_exception_text_verbatim(stubborn_registry):
    """The recorded error is the exception and nothing the manager adds.

    ``status`` is rendered into the TUI and written to the log, so forwarding
    the exception's own text is the whole contract: a server name, a tool name
    or a configuration value interpolated here would be text the exception
    never contained, arriving at the one surface a user reads.
    """
    tool_manager, _armed = stubborn_registry("mcp__weather__a")
    factory = _hostile_factory({"weather": {"tools": ("a",)}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    await manager.stop()

    assert (
        manager.status()["weather"].error == "RuntimeError: registry exploded"
    ), "no server name, no tool name, no decoration"


@pytest.mark.asyncio
async def test_the_log_names_the_tool_a_failed_unregistration_could_not_release(
    stubborn_registry,
):
    """The tool that stayed is named in the log, and not in ``status``.

    ``status`` describes a server; the log is where an operator reads which of
    its tools is still holding a registry slot.  A tool name is arbitrary text
    from the server, so the two surfaces deliberately carry different amounts
    and each has to be checked on its own.
    """
    tool_manager, _armed = stubborn_registry("mcp__weather__a")
    factory = _hostile_factory({"weather": {"tools": ("a",)}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()

    with _captured_logs() as lines:
        await manager.stop()

    reported = [line for line in lines if "failed to unregister" in line]
    assert len(reported) == 1, f"expected one unregistration failure, got {reported}"
    assert "mcp__weather__a" in reported[0], "the log names the tool that stayed"
    assert "weather" in reported[0], "and the server it belongs to"
    assert "RuntimeError: registry exploded" in reported[0], "and why it stayed"
    assert (
        "mcp__weather__a" not in manager.status()["weather"].error
    ), "status describes the server, not the tool"


@pytest.mark.asyncio
async def test_a_connect_failure_does_not_publish_a_credential_from_the_endpoint():
    """The error the manager records is scrubbed of endpoint credentials.

    An unreachable server is the failure most likely to quote the endpoint it
    was called on, and a remote endpoint may legitimately carry its token in the
    query string.  ``status()`` is the pane an operator reads, so the scrub has
    to happen on the way in rather than being left to the reader to notice.
    """
    manager = McpManager(
        _config(
            (
                "remote",
                _http_secret_spec("remote"),
            )
        ),
        tool_manager=ToolManager(),
        client_factory=_hostile_factory(
            {
                "remote": {
                    "rude": "connect",
                    "message": (
                        "Internal error while proxying "
                        f"https://user:{_SECRET_USERINFO}@remote.invalid/mcp"
                        f"?access_token={_SECRET_HEADER}"
                    ),
                }
            }
        ),
    )

    await manager.start()

    error = manager.status()["remote"].error
    assert error != "", "the failure was recorded at all"
    assert "remote.invalid" in error, "and it still says which server failed"
    assert "<redacted>" in error, "with the credential-bearing parts marked"
    _assert_carries_no_secret(error, "status() for a failed connect")
    await manager.stop()


@pytest.mark.asyncio
async def test_the_log_says_why_an_unavailable_server_was_skipped():
    """A client that reports failure rather than raising it still explains itself.

    ``McpClient.connect`` returns ``False`` for a server that is merely down,
    which is the common case, so this branch is what an operator reads for
    almost every misconfigured entry.  Naming the server and nothing else
    leaves the one question the log was opened to answer unanswered.
    """
    factory = _hostile_factory({"weather": {"succeed": False}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=ToolManager(),
        client_factory=factory,
    )

    with _captured_logs() as lines:
        await manager.start()

    reported = [line for line in lines if "is unavailable" in line]
    assert len(reported) == 1, f"expected one unavailable server, got {reported}"
    assert "weather" in reported[0], "the server is named"
    assert "connection refused" in reported[0], "and so is why it was unavailable"
    assert manager.status()["weather"].state == "failed"


@pytest.mark.asyncio
async def test_the_unavailable_warning_survives_a_client_that_reports_no_reason():
    """A client with nothing to say must not produce a dangling colon.

    The manager accepts any duck-typed client, so an empty ``last_error`` is
    reachable even though the real one always explains itself.
    """
    factory = _hostile_factory({"weather": {"succeed": False}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=ToolManager(),
        client_factory=factory,
    )
    factory.made["weather"].last_error = ""

    with _captured_logs() as lines:
        await manager.start()

    reported = [line for line in lines if "is unavailable" in line]
    assert len(reported) == 1, f"expected one unavailable server, got {reported}"
    assert not reported[0].rstrip().endswith(":"), "no dangling separator is left"
    assert "not registered" in reported[0], "and the warning still says what it means"


@pytest.mark.asyncio
async def test_a_shutdown_signal_from_a_server_is_recorded_and_re_raised():
    """A ``BaseException`` that is neither an ``Exception`` nor a cancel is neither
    swallowed nor silently dropped.

    ``gather(..., return_exceptions=True)`` hands back *any* exception a child
    raised as a plain result, so a ``KeyboardInterrupt`` or a framework's own
    shutdown signal arrives at the manager as data.  Inspecting only for
    ``Exception`` and ``CancelledError`` left it on the floor: no record, no log,
    and ``start()`` returning normally while a server was being torn down by the
    whole process.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {
            "first": {"signal_on": "connect"},
            "second": {"tools": ("b",)},
        }
    )
    manager = McpManager(
        _config(
            ("first", {"url": "https://first.invalid/mcp"}),
            ("second", {"url": "https://second.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    with pytest.raises(_ShutdownSignal):
        await manager.start()

    assert (
        factory.made["first"].attempts == 1
    ), "the server that signalled was still given its turn"
    assert (
        "shutting down" in manager.status()["first"].error
    ), "and the signal was recorded against it"
    assert set(tool_manager.tools_map()) == {
        "mcp__second__b"
    }, "a signal from one server must not cost another its tools"


@pytest.mark.asyncio
async def test_a_shutdown_signal_from_a_teardown_is_recorded_and_re_raised():
    """The same rule governs ``stop()``, which runs the same gather."""
    tool_manager = ToolManager()
    factory = _hostile_factory({"weather": {"tools": ("a",)}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    await manager.start()
    factory.made["weather"]._signal_on = "disconnect"

    with pytest.raises(_ShutdownSignal):
        await manager.stop()

    assert (
        "shutting down" in manager.status()["weather"].error
    ), "the teardown's shutdown signal was recorded, not dropped"


@pytest.mark.asyncio
async def test_the_log_says_a_tool_the_registry_no_longer_holds_was_dropped(
    monkeypatch,
):
    """An unregistration the registry reports as a no-op is logged, not silent.

    The registry claims the tool is gone while the manager stops owning it.  The
    two agree in every real case, so a quiet line here is exactly how a
    registry and a manager that disagree about who owns what would go unnoticed
    — and the tool would be left in the registry with nobody to remove it.
    """
    tool_manager = ToolManager()
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=_hostile_factory({"weather": {"tools": ("a",)}}),
    )
    await manager.start()
    monkeypatch.setattr(tool_manager, "unregister", lambda name: False)

    with _captured_logs() as lines:
        await manager.stop()

    reported = [line for line in lines if "no longer owns it" in line]
    assert len(reported) == 1, f"expected one dropped tool, got {reported}"
    assert "mcp__weather__a" in reported[0], "the tool that was dropped is named"
    assert "weather" in reported[0], "and the server it belonged to"


def test_a_duplicate_name_error_does_not_echo_any_configuration_value():
    """The refusal names the name and the positions, and nothing else.

    Both entries carry a distinct secret in every field that accepts one, so
    the only way to keep them out of the message is for the message not to
    carry either entry at all.
    """
    config = [
        McpServerConfig.parse("dup", _http_secret_spec("one")),
        McpServerConfig.parse("dup", _stdio_secret_spec()),
    ]

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpManager(
            config, tool_manager=ToolManager(), client_factory=_hostile_factory({})
        )

    message = str(excinfo.value)
    assert "dup" in message, "the duplicated name is named"
    assert "1" in message, "and so is the offending position"
    _assert_carries_no_secret(message, "the duplicate-name error")


def test_a_mistyped_list_entry_error_does_not_echo_any_configuration_value():
    """A rejected element is described by its position and its type.

    The rejected element *is* the secret-bearing one here, and its neighbours
    are real entries carrying secrets too, so the message has to name the
    element that was wrong without ever rendering any of them.
    """
    config = [
        McpServerConfig.parse("first", _http_secret_spec("one")),
        _http_secret_spec("rogue"),
        McpServerConfig.parse("third", _stdio_secret_spec()),
    ]

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        McpManager(
            config, tool_manager=ToolManager(), client_factory=_hostile_factory({})
        )

    message = str(excinfo.value)
    assert "1" in message, "the offending position is named"
    assert "dict" in message, "and so is the type that was found there"
    _assert_carries_no_secret(message, "the mistyped-entry error")


@pytest.mark.asyncio
async def test_an_unknown_refresh_name_error_does_not_echo_any_configuration_value():
    """The refusal names the server asked for, not how it is configured.

    The existing probe puts a secret in ``headers`` only, so ``url``,
    ``env``, ``args`` and ``cwd`` were never at risk from this module's own
    message.  Here each carries one of its own.
    """
    manager = McpManager(
        _config(
            ("remote", _http_secret_spec("remote")),
            ("local", _stdio_secret_spec()),
        ),
        tool_manager=ToolManager(),
        client_factory=_hostile_factory({"remote": {}, "local": {}}),
    )

    with pytest.raises(MitoCodeMcpConfigError) as excinfo:
        await manager.refresh("nope")

    message = str(excinfo.value)
    assert "nope" in message, "the requested name is named"
    assert "remote" in message and "local" in message, "and the configured servers"
    _assert_carries_no_secret(message, "the unknown-server error")


@pytest.mark.asyncio
async def test_a_registration_that_raises_partway_publishes_nothing_at_all():
    """A catalogue that cannot be read whole publishes no part of itself.

    Driven through the real ``ToolManager``, not a patched one: the second
    advertised tool carries a description that is not a string, so the
    registry raises while reading it, after the first tool would otherwise
    have been published.  ``start`` is told the registration failed and
    records no ownership for the server, so anything left behind would be a
    dispatchable tool that ``status`` does not count and ``stop`` can never
    reclaim -- the hazard the ``stuck`` merge defends against, arriving from
    the registration side instead.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory(
        {"weather": {"malformed": True}, "healthy": {"tools": ("g",)}}
    )
    manager = McpManager(
        _config(
            ("weather", {"url": "https://w.invalid/mcp"}),
            ("healthy", {"url": "https://healthy.invalid/mcp"}),
        ),
        tool_manager=tool_manager,
        client_factory=factory,
    )

    await manager.start()

    assert set(tool_manager.tools_map()) == {
        "mcp__healthy__g"
    }, "the other server still publishes, and the broken one publishes nothing"
    assert manager.status()["weather"].error != "", "the failure was recorded"
    assert (
        manager.status()["weather"].tool_count == 0
    ), "a server whose registration failed owns nothing"

    await manager.stop()

    assert (
        tool_manager.tools_map() == {}
    ), "a tool the manager cannot name is a tool nobody can remove"


@pytest.mark.asyncio
async def test_a_failed_registration_does_not_adopt_a_name_another_layer_owns():
    """The constraint that rules out composing ownership from the catalogue.

    The catalogue's first tool collides with a tool some other layer
    registered, so the registry skips it, and its second tool cannot be
    described, so the registration fails.  A manager that adopted the
    advertised catalogue on failure would put both names into its ownership
    record -- including the one it skipped -- and ``stop`` would then delete a
    tool it never published.  Green today; the lock on what any fix here has
    to preserve.
    """
    tool_manager = ToolManager()

    async def pre_existing(**_kwargs):
        """Stand-in for a tool some other layer registered.

        Returns:
            A fixed string, so a dispatch proves the original survived.
        """
        return "not from MCP"

    tool_manager.register(pre_existing, name="mcp__weather__healthy", human_doc="mine")
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=_hostile_factory(
            {"weather": {"malformed": True, "healthy_tool": "healthy"}}
        ),
    )

    await manager.start()

    assert (
        manager.status()["weather"].tool_count == 0
    ), "the registration failed, so the manager owns nothing at all"

    await manager.stop()

    assert set(tool_manager.tools_map()) == {
        "mcp__weather__healthy"
    }, "the tool another layer owns survived a start/stop cycle"
    assert await tool_manager.tools_map()["mcp__weather__healthy"]() == "not from MCP"


@pytest.mark.asyncio
async def test_a_stdio_server_that_dies_costs_only_its_own_tools(tmp_path):
    """The isolation claim end to end, against two real subprocesses.

    One child exits immediately without speaking MCP; the other serves a tool
    over a real stdio transport.  No ``client_factory`` is involved, so this is
    the feature's central promise proven through the real transport rather than
    through a stand-in.
    """
    server_script = tmp_path / "srv.py"
    server_script.write_text(
        "from fastmcp import FastMCP\n"
        "app = FastMCP('live')\n"
        "@app.tool\n"
        "def ping() -> str:\n"
        "    '''Answer a ping.'''\n"
        "    return 'pong'\n"
        "app.run()\n"
    )

    tool_manager = ToolManager()
    manager = McpManager(
        {
            "mcpServers": {
                "dead": {
                    "command": sys.executable,
                    "args": ["-c", "import sys; sys.exit(3)"],
                    "connect_timeout": 20.0,
                },
                "live": {
                    "command": sys.executable,
                    "args": [str(server_script)],
                    "connect_timeout": 60.0,
                },
            }
        },
        tool_manager=tool_manager,
    )

    try:
        await manager.start()

        dead = manager.status()["dead"]
        assert (dead.state, dead.tool_count) == (
            "failed",
            0,
        ), f"the dead server reported {dead.error!r}"
        assert dead.error != "", "and it says why"

        live = manager.status()["live"]
        assert (live.state, live.error, live.tool_count) == ("connected", "", 1)
        assert set(tool_manager.tools_map()) == {"mcp__live__ping"}
        assert await tool_manager.tools_map()["mcp__live__ping"]() == "pong"
    finally:
        await manager.stop()

    assert tool_manager.tools_map() == {}
    assert manager.status()["live"].state == "disconnected"


@pytest.mark.asyncio
async def test_refresh_after_a_failed_refresh_publishes_the_catalogue_again():
    """A server that failed to reconnect is not permanently marked broken.

    The failed refresh leaves no tools behind, so the ownership record is empty
    when the next one runs.  That is what makes recovery possible: the retry has
    to start from a clean slate rather than from a record naming tools that
    already left.
    """
    tool_manager = ToolManager()
    factory = _hostile_factory({"weather": {"tools": ("a",)}})
    manager = McpManager(
        _config(("weather", {"url": "https://w.invalid/mcp"})),
        tool_manager=tool_manager,
        client_factory=factory,
    )
    client = factory.made["weather"]
    client.go_offline()

    await manager.refresh("weather")

    assert tool_manager.tools_map() == {}, "the failed refresh published nothing"
    assert manager.status()["weather"].state == "failed"

    client.go_online()
    # The failed connect dropped the cached catalogue, exactly as the real
    # client does, so the server "coming back" is a server advertising
    # something new rather than the same thing twice.
    for name in ("a", "b"):
        client.tools.append(
            SimpleNamespace(
                name=name,
                description=f"{name} on weather",
                input_schema={"type": "object", "properties": {}},
            )
        )
    await manager.refresh("weather")

    assert set(tool_manager.tools_map()) == {
        "mcp__weather__a",
        "mcp__weather__b",
    }, "the recovery publishes the catalogue the server advertises now"
    assert manager.status()["weather"] == McpServerStatus(
        "connected", "", 2
    ), "a failure the retry resolved is history, not a current error"


@pytest.mark.asyncio
async def test_refresh_without_a_name_never_touches_a_disabled_server():
    """``refresh()`` with no name means every *enabled* server.

    A disabled entry has no client, so reaching one is not a failure a
    ``try``/``except`` around a reconnect could absorb: it would be an
    unrecorded ``KeyError`` out of a method documented never to raise for a
    server.
    """
    factory = _hostile_factory({"on": {"tools": ("a",)}})
    manager = McpManager(
        _config(
            ("on", {"url": "https://on.invalid/mcp"}),
            ("off", {"url": "https://off.invalid/mcp", "enabled": False}),
        ),
        tool_manager=ToolManager(),
        client_factory=factory,
    )

    await manager.refresh()

    assert "off" not in factory.made, "a disabled server is never given a client"
    assert factory.made["on"].state is McpConnectionState.CONNECTED
    assert manager.status()["off"] == McpServerStatus("disabled", "", 0)
