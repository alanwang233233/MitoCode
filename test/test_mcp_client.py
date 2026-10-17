import asyncio
import os
import sys
from typing import Optional

import pytest
from fastmcp import Client, FastMCP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpToolCallError
from mitocode.backend.mcp.mcp_client import McpClient, McpConnectionState
from mitocode.backend.mcp.mcp_config import McpServerConfig


def _server() -> FastMCP:
    """Build an in-memory FastMCP server exposing three tools."""
    server = FastMCP("probe")

    @server.tool
    def echo(text: str) -> str:
        """Echo the given text back."""
        return f"echo:{text}"

    @server.tool
    def add(a: int, b: int) -> dict:
        """Add two integers."""
        return {"sum": a + b}

    @server.tool
    def boom() -> str:
        """Always fail."""
        raise ValueError("kaboom")

    return server


def _in_memory_config() -> McpServerConfig:
    """Config placeholder reused by connect() monkeypatching."""
    return McpServerConfig.parse("probe", {"url": "https://probe.invalid/mcp"})


def _slow_config(connect_timeout: float) -> McpServerConfig:
    """Build a config pointing at an address that never answers."""
    return McpServerConfig.parse(
        "slow",
        {"url": "https://10.255.255.1/mcp", "connect_timeout": connect_timeout},
    )


class _CloseRecorder:
    """Stand-in for a fastmcp ``Client`` that records whether close ran."""

    def __init__(self, error: Optional[Exception] = None):
        """Arm the recorder, optionally to raise *error* instead of returning."""
        self.error = error
        self.closed = False

    async def close(self) -> None:
        """Mark the client closed, then fail if this recorder was armed."""
        self.closed = True
        if self.error is not None:
            raise self.error


class _StackRecorder:
    """Stand-in for an ``AsyncExitStack`` that records whether aclose ran."""

    def __init__(self, error: Optional[Exception] = None):
        """Arm the recorder, optionally to raise *error* instead of returning."""
        self.error = error
        self.closed = False

    async def aclose(self) -> None:
        """Mark the stack closed, then fail if this recorder was armed."""
        self.closed = True
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_connect_discovers_tools_from_server(monkeypatch):
    config = _in_memory_config()
    client = McpClient(config)
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))

    assert await client.connect() is True
    assert client.state is McpConnectionState.CONNECTED
    assert client.last_error == ""
    assert sorted(t.name for t in client.tools) == ["add", "boom", "echo"]


@pytest.mark.asyncio
async def test_call_tool_returns_text_result(monkeypatch):
    config = _in_memory_config()
    client = McpClient(config)
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    await client.connect()

    assert await client.call_tool("echo", {"text": "hi"}) == "echo:hi"


@pytest.mark.asyncio
async def test_call_tool_serializes_structured_result_as_json(monkeypatch):
    config = _in_memory_config()
    client = McpClient(config)
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    await client.connect()

    result = await client.call_tool("add", {"a": 2, "b": 3})
    assert result == '{"sum": 5}'


@pytest.mark.asyncio
async def test_call_tool_raises_on_tool_error(monkeypatch):
    config = _in_memory_config()
    client = McpClient(config)
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    await client.connect()

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("boom", {})
    assert "kaboom" in str(excinfo.value)


@pytest.mark.asyncio
async def test_connect_failure_is_recorded_not_raised():
    config = McpServerConfig.parse(
        "broken",
        {"url": "https://127.0.0.1:1/mcp", "connect_timeout": 3.0},
    )
    client = McpClient(config)

    assert await client.connect() is False
    assert client.state is McpConnectionState.FAILED
    assert client.last_error != ""
    assert client.tools == []


@pytest.mark.asyncio
async def test_connect_timeout_is_honoured():
    client = McpClient(_slow_config(1.0))

    loop = asyncio.get_event_loop()
    started = loop.time()
    assert await client.connect() is False
    assert loop.time() - started < 5.0


@pytest.mark.asyncio
async def test_connect_timeout_force_closes_the_client():
    """A timed-out connect must not leave a subprocess running.

    Review Focus #6: FastMCP's StdioTransport spawns the process inside a
    detached task, so cancelling our await alone orphans it. Only
    Client.close() tears the process group down.
    """
    client = McpClient(_slow_config(1.0))

    class _HangingClient:
        """Async context manager whose __aenter__ never completes."""

        closed = False

        async def __aenter__(self):
            await asyncio.Event().wait()

        async def __aexit__(self, *exc_info):
            return False

        async def close(self):
            type(self).closed = True

    built = _HangingClient()
    client._build_fastmcp_client = lambda: built

    assert await client.connect() is False
    assert built.closed is True
    assert client._client is None


@pytest.mark.asyncio
async def test_connect_timeout_during_tool_discovery_is_honoured():
    """Discovery is a second round trip and needs the connect deadline too.

    A server that completes the handshake but then never answers list_tools
    would hang forever if only __aenter__ were wrapped in wait_for.
    """
    client = McpClient(_slow_config(1.0))

    class _StallingDiscoveryClient:
        """Client that connects instantly but never finishes listing tools."""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def list_tools(self):
            await asyncio.Event().wait()

        async def close(self):
            return None

    client._build_fastmcp_client = lambda: _StallingDiscoveryClient()

    loop = asyncio.get_event_loop()
    started = loop.time()
    # The outer deadline is a backstop: without wait_for around list_tools the
    # connect hangs forever, and this test should fail rather than hang CI.
    assert await asyncio.wait_for(client.connect(), timeout=10.0) is False
    assert loop.time() - started < 5.0
    assert client.state is McpConnectionState.FAILED
    assert client.tools == []


@pytest.mark.asyncio
async def test_call_tool_raises_when_not_connected():
    config = _in_memory_config()
    client = McpClient(config)

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("echo", {"text": "hi"})
    # Not merely "some error": the message must say the server is not connected,
    # otherwise an unreachable server is indistinguishable from a bad tool call.
    assert "not connected" in str(excinfo.value)


@pytest.mark.asyncio
async def test_call_tool_notes_non_text_content():
    """Review Focus #4: a non-text-only result must not become an empty string."""
    client = McpClient(_in_memory_config())

    class _Block:
        type = "image"

    class _Result:
        data = None
        content = [_Block()]
        is_error = False

    async def _fake_call_tool(name, arguments, **kwargs):
        return _Result()

    fake_client = type("C", (), {})()
    fake_client.call_tool = _fake_call_tool
    client._client = fake_client

    result = await client.call_tool("anything", {})
    assert "non-text" in result


@pytest.mark.asyncio
async def test_call_tool_renders_empty_result_as_empty_string(monkeypatch):
    client = McpClient(_in_memory_config())
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    await client.connect()

    class _EmptyResult:
        data = None
        content = []

    class _NullClient:
        async def call_tool(self, name, arguments, **kwargs):
            return _EmptyResult()

    client._client = _NullClient()

    assert await client.call_tool("echo", {}) == ""


@pytest.mark.asyncio
async def test_disconnect_clears_state_and_is_idempotent(monkeypatch):
    client = McpClient(_in_memory_config())
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    await client.connect()
    assert client.tools

    await client.disconnect()

    assert client.state is McpConnectionState.DISCONNECTED
    assert client.tools == []
    assert client._client is None
    await client.disconnect()
    assert client.state is McpConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_disconnect_records_cleanup_failure_without_raising(monkeypatch):
    """A teardown error must not stop the remaining servers from closing."""
    client = McpClient(_in_memory_config())
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    await client.connect()
    assert client.last_error == ""

    class _BadClose:
        async def close(self):
            raise RuntimeError("transport wedged")

    client._client = _BadClose()

    await client.disconnect()

    assert client.state is McpConnectionState.DISCONNECTED
    assert "transport wedged" in client.last_error
    assert client.tools == []


@pytest.mark.asyncio
async def test_connect_returns_early_when_already_connected(monkeypatch):
    client = McpClient(_in_memory_config())
    builds = []
    monkeypatch.setattr(
        client, "_build_fastmcp_client", lambda: builds.append(1) or Client(_server())
    )
    await client.connect()

    assert await client.connect() is True

    assert len(builds) == 1


def test_fresh_client_reports_its_server_name_and_disconnected_state():
    client = McpClient(_in_memory_config())

    assert client.server_name == "probe"
    assert client.state is McpConnectionState.DISCONNECTED
    assert client.tools == []
    assert client.last_error == ""


@pytest.mark.asyncio
async def test_call_tool_times_out_and_reports_the_deadline():
    client = McpClient(
        McpServerConfig.parse(
            "slowtool", {"url": "https://x.invalid/mcp", "timeout": 0.05}
        )
    )

    class _HangingCall:
        async def call_tool(self, name, arguments, **kwargs):
            await asyncio.Event().wait()

    client._client = _HangingCall()

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("echo", {"text": "hi"})
    assert "timed out" in str(excinfo.value)
    assert "0.05" in str(excinfo.value)


@pytest.mark.asyncio
async def test_call_tool_wraps_unexpected_error_without_leaking_arguments():
    """Arguments routinely carry file contents and credentials, so they must
    never reach an exception message that ends up in the TUI and in logs."""
    client = McpClient(_in_memory_config())

    class _ExplodingCall:
        async def call_tool(self, name, arguments, **kwargs):
            raise RuntimeError("transport gave up")

    client._client = _ExplodingCall()
    secret = {"token": "sk-do-not-log-me"}

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("echo", secret)

    message = str(excinfo.value)
    assert "transport gave up" in message
    assert "RuntimeError" in message
    assert "sk-do-not-log-me" not in message


@pytest.mark.asyncio
async def test_call_tool_does_not_re_wrap_its_own_error():
    client = McpClient(_in_memory_config())
    original = MitoCodeMcpToolCallError("precise original message")

    class _RaisingCall:
        async def call_tool(self, name, arguments, **kwargs):
            raise original

    client._client = _RaisingCall()

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("echo", {})

    assert excinfo.value is original


@pytest.mark.asyncio
async def test_call_tool_joins_text_content_blocks():
    client = McpClient(_in_memory_config())

    class _TextBlock:
        def __init__(self, text):
            self.text = text

    class _Result:
        data = None
        content = [_TextBlock("first "), _TextBlock("second")]

    class _Call:
        async def call_tool(self, name, arguments, **kwargs):
            return _Result()

    client._client = _Call()

    assert await client.call_tool("echo", {}) == "first second"


@pytest.mark.asyncio
async def test_force_close_records_the_type_of_a_failing_client_close():
    """A wedged transport must become a recorded error, never a raised one."""
    client = McpClient(_in_memory_config())
    recorder = _CloseRecorder(RuntimeError("transport wedged"))
    client._client = recorder
    client._stack = _StackRecorder()

    await client.disconnect()

    assert client.last_error == "close failed: RuntimeError: transport wedged"
    assert recorder.closed is True
    assert client._client is None
    assert client._stack is None
    assert client.tools == []


@pytest.mark.asyncio
async def test_force_close_still_closes_the_stack_when_client_close_raises():
    """A client that cannot be closed must not strand the stack holding it."""
    client = McpClient(_in_memory_config())
    client._client = _CloseRecorder(RuntimeError("client wedged"))
    stack = _StackRecorder()
    client._stack = stack

    await client.disconnect()

    assert stack.closed is True
    assert client.last_error == "close failed: RuntimeError: client wedged"


@pytest.mark.asyncio
async def test_force_close_records_a_failing_stack_close():
    """A stack that cannot unwind is recorded instead of aborting the teardown."""
    client = McpClient(_in_memory_config())
    recorder = _CloseRecorder()
    client._client = recorder
    stack = _StackRecorder(OSError("process group already gone"))
    client._stack = stack

    await client.disconnect()

    assert (
        client.last_error == "stack close failed: OSError: process group already gone"
    )
    assert recorder.closed is True
    assert stack.closed is True


@pytest.mark.asyncio
async def test_force_close_keeps_both_errors_when_both_handles_fail():
    """Neither teardown error may be lost behind the other."""
    client = McpClient(_in_memory_config())
    client._client = _CloseRecorder(RuntimeError("client wedged"))
    client._stack = _StackRecorder(OSError("stack wedged"))

    await client.disconnect()

    assert client.last_error == (
        "close failed: RuntimeError: client wedged; "
        "stack close failed: OSError: stack wedged"
    )


@pytest.mark.asyncio
async def test_disconnect_stays_idempotent_after_a_failing_close(monkeypatch):
    """A failing teardown still leaves the object clean enough to be retried."""
    client = McpClient(_in_memory_config())
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(_server()))
    assert await client.connect() is True
    assert client.tools

    client._client = _CloseRecorder(RuntimeError("transport wedged"))

    await client.disconnect()

    assert client.state is McpConnectionState.DISCONNECTED
    assert client._client is None
    assert client._stack is None
    assert client.tools == []
    assert "transport wedged" in client.last_error

    # The second teardown is a no-op, not a retry: the error stays recorded.
    await client.disconnect()

    assert client._client is None
    assert client._stack is None
    assert client.state is McpConnectionState.DISCONNECTED
    assert client.last_error == "close failed: RuntimeError: transport wedged"


@pytest.mark.asyncio
async def test_disconnect_on_a_never_connected_client_changes_nothing():
    """Tearing down a client that never opened must not invent an error."""
    client = McpClient(_in_memory_config())
    client._state = McpConnectionState.FAILED
    client._last_error = "connect never got that far"

    await client.disconnect()

    assert client.state is McpConnectionState.FAILED
    assert client.last_error == "connect never got that far"
    assert client._client is None
    assert client._stack is None
    assert client.tools == []


@pytest.mark.asyncio
async def test_connect_recovers_when_no_handle_was_ever_created(monkeypatch):
    """A transport that cannot even be built leaves nothing to close, cleanly."""

    def _explode():
        raise RuntimeError("bad transport arguments")

    client = McpClient(_in_memory_config())
    monkeypatch.setattr(client, "_build_fastmcp_client", _explode)

    assert await client.connect() is False

    assert client.state is McpConnectionState.FAILED
    assert client.last_error == "RuntimeError: bad transport arguments"
    assert client._client is None
    assert client._stack is None
    assert client.tools == []
