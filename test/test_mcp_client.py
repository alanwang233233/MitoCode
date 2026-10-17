import asyncio
import os
import sys
from typing import Optional

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpToolCallError
from mitocode.backend.mcp.mcp_client import (
    McpClient,
    McpConnectionState,
    _redact_arguments,
)
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


class _HandshakeHangingClient:
    """Fake client whose handshake never completes and whose close is slow.

    The slow ``close`` is the whole point: it opens a window in which a second
    cancellation can land while the teardown started by the first one is still
    awaiting, which is the only way to tell a shielded teardown from an
    unshielded one.
    """

    def __init__(self) -> None:
        """Arm the events a test needs to synchronise on."""
        self.entered = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()

    async def __aenter__(self):
        """Announce that the handshake started, then wait forever."""
        self.entered.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *exc_info):
        """Never claim to have suppressed anything."""
        return False

    async def close(self) -> None:
        """Record that teardown began, yield to the loop, then record it ended."""
        self.close_started.set()
        await asyncio.sleep(0.05)
        self.close_finished.set()


class _ShutdownSignal(BaseException):
    """A shutdown request that is neither an ``Exception`` nor a cancellation.

    Stand-in for the third ``BaseException`` an awaited operation can raise: a
    framework's own "stop everything" signal.  ``KeyboardInterrupt`` and
    ``SystemExit`` cannot be used here because asyncio re-raises those out of
    the event loop itself instead of delivering them to the awaiting coroutine.
    """


async def _cancel_twice_during_teardown(
    client: McpClient, built: _HandshakeHangingClient
) -> None:
    """Cancel a hanging ``connect`` again while its own teardown is in flight.

    The second cancellation is timed off ``close_started`` so it cannot race
    the teardown: without it the test would pass whether or not the teardown is
    shielded.

    Args:
        client: The client whose :meth:`connect` must be cancelled twice.
        built: The fake transport installed as the client's build seam.
    """
    task = asyncio.ensure_future(client.connect())
    await asyncio.wait_for(built.entered.wait(), timeout=5)
    task.cancel()

    async def _cancel_mid_teardown() -> None:
        await asyncio.wait_for(built.close_started.wait(), timeout=5)
        task.cancel()

    canceller = asyncio.ensure_future(_cancel_mid_teardown())
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(canceller, timeout=5)


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
async def test_connect_cancellation_closes_the_transport_and_still_propagates():
    """Cancelling a handshake must tear the transport down, then keep cancelling.

    ``CancelledError`` derives from ``BaseException``, so it walks straight past
    an ``except Exception`` handler: Ctrl-C, a TUI shutdown or an enclosing
    ``wait_for`` would leave ``_client``/``_stack`` populated and stdio's
    detached connect task holding the child process alive.  Swallowing the
    cancellation would be wrong too — cleanup, then propagate.
    """
    client = McpClient(_slow_config(1.0))
    entered = asyncio.Event()

    class _HangingClient:
        """Async context manager whose __aenter__ never completes."""

        closed = False

        async def __aenter__(self):
            entered.set()
            await asyncio.Event().wait()

        async def __aexit__(self, *exc_info):
            return False

        async def close(self):
            type(self).closed = True

    built = _HangingClient()
    client._build_fastmcp_client = lambda: built

    task = asyncio.ensure_future(client.connect())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert built.closed is True, "cancellation orphaned the transport"
    assert client._client is None
    assert client._stack is None
    assert client.tools == []
    assert client.state is McpConnectionState.FAILED


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
async def test_tool_error_message_never_echoes_the_arguments_it_validated(
    monkeypatch,
):
    """A schema mistake must not turn the rejected argument into an error message.

    fastmcp re-raises pydantic's validation text as a ``ToolError``, and every
    ``input_value=`` segment in it is the value the model just sent.  This is
    the *common* failure — the model calls a tool with the wrong shape — and the
    generic ``except Exception`` path cannot help because it never sees it.
    """
    secret = "sk-TOOL-ERROR-DO-NOT-LEAK"
    server = FastMCP("probe")

    @server.tool
    def login(token: str) -> str:
        """Accept a token."""
        return "ok"

    client = McpClient(_in_memory_config())
    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(server))
    assert await client.connect() is True

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("login", {"token": [secret]})

    message = str(excinfo.value)
    assert secret not in message, f"the rejected argument leaked: {message}"
    # Redaction must cost diagnosis nothing: which tool, on which server, which
    # field, and what was expected of it all have to survive.
    assert "login" in message
    assert "probe" in message
    assert "token" in message
    assert "should be a valid string" in message
    await client.disconnect()


def test_redaction_covers_an_input_value_with_no_type_suffix():
    """Text that quotes an argument without pydantic's type suffix is still cut.

    ``input_type=`` is the common tail, but the rule must not depend on it: an
    argument payload is never safe to echo, whatever surrounds it.
    """
    redacted = _redact_arguments("server said no: input_value=sk-BARE-DO-NOT-LEAK")

    assert "sk-BARE-DO-NOT-LEAK" not in redacted
    assert "server said no" in redacted


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


@pytest.mark.asyncio
async def test_connect_repeat_cancellation_still_completes_the_transport_teardown():
    """A second cancellation must not abort the teardown the first one started.

    ``connect`` runs its cleanup outside the timed-out await and shields it, so
    a repeat cancellation -- a second Ctrl-C, or an enclosing ``wait_for``
    firing in the same tick -- cannot interrupt ``Client.close()`` midway and
    leave stdio's detached connect task holding the child process alive.  Only
    the fact that ``close`` *ran to completion* distinguishes the shield from a
    plain await; both leave the handles cleared.
    """
    client = McpClient(_in_memory_config())
    built = _HandshakeHangingClient()
    client._build_fastmcp_client = lambda: built

    await _cancel_twice_during_teardown(client, built)

    await asyncio.wait_for(built.close_finished.wait(), timeout=5)
    assert client._client is None
    assert client._stack is None
    assert client.tools == []


@pytest.mark.asyncio
async def test_connect_repeat_cancellation_skips_the_failed_bookkeeping():
    """A repeat cancellation propagates before ``connect`` records the failure.

    The cancellation is re-raised out of the shielded await, so the two
    statements that would set ``FAILED`` and record the error never run.  The
    object is still consistent -- no client, no stack, no tools -- but a reader
    of :attr:`last_error` must not expect to find the cancellation logged here.
    """
    client = McpClient(_in_memory_config())
    built = _HandshakeHangingClient()
    client._build_fastmcp_client = lambda: built

    await _cancel_twice_during_teardown(client, built)

    assert client.state is McpConnectionState.DISCONNECTED
    assert client.last_error == ""


@pytest.mark.asyncio
async def test_connect_shutdown_signal_closes_the_transport_and_still_propagates():
    """A ``BaseException`` that is not an ``Exception`` must be cleaned up after.

    ``CancelledError`` is only the best-known member of that family.  Catching
    ``(Exception, CancelledError)`` would pass every cancellation test while
    leaving an application shutdown signal to orphan the subprocess, so the
    handler has to be keyed on ``BaseException`` itself.
    """
    client = McpClient(_in_memory_config())

    class _ShuttingDownClient:
        """Client whose handshake fails with a shutdown signal."""

        def __init__(self) -> None:
            """Start with the transport still open."""
            self.closed = False

        async def __aenter__(self):
            """Abort the handshake the way a framework shutdown would."""
            raise _ShutdownSignal("application is shutting down")

        async def __aexit__(self, *exc_info):
            """Never claim to have suppressed anything."""
            return False

        async def close(self) -> None:
            """Record that the transport was released."""
            self.closed = True

    built = _ShuttingDownClient()
    client._build_fastmcp_client = lambda: built

    with pytest.raises(_ShutdownSignal):
        await client.connect()

    assert built.closed is True, "shutdown signal orphaned the transport"
    assert client.state is McpConnectionState.FAILED
    assert client.last_error == "_ShutdownSignal: application is shutting down"
    assert client._client is None
    assert client._stack is None
    assert client.tools == []


@pytest.mark.asyncio
async def test_connect_failure_reports_the_handshake_error_not_the_teardown_error():
    """A failing teardown must not replace the error that explains the failure.

    ``_force_close`` records cleanup problems in ``last_error`` and ``connect``
    then overwrites it with the exception that caused the failure.  The message a
    user reads is therefore the handshake error alone; the fact that teardown
    *also* failed is only visible in the teardown's own effects.
    """
    client = McpClient(_in_memory_config())

    class _ExplodingHandshake:
        """Client that fails to connect and cannot be closed either."""

        def __init__(self) -> None:
            """Start with the transport still open."""
            self.closed = False

        async def __aenter__(self):
            """Refuse the handshake."""
            raise RuntimeError("handshake exploded")

        async def __aexit__(self, *exc_info):
            """Never claim to have suppressed anything."""
            return False

        async def close(self) -> None:
            """Attempt the teardown, which fails."""
            self.closed = True
            raise OSError("process group wedged")

    built = _ExplodingHandshake()
    client._build_fastmcp_client = lambda: built

    assert await client.connect() is False

    assert built.closed is True
    assert client.last_error == "RuntimeError: handshake exploded"
    assert client.state is McpConnectionState.FAILED


@pytest.mark.asyncio
async def test_connect_after_a_failure_clears_the_stale_error(monkeypatch):
    """A successful retry must not leave the previous failure in the report."""
    client = McpClient(_in_memory_config())

    class _ExplodingHandshake:
        """Client that never completes a handshake."""

        async def __aenter__(self):
            """Refuse the handshake."""
            raise RuntimeError("transient outage")

        async def __aexit__(self, *exc_info):
            """Never claim to have suppressed anything."""
            return False

        async def close(self) -> None:
            """Succeed at the teardown."""
            return None

    client._build_fastmcp_client = lambda: _ExplodingHandshake()
    assert await client.connect() is False
    assert client.last_error == "RuntimeError: transient outage"

    server = FastMCP("probe")

    @server.tool
    def ping() -> str:
        """Report that the server is alive."""
        return "pong"

    monkeypatch.setattr(client, "_build_fastmcp_client", lambda: Client(server))

    assert await client.connect() is True
    assert client.state is McpConnectionState.CONNECTED
    assert client.last_error == ""
    assert sorted(t.name for t in client.tools) == ["ping"]
    await client.disconnect()


def test_redaction_cuts_the_payload_up_to_the_last_injected_type_marker():
    """A value carrying pydantic's own suffix text is cut up to that decoy.

    The pattern consumes a payload up to ``, input_type=``; when the argument
    itself contains that text the greedy match runs to the *last* occurrence on
    the line.  Everything before the decoy is cut, which is the guarantee the
    module documents: text after the marker is indistinguishable from the
    framework's suffix and is deliberately kept so the report stays readable.
    """
    redacted = _redact_arguments("input_value='sk-PART1\", input_type=sk-PART2LEAK'")

    assert "sk-PART1" not in redacted
    assert redacted == "input_value=<redacted>, input_type=sk-PART2LEAK'"


def test_redaction_keeps_the_report_after_an_injected_marker():
    """Surrounding report text must survive a payload that fakes the suffix.

    Redaction that eats the sentence, the field name and the expected type
    would make the error undiagnosable -- and a test asserting only "the
    secret is gone" would pass just as happily against such an implementation.
    """
    redacted = _redact_arguments(
        "tool 'login' failed on field 'token': "
        'input_value=sk-DECOY", input_type=str; expected a valid string]'
    )

    assert "sk-DECOY" not in redacted
    assert redacted == (
        "tool 'login' failed on field 'token': "
        "input_value=<redacted>, input_type=str; expected a valid string]"
    )


def test_redaction_cuts_each_error_of_a_multiline_report_independently():
    """One payload must not eat the next error's report on a following line.

    ``.`` does not cross a newline, so a validation report naming two offending
    fields loses neither of them -- and each keeps the type pydantic expected.
    """
    report = (
        "2 validation errors for login\n"
        "token\n"
        "  Input should be a valid string [type=string_type, "
        "input_value='sk-FIRST', input_type=str]\n"
        "count\n"
        "  Input should be a valid integer [type=string_type, "
        "input_value='sk-SECOND', input_type=int]"
    )

    redacted = _redact_arguments(report)

    assert redacted == (
        "2 validation errors for login\n"
        "token\n"
        "  Input should be a valid string [type=string_type, "
        "input_value=<redacted>, input_type=str]\n"
        "count\n"
        "  Input should be a valid integer [type=string_type, "
        "input_value=<redacted>, input_type=int]"
    )


def test_redaction_handles_an_empty_payload():
    """An ``input_value=`` with nothing after it is still a payload."""
    assert _redact_arguments(
        "Input should be a valid string [type=string_type, input_value=, input_type=str]"
    ) == (
        "Input should be a valid string [type=string_type, "
        "input_value=<redacted>, input_type=str]"
    )
    assert _redact_arguments("nothing here: input_value=") == (
        "nothing here: input_value=<redacted>"
    )


def test_redaction_cuts_a_suffixless_payload_to_the_end_of_its_line():
    """With no ``, input_type=`` to stop at, the whole rest of the line is cut.

    Stopping at the next space instead would leave the tail of the value -- and
    a token is often longer than its first word.
    """
    assert _redact_arguments("input_value=sk-BARE extra words follow") == (
        "input_value=<redacted>"
    )
    assert _redact_arguments("input_value=sk-BARE extra words follow\nnext line") == (
        "input_value=<redacted>\nnext line"
    )


@pytest.mark.asyncio
async def test_call_tool_tool_error_redacts_the_payload_and_keeps_the_report():
    """The common validation failure must lose the value and keep everything else.

    This is the ``ToolError`` branch, which the generic handler cannot see, so
    it carries the full pydantic report: the tool, the server, the offending
    field and the expected type all have to remain, while the rejected value
    does not.
    """
    client = McpClient(_in_memory_config())
    secret = "sk-ADVERSARIAL-DO-NOT-LEAK"

    class _ValidatingCall:
        """Client that rejects its argument the way fastmcp reports it."""

        async def call_tool(self, name, arguments, **kwargs):
            raise ToolError(
                "1 validation error for login\n"
                "token\n"
                "  Input should be a valid string [type=string_type, "
                f"input_value='{secret}\", input_type=str]"
            )

    client._client = _ValidatingCall()

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("login", {"token": secret})

    message = str(excinfo.value)
    assert secret not in message, f"the rejected argument leaked: {message}"
    assert "MCP tool 'login' on server 'probe' failed" in message
    assert "token" in message
    assert "should be a valid string" in message
    assert "input_type=str" in message


@pytest.mark.asyncio
async def test_call_tool_unexpected_error_redacts_the_payload_and_keeps_the_report():
    """A local validation failure quotes the arguments just as a server one does.

    The generic handler renders the exception type before the text, so the
    redaction has to survive an extra prefix and must not cost the surrounding
    report anything.
    """
    client = McpClient(_in_memory_config())
    secret = "sk-LOCAL-VALIDATION-DO-NOT-LEAK"

    class _LocallyValidatingCall:
        """Client that fails locally while validating its arguments."""

        async def call_tool(self, name, arguments, **kwargs):
            raise ValueError(
                "field 'token' rejected\n"
                "  Input should be a valid string [type=string_type, "
                f"input_value='{secret}', input_type=str]"
            )

    client._client = _LocallyValidatingCall()

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("login", {"token": secret})

    message = str(excinfo.value)
    assert secret not in message, f"the rejected argument leaked: {message}"
    assert "MCP tool 'login' on server 'probe' raised ValueError" in message
    assert "token" in message
    assert "should be a valid string" in message
    assert "input_type=str" in message


@pytest.mark.asyncio
async def test_call_tool_forwards_the_configured_timeout_to_the_server_call():
    """The per-call deadline must reach the server call, not just the local wait.

    The deadline is applied twice on purpose: the outer ``wait_for`` bounds the
    local await and the ``timeout`` keyword bounds the request the transport has
    actually sent.  Dropping the keyword leaves a hanging server unbounded
    server-side while every local test still passes.
    """
    client = McpClient(
        McpServerConfig.parse(
            "slowtool", {"url": "https://x.invalid/mcp", "timeout": 12.5}
        )
    )
    seen = {}

    class _RecordingCall:
        """Client that records how it was called."""

        async def call_tool(self, name, arguments, **kwargs):
            seen["name"] = name
            seen["arguments"] = arguments
            seen["timeout"] = kwargs.get("timeout", "absent")
            return None

    client._client = _RecordingCall()

    await client.call_tool("echo", {"text": "hi"})

    assert seen == {"name": "echo", "arguments": {"text": "hi"}, "timeout": 12.5}


@pytest.mark.asyncio
async def test_call_tool_keeps_non_ascii_structured_data_readable():
    """Structured output must not be escaped into ``\\uXXXX`` noise.

    ``ensure_ascii`` defaults to true, which turns a Japanese or accented field
    into escape sequences; a result rendered that way still parses, so nothing
    downstream notices except the reader.
    """
    client = McpClient(_in_memory_config())

    class _Result:
        data = {"memo": "café", "city": "東京"}
        content = []

    class _Call:
        async def call_tool(self, name, arguments, **kwargs):
            return _Result()

    client._client = _Call()

    assert await client.call_tool("echo", {}) == '{"memo": "café", "city": "東京"}'


@pytest.mark.asyncio
async def test_call_tool_renders_an_unserialisable_result_value_with_str():
    """A result carrying a non-JSON value must render, not raise.

    ``json.dumps`` refuses a set, a handle or a custom object, and the failure
    would escape ``call_tool`` as a bare ``TypeError`` because rendering happens
    outside the error handling.  ``default=str`` is what keeps that reachable.
    """
    client = McpClient(_in_memory_config())

    class _Handle:
        """A value the model cannot receive as JSON."""

        def __str__(self) -> str:
            return "opaque-handle"

    class _Result:
        data = {"handle": _Handle(), "ids": 3}
        content = []

    class _Call:
        async def call_tool(self, name, arguments, **kwargs):
            return _Result()

    client._client = _Call()

    assert await client.call_tool("echo", {}) == (
        '{"handle": "opaque-handle", "ids": 3}'
    )


def test_redaction_cuts_the_last_payload_when_a_line_reports_several():
    """Every payload on a line must be cut, not just the first one.

    Matching greedily to the *last* ``, input_type=`` is what makes a payload
    containing a decoy marker safe: stopping at the first marker instead would
    leave the remainder of the line -- including the next payload -- untouched,
    so a hostile value that merely looks like the framework's suffix would hand
    the following argument straight back to the log.
    """
    redacted = _redact_arguments(
        "probe failed: input_value=sk-FIRST, input_type=str; "
        "input_value=sk-SECOND, input_type=int"
    )

    assert "sk-FIRST" not in redacted
    assert "sk-SECOND" not in redacted
    assert redacted == "probe failed: input_value=<redacted>, input_type=int"


def test_build_fastmcp_client_wires_the_configured_transport_and_name():
    """The real builder must use the config, not a hardcoded transport.

    Every connect test swaps this method out, so without this the transport a
    server is reached through -- HTTP versus stdio, and its command line -- is
    never exercised at all.
    """
    remote = McpClient(
        McpServerConfig.parse(
            "remote", {"url": "https://remote.invalid/mcp", "timeout": 7.5}
        )
    )._build_fastmcp_client()
    local = McpClient(
        McpServerConfig.parse(
            "local", {"command": "my-server", "args": ["--flag", "value"]}
        )
    )._build_fastmcp_client()

    assert isinstance(remote, Client)
    assert remote.name == "remote"
    assert type(remote.transport).__name__ == "StreamableHttpTransport"
    assert remote.transport.url == "https://remote.invalid/mcp"
    assert isinstance(local, Client)
    assert local.name == "local"
    assert type(local.transport).__name__ == "StdioTransport"
    assert local.transport.command == "my-server"
    assert local.transport.args == ["--flag", "value"]


@pytest.mark.asyncio
async def test_call_tool_names_the_server_and_the_tool_when_not_connected():
    """The not-connected report must say which server and which tool.

    With several servers configured, "not connected" on its own does not say
    which one to fix; the caller needs both halves of the address before it
    built one.
    """
    client = McpClient(_in_memory_config())

    with pytest.raises(MitoCodeMcpToolCallError) as excinfo:
        await client.call_tool("echo", {"text": "hi"})

    message = str(excinfo.value)
    assert "MCP server 'probe' is not connected" in message
    assert "cannot call tool 'echo'" in message
