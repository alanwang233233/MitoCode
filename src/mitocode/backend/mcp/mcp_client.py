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
import re
from contextlib import AsyncExitStack
from enum import Enum
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError

from mitocode.backend.exception.mitocode_exception import MitoCodeMcpToolCallError
from mitocode.backend.mcp.mcp_config import McpServerConfig

#: Stands in for an argument payload that has been cut out of error text.
_REDACTED = "<redacted>"

#: The argument payload pydantic quotes back after rejecting a value.  A server
#: that validates its input hands the offending value straight to the client in
#: its error text, and that value is whatever the model sent — routinely a token
#: or a file's contents.  The payload runs up to ``input_type=`` when pydantic
#: supplies that trailing context and to the end of the line when it does not;
#: ``.*`` is greedy so a value that itself contains the text ``", input_type="``
#: is consumed up to the real suffix instead of leaking its tail.  ``.`` does not
#: cross a newline, so one error's payload never eats the next error's report.
_ARGUMENT_PAYLOAD = re.compile(
    r"input_value=(?:(?P<payload>.*)(?P<suffix>, input_type=[^\]\n]*)|[^\n]*)"
)

#: An endpoint quoted inside free-form error text, in either of the two shapes
#: that can carry a credential.  The first branch is a full URL — what httpx, a
#: proxy or fastmcp's own transport ``repr`` renders — and runs to the first
#: character that cannot belong to one: whitespace, the quotes an error wraps a
#: URL in, or the brackets a repr or traceback puts around it.  The second is
#: the scheme-less reference a JSON-RPC server echoes back in its own error text
#: (``/mcp?access_token=…``), which is precisely the shape the MCP SDK's own
#: URL scrubbing cannot see because it only ever parses URLs it is handed.  That
#: branch requires a ``key=`` in the query so ordinary prose containing a
#: question mark is not mistaken for an endpoint.
_ENDPOINT_RUN = re.compile(
    r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>)\]}`]+"
    r"|(?:/[^\s'\"<>)\]}`]*|[A-Za-z0-9_.~-]+)\?[A-Za-z0-9_.~-]+=[^\s'\"<>)\]}`]*"
)

#: The three parts of a URL that can carry a credential.  Each matches only the
#: first occurrence, and each is applied to a run :data:`_ENDPOINT_RUN` already
#: bounded, so none of them can reach past the endpoint into the sentence.
_ENDPOINT_USERINFO = re.compile(r"://[^/?#@]*@")
_ENDPOINT_QUERY = re.compile(r"\?[^#\s]*")
_ENDPOINT_FRAGMENT = re.compile(r"#[^\s]*")


def _redact_arguments(text: str) -> str:
    """Cut argument payloads out of error text while keeping it diagnosable.

    Only the value is removed: the tool, the server, the offending field name
    and the description of what was expected all survive, so the message still
    says *what* was wrong even though it no longer says *with what*.

    Args:
        text: Error text as reported, possibly quoting the call's arguments.

    Returns:
        The same text with every ``input_value=`` payload replaced by
        :data:`_REDACTED`.
    """
    return _ARGUMENT_PAYLOAD.sub(rf"input_value={_REDACTED}\g<suffix>", text)


def _scrub_endpoint_run(match: "re.Match[str]") -> str:
    """Rewrite one matched endpoint with its credential-bearing parts removed.

    Args:
        match: One :data:`_ENDPOINT_RUN` match, either a full URL or a
            scheme-less reference.

    Returns:
        The endpoint with its userinfo, query and fragment replaced by
        :data:`_REDACTED`.  Scheme, host, port and path are kept, and so is
        the fact that something was there at all.
    """
    run = match.group(0)
    if "://" not in run:
        # A scheme-less reference: only a query can have been scrubbed, because
        # userinfo needs an authority to live in and there is none.
        return f"{run.split('?', 1)[0]}?{_REDACTED}"

    scrubbed = _ENDPOINT_USERINFO.sub(f"://{_REDACTED}@", run, count=1)
    # The query stops at a '#' so that a fragment stays a fragment rather than
    # being swallowed into the query's redaction.
    scrubbed = _ENDPOINT_QUERY.sub(f"?{_REDACTED}", scrubbed, count=1)
    return _ENDPOINT_FRAGMENT.sub(f"#{_REDACTED}", scrubbed, count=1)


def _scrub_endpoint(text: str) -> str:
    """Cut credentials out of any endpoint quoted in *text*.

    A remote MCP endpoint may legitimately carry a secret in its userinfo, its
    query string or its fragment — ``?access_token=…`` is common enough that
    :class:`~mitocode.backend.mcp.mcp_config.McpServerConfig` has to accept it
    — and a handshake failure routinely quotes the endpoint it was made
    against, in text that reaches the TUI error pane and the log.  The MCP SDK
    strips the same parts from the URLs *it* builds, but nothing strips them
    from text a remote server chose to write.

    The scrub is deliberately URL-shaped rather than blanket: host, path and
    the exception's own type survive, so the message still says which endpoint
    failed, and a message that merely mentions a host, or nothing at all, is
    left byte-identical.

    Args:
        text: Error text as reported, possibly quoting an endpoint.

    Returns:
        The same text with every endpoint's userinfo, query and fragment
        replaced by :data:`_REDACTED`.
    """
    return _ENDPOINT_RUN.sub(_scrub_endpoint_run, text)


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

        No deadline is set on the client itself.  ``Client(timeout=...)`` becomes
        the session's default for *every* request, and the one request that
        carries no deadline of its own is the ``list_tools`` discovery inside
        :meth:`connect` — so with ``timeout`` below ``connect_timeout`` the
        discovery would be bounded by the wrong deadline and report the wrong
        one.  Both callers apply their own instead: :meth:`connect` wraps the
        handshake and the discovery in ``asyncio.wait_for``, and
        :meth:`call_tool` passes ``timeout`` to the request itself.
        """
        return Client(
            self._config.build_transport(),
            name=self._config.name,
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

        Cancellation is the same story with a different trigger: ``Ctrl-C``, a
        TUI shutdown or an enclosing ``wait_for`` delivers
        :class:`asyncio.CancelledError`, which derives from ``BaseException``
        and so bypasses an ``except Exception`` handler entirely.  It is caught,
        cleaned up after and then re-raised — swallowing a cancellation is not
        this method's call to make, but orphaning a subprocess is not an option
        either.

        Returns:
            ``True`` when the client is connected and its tools are cached,
            ``False`` otherwise.

        Raises:
            asyncio.CancelledError: If the caller cancels the connection.
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
        except BaseException as exc:  # pylint: disable=broad-except
            # Cleanup runs outside the timed-out await, or the teardown itself
            # would be cancelled and the subprocess orphaned after all.  The
            # shield keeps a *repeat* cancellation — a second Ctrl-C, or an
            # enclosing wait_for firing as well — from interrupting it too.
            cleanup_errors = await asyncio.shield(self._force_close())
            self._state = McpConnectionState.FAILED
            self._last_error = f"{type(exc).__name__}: {_scrub_endpoint(str(exc))}"
            if cleanup_errors:
                # Appended, never substituted.  The first half is what ended the
                # connect; the rest is the teardown saying it could not release
                # the transport either, which on the timeout path is the whole
                # warning — a subprocess that may still be running.  Discarding it
                # the way this used to leaves the one diagnostic an operator needs.
                self._last_error += "; " + "; ".join(cleanup_errors)
            if isinstance(exc, Exception):
                return False
            raise

        self._state = McpConnectionState.CONNECTED
        self._last_error = ""
        return True

    async def _force_close(self) -> List[str]:
        """Tear the connection down, swallowing but reporting cleanup errors.

        Shared by the failure path of :meth:`connect` and by
        :meth:`disconnect` so there is exactly one way to release a connection.
        The handles are dropped before the awaits so that a transport which
        raises while closing still leaves the object in a clean state.

        The collected errors are *returned* rather than written to
        :attr:`last_error` here, because the two callers owe that attribute
        different things: :meth:`disconnect` records them, while
        :meth:`connect` is about to overwrite it with the failure that triggered
        the cleanup and can only append them to that.

        Returns:
            One description per handle that could not be released, empty when
            the teardown was clean.
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
                errors.append(
                    f"close failed: {type(exc).__name__}: {_scrub_endpoint(str(exc))}"
                )
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(
                    f"stack close failed: {type(exc).__name__}: "
                    f"{_scrub_endpoint(str(exc))}"
                )
        return errors

    async def disconnect(self) -> None:
        """Close the connection if one is open.  Idempotent.

        Errors raised while closing are swallowed and recorded in
        :attr:`last_error` rather than propagated, because a failing teardown
        must not prevent the remaining servers from being shut down — *unless*
        they are not an :class:`Exception`.  A :class:`BaseException` that is
        neither one nor a cancellation — a framework's shutdown signal — keeps
        travelling, because recording it away would silently downgrade the
        second Ctrl-C or the enclosing cancellation that caused it, and
        :attr:`last_error` stays untouched rather than blaming a cleanup
        problem that never happened.

        The teardown is shielded for the same reason :meth:`connect`'s is: a
        *repeat* cancellation — a second Ctrl-C, or an enclosing ``wait_for``
        firing in the same tick — must not interrupt ``Client.close()`` midway
        and leave stdio's detached connect task holding the child process alive.
        Only the fact that ``close`` ran to completion tells a shielded teardown
        from a plain await; both leave the handles cleared.

        :attr:`state` is set before that await rather than after it, for the
        same reason :meth:`_force_close` drops the handles before its own
        awaits: the transport is gone from the instant the teardown starts, so a
        caller cancelled part-way through it must not find a client still
        reporting ``CONNECTED``.  That is not cosmetic — :meth:`connect` returns
        early for a connected client, so the state left behind would hand the
        next reconnect a client with no transport under it.
        """
        if self._stack is None and self._client is None:
            return
        self._state = McpConnectionState.DISCONNECTED
        errors = await asyncio.shield(self._force_close())
        if errors:
            self._last_error = "; ".join(errors)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke *tool_name* on the connected server and render the result as text.

        Whatever the server said about a failure is carried over as it wrote
        it, which is what makes the error diagnosable — minus two scrubs, both
        applied because this text is rendered into the TUI error pane and the
        log: the arguments it quoted back, and the endpoint it names when a
        call is refused.  Host, path and the exception's own type survive both.

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
            # The server's own text is the most likely place for either kind of
            # secret to reappear: an argument-validation failure quotes the
            # value it rejected, and a refused call routinely names the endpoint
            # it was made against.  Both are cut before anything is
            # interpolated.
            raise MitoCodeMcpToolCallError(
                f"MCP tool '{tool_name}' on server '{server}' failed: "
                f"{_scrub_endpoint(_redact_arguments(str(exc)))}"
            ) from exc
        except asyncio.TimeoutError as exc:
            # Scrubbed like the two branches above even though the only text the
            # server chose here is the tool name: the rule is applied where the
            # message is built, not where the leak was noticed, so a later edit
            # that quotes something else here cannot reintroduce one silently.
            raise MitoCodeMcpToolCallError(
                _scrub_endpoint(
                    f"MCP tool '{tool_name}' on server '{server}' timed out "
                    f"after {timeout} seconds"
                )
            ) from exc
        except MitoCodeMcpToolCallError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            # Arguments are never echoed here either: they routinely carry file
            # contents and credentials, and this text reaches the TUI error pane
            # and logs.  A local validation failure can quote them just as a
            # server-side one does, so the same redaction applies -- and a
            # transport failure quotes the endpoint for the same reason a
            # handshake does, so the endpoint scrub applies here too.
            raise MitoCodeMcpToolCallError(
                f"MCP tool '{tool_name}' on server '{server}' raised "
                f"{type(exc).__name__}: "
                f"{_scrub_endpoint(_redact_arguments(str(exc)))}"
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
