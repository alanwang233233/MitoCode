"""Mutable runtime state shared by the CLI's REPL, event pump and commands.

:class:`CliState` is the single holder of the objects that live for one CLI
process: the parsed arguments, the workspace, the tool registry, the provider,
the orchestrator, the MCP manager and the optional UI and approver.  Nothing in
this module talks to a terminal or starts a task; it is a plain dataclass plus
two derived values the REPL and the commands ask it for.

The counters (:attr:`CliState.turn_count`, :attr:`CliState.tool_call_count` and
:attr:`CliState.total_chars`) are maintained by the REPL and the event pump.
:meth:`CliState.estimate_tokens` deliberately recomputes from the session's
message blocks instead of trusting a running total: provider responses carry no
usage, and a value derived from the history cannot drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from mitocode.cli.approval import CliApprover
    from mitocode.cli.args import CliArgs
    from mitocode.cli.ui import CliUI
    from mitocode.core.agent.agent_orchestrator import AgentOrchestrator
    from mitocode.core.agent.agent_session import AgentSession
    from mitocode.core.chat_provider.chat_provider_openai import ChatProviderOpenAI
    from mitocode.core.mcp.mcp_manager import McpManager
    from mitocode.core.skills.skills_manager import SkillsManager
    from mitocode.core.tool.builtin.context import ToolContext
    from mitocode.core.tool.tool_manager import ToolManager
    from mitocode.core.workspace.workspace import Workspace


def _block_char_count(block: Any) -> int:
    """Count the characters one history block contributes to the context.

    The internal session blocks are duck-typed here so a test fake made of
    ``SimpleNamespace`` objects works exactly like a real
    :class:`~mitocode.core.agent.agent_session._PlainTextMessageBlock`.  Plain
    text contributes its message, tool results their content, and tool calls
    their serialized arguments.

    Args:
        block: One entry from ``AgentSession.messages``, or a mapping standing
            in for one.

    Returns:
        The number of characters the block carries; ``0`` for an object that
        exposes none of the known fields.
    """
    if isinstance(block, dict):
        fields = ("message", "content", "arguments_string", "reasoning_content")
        return sum(
            len(value)
            for value in (block.get(name) for name in fields)
            if isinstance(value, str)
        )

    total = 0
    message = getattr(block, "message", None)
    if isinstance(message, str):
        total += len(message)
    content = getattr(block, "content", None)
    if isinstance(content, str):
        total += len(content)
    reasoning = getattr(block, "reasoning_content", None)
    if isinstance(reasoning, str):
        total += len(reasoning)
    arguments = getattr(block, "arguments_string", None)
    if isinstance(arguments, str) and arguments:
        total += len(arguments)
    else:
        # A finished tool call keeps the raw JSON in ``arguments_string``; a
        # fake or a partially built block may only have the parsed mapping.
        parameter = getattr(block, "parameter", None)
        if isinstance(parameter, dict) and parameter:
            total += len(str(parameter))
    return total


@dataclass
class CliState:  # pylint: disable=too-many-instance-attributes
    """Runtime objects and counters for one interactive CLI session.

    Attributes:
        args: Parsed command line arguments; ``Repl`` reads
            ``args.initial_prompt`` from here.
        workspace: Workspace the agent operates in.
        tool_manager: Registry backing ``/tools`` and MCP refreshes.
        provider: Chat provider; ``/model`` reads and writes its
            ``default_model``.
        orchestrator: Agent registry used by ``/clear``, ``/agents`` and the
            event pump's tree events.
        mcp_manager: Connected MCP manager, or ``None`` when MCP is disabled.
        tool_context: Built-in tools' shared context, kept for inspection.
        skills_manager: Skills manager over the workspace root, with its
            startup discovery already run; ``/skills`` reads and refreshes it.
            Always wired by the app; ``None`` only in tests.
        ui: Terminal surface, so objects that only receive a state can still
            print.  ``CommandContext`` carries the same instance.
        approver: Approval policy/cache, or ``None`` when approvals are not
            interactive.  ``CommandContext`` carries the same instance.
        log_path: File receiving the CLI's logs, shown by ``/log``.
        main_id: Agent id of the interactive main session.  ``/clear`` replaces
            it with the id of the new session.
        turn_count: Completed turns in this process.
        tool_call_count: Tool calls made by the main agent in this process.
        total_chars: Characters streamed by the main agent, used for the
            end-of-turn token estimate.
        agents: Last tree snapshot received from the orchestrator.
        mcp_error: Non-fatal MCP startup error, shown by ``/mcp`` and
            ``/status``.
        verbose: Whether full tool output is shown; ``/verbose`` toggles this
            together with :attr:`ui`'s flag.
    """

    args: "CliArgs"
    workspace: "Workspace"
    tool_manager: "ToolManager"
    provider: "ChatProviderOpenAI"
    orchestrator: "AgentOrchestrator"
    mcp_manager: Optional["McpManager"] = None
    tool_context: Optional["ToolContext"] = None
    skills_manager: Optional["SkillsManager"] = None
    ui: Optional["CliUI"] = None
    approver: Optional["CliApprover"] = None
    log_path: Path = field(default_factory=lambda: Path("cli.log"))
    main_id: int = 0
    turn_count: int = 0
    tool_call_count: int = 0
    total_chars: int = 0
    agents: List[dict] = field(default_factory=list)
    mcp_error: str = ""
    verbose: bool = False

    def main_session(self) -> "AgentSession":
        """Return the interactive main agent's session.

        Returns:
            The session registered under :attr:`main_id`.

        Raises:
            RuntimeError: When the orchestrator has no session under
                :attr:`main_id`, for example after a failed ``/clear``.
        """
        session = self.orchestrator.get_session(self.main_id)
        if session is None:
            raise RuntimeError(f"Main agent #{self.main_id} does not exist")
        return session

    def estimate_tokens(self) -> int:
        """Estimate the main session's token count from its history.

        Providers stream no usage figures, so the CLI uses the usual rough
        approximation of four characters per token.  The value is recomputed
        from the message blocks each time and never cached, which keeps it
        consistent with the conversation the provider will actually receive.

        Returns:
            Whole estimated tokens, ``0`` when the main session is missing.
        """
        session = self.orchestrator.get_session(self.main_id)
        if session is None:
            return 0
        messages = getattr(session, "messages", None) or []
        characters = sum(_block_char_count(block) for block in messages)
        return characters // 4


@dataclass(frozen=True)
class CommandContext:
    """Everything one slash-command handler may touch.

    Attributes:
        state: Mutable process-wide state.
        ui: Terminal surface the command prints through.
        approver: Approval object for ``/permissions``, or ``None``.
    """

    state: CliState
    ui: "CliUI"
    approver: Optional["CliApprover"] = None


__all__ = ["CliState", "CommandContext"]
