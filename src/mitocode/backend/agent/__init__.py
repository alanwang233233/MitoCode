"""Agent subsystem for MitoCode.

Provides session/state containers, the streaming runtime, the orchestrator,
and the session factory used to build agent sessions.
"""

from mitocode.backend.tool.tool_manager import ToolManager, ToolSpec, declare_tool
from .agent_node import AgentNode
from .agent_orchestrator import AgentOrchestrator
from .agent_runtime import AgentRuntime
from .agent_state import AgentState
from .agent_session import (
    AgentSession,
    _MessageBlock,
    _PlainTextMessageBlock,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)
from .legacy_agent import LegacyAgent
from .session_factory import SessionFactory

__all__ = [
    "LegacyAgent",
    "AgentNode",
    "AgentOrchestrator",
    "AgentRuntime",
    "AgentSession",
    "AgentState",
    "SessionFactory",
    "ToolManager",
    "ToolSpec",
    "declare_tool",
    "_MessageBlock",
    "_PlainTextMessageBlock",
    "_ToolCallMessageBlock",
    "_ToolResultMessageBlock",
]
