"""Built-in tools for MitoCode: workspace-confined file, shell, and agent tools.

The package is assembled bottom-up: ``approval`` is the approval protocol,
``context`` holds the state every tool reads instead of reaching for
globals, ``paths`` is the containment chokepoint, ``errors`` the one error
boundary every tool returns through, ``call_dsl`` and ``background`` the
plumbing behind the Background tools, and the four tool modules carry the
eleven tool coroutines themselves.  ``register`` is the one call that
publishes them all onto a ``ToolManager``.

This module is the package's public surface: the names re-exported here are
the API the rest of MitoCode -- and the tests -- build against, and the
modules beneath them stay free to rearrange their internals.
"""

from mitocode.core.tool.builtin.approval import (
    AlwaysAllow,
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    DenyAll,
)
from mitocode.core.tool.builtin.context import (
    ToolContext,
    current,
    current_agent_context,
    current_agent_id,
    set_current,
)
from mitocode.core.tool.builtin.errors import (
    MitoCodeBuiltinToolError,
    tool_boundary,
)
from mitocode.core.tool.builtin.register import register_builtin_tools

__all__ = [
    "AlwaysAllow",
    "ApprovalDecision",
    "ApprovalRequest",
    "Approver",
    "DenyAll",
    "MitoCodeBuiltinToolError",
    "ToolContext",
    "current",
    "current_agent_context",
    "current_agent_id",
    "register_builtin_tools",
    "set_current",
    "tool_boundary",
]
