"""Tool management subsystem for MitoCode.

Provides decorator-based tool declaration, automatic JSON Schema generation
from Python type annotations, and dual human/agent documentation support.
"""

from .tool_manager import ToolManager, ToolSpec, declare_tool

__all__ = [
    "ToolManager",
    "ToolSpec",
    "declare_tool",
]
