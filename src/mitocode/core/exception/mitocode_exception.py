"""Custom exception hierarchy for MitoCode.

All application errors derive from :class:`MitoCodeBaseError`, organized into
one branch per subsystem: workspace, tool, chat provider, MCP, and skills.
"""

import builtins


class MitoCodeBaseError(builtins.Exception):
    """Base class for all MitoCode-specific errors."""

    def __init__(self, message="Unknown Error") -> None:
        """Initialise the error with an optional human-readable message.

        Args:
            message: Description of the error (default ``"Unknown Error"``).
        """
        self.message = message
        super().__init__(self.message)


class MitoCodePathConvertError(MitoCodeBaseError):
    """Raised when a path argument cannot be converted to a ``Path``."""


class MitoCodeConfigError(MitoCodeBaseError):
    """Base class for configuration-location errors."""


class MitoCodeConfigPathError(MitoCodeConfigError):
    """Raised when a configuration name cannot be turned into a safe path.

    Covers names that are empty, absolute, carry a drive or UNC prefix, or
    contain a ``..`` segment, as well as requests for a project path when no
    workspace was supplied.
    """


class MitoCodeConfigNotFoundError(MitoCodeConfigError, builtins.FileNotFoundError):
    """Raised when neither the project nor the global copy of a file exists.

    Also inherits ``FileNotFoundError``: callers that already handle a missing
    file keep working, while callers that care about the configuration layer
    can catch the more specific type.
    """


class MitoCodeConfigWriteError(MitoCodeConfigError):
    """Raised when a configuration file cannot be created or opened for writing."""


class MitoCodeWorkspaceError(MitoCodeBaseError):
    """Base class for workspace-related errors."""


class MitoCodeWorkspaceInitError(MitoCodeWorkspaceError):
    """Raised when a workspace cannot be loaded or initialized."""


class MitoCodeWorkspaceCreateError(MitoCodeWorkspaceError):
    """Raised when creating a new workspace fails."""


class MitoCodeWorkspaceDumpError(MitoCodeWorkspaceError):
    """Raised when persisting the workspace configuration fails."""


class MitoCodeWorkspaceLockError(MitoCodeWorkspaceError):
    """Raised when the workspace lock cannot be taken or released.

    This is a technical failure -- the lock file could not be created or
    opened -- and is deliberately distinct from "another process holds the
    lock", which is a conflict the user can decide about.
    """


class MitoCodeWorkspacePathError(MitoCodeWorkspaceError, builtins.ValueError):
    """Raised when a path argument resolves outside the workspace root.

    Dual-inherits ``ValueError``: a path rejected for being out of bounds is
    genuinely a bad argument, and ``test_workspace.py`` already asserts
    ``pytest.raises(ValueError)`` against the private alias of the method that
    raises it.  The ``"Path traversal detected"`` prefix is part of that match.
    """


class MitoCodeWorkspaceTooOldError(MitoCodeWorkspaceError):
    """Raised when the workspace version is older than supported and must be migrated."""


class MitoCodeAppTooOldError(MitoCodeBaseError):
    """Raised when the workspace is newer than this app; an upgrade is required."""


class MitoCodeMigrateError(MitoCodeWorkspaceError):
    """Raised when a workspace configuration migration step fails."""


class MitoCodeToolError(MitoCodeBaseError):
    """Base class for tool-related errors."""


class MitoCodeToolDefinitionError(MitoCodeToolError):
    """Raised when a tool definition or its type annotations are invalid."""


class MitoCodeChatProviderError(MitoCodeBaseError):
    """Base class for chat provider related errors."""


class MitoCodeChatProviderConfigError(MitoCodeChatProviderError):
    """Raised when chat provider configuration is invalid."""


class MitoCodeChatProviderModelError(MitoCodeChatProviderError):
    """Raised when requested model is not available."""


class MitoCodeChatProviderInitError(MitoCodeChatProviderError):
    """Raised when chat provider initialization fails."""


class MitoCodeMcpError(MitoCodeBaseError):
    """Base class for MCP-related errors."""


class MitoCodeMcpConfigError(MitoCodeMcpError):
    """Raised when an MCP server configuration entry is invalid."""


class MitoCodeMcpConnectionError(MitoCodeMcpError):
    """Reserved for when an MCP connection cannot be established or is lost.

    Nothing raises it yet, and that is deliberate rather than an oversight:
    :class:`~mitocode.core.mcp.mcp_client.McpClient` records a failed
    connection in ``state``/``last_error`` and returns ``False``, because one
    unreachable server must not abort the startup of the servers that work.
    Turning that into an exception belongs to whichever layer turns connection
    failure into a user-visible abort.
    """


class MitoCodeMcpToolCallError(MitoCodeMcpError):
    """Raised when an MCP tool call fails or returns an error result."""


class MitoCodeSkillError(MitoCodeBaseError):
    """Base class for Agent Skill related errors."""


class MitoCodeSkillDefinitionError(MitoCodeSkillError):
    """Raised when a SKILL.md file is structurally broken or its frontmatter is invalid."""


class MitoCodeSkillNotFoundError(MitoCodeSkillError):
    """Raised when a skill name is looked up outside the manager's catalogue."""


class MitoCodeSkillReadError(MitoCodeSkillError):
    """Raised when a catalogued skill's SKILL.md cannot be read from disk."""


class MitoCodeTemplateError(MitoCodeBaseError):
    """Base class for Agent Template related errors."""


class MitoCodeTemplateDefinitionError(MitoCodeTemplateError):
    """Raised when a template.json is structurally broken or a message block is invalid."""


class MitoCodeTemplateReadError(MitoCodeTemplateError):
    """Raised when a catalogued template's template.json cannot be read from disk."""
