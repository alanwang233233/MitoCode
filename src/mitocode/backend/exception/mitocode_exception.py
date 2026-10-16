"""Custom exception hierarchy for MitoCode.

All application errors derive from :class:`MitoCodeBaseError`, organized
into workspace-related and tool-related branches.
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


class MitoCodeWorkspaceError(MitoCodeBaseError):
    """Base class for workspace-related errors."""


class MitoCodeWorkspaceInitError(MitoCodeWorkspaceError):
    """Raised when a workspace cannot be loaded or initialized."""


class MitoCodeWorkspaceCreateError(MitoCodeWorkspaceError):
    """Raised when creating a new workspace fails."""


class MitoCodeWorkspaceDumpError(MitoCodeWorkspaceError):
    """Raised when persisting the workspace configuration fails."""


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
    """Raised when an MCP server connection cannot be established or is lost."""


class MitoCodeMcpToolCallError(MitoCodeMcpError):
    """Raised when an MCP tool call fails or returns an error result."""
