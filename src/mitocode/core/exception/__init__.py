"""Custom exception hierarchy for MitoCode."""

from .mitocode_exception import (
    MitoCodeBaseError,
    MitoCodePathConvertError,
    MitoCodeWorkspaceError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceCreateError,
    MitoCodeWorkspaceDumpError,
    MitoCodeWorkspaceTooOldError,
    MitoCodeMigrateError,
    MitoCodeAppTooOldError,
    MitoCodeToolError,
    MitoCodeToolDefinitionError,
    MitoCodeChatProviderError,
    MitoCodeChatProviderConfigError,
    MitoCodeChatProviderModelError,
    MitoCodeChatProviderInitError,
)

__all__ = [
    "MitoCodeBaseError",
    "MitoCodePathConvertError",
    "MitoCodeWorkspaceError",
    "MitoCodeWorkspaceInitError",
    "MitoCodeWorkspaceCreateError",
    "MitoCodeWorkspaceDumpError",
    "MitoCodeWorkspaceTooOldError",
    "MitoCodeMigrateError",
    "MitoCodeAppTooOldError",
    "MitoCodeToolError",
    "MitoCodeToolDefinitionError",
    "MitoCodeChatProviderError",
    "MitoCodeChatProviderConfigError",
    "MitoCodeChatProviderModelError",
    "MitoCodeChatProviderInitError",
]
