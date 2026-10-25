"""Error types for the MitoCode command line interface."""

from mitocode.core.exception.mitocode_exception import MitoCodeBaseError


class MitoCodeCliError(MitoCodeBaseError):
    """Base class for all command line interface errors."""


class MitoCodeCliConfigError(MitoCodeCliError):
    """Raised when CLI or provider configuration is invalid.

    The message is shown to the user in an error panel and must never contain
    credentials or other secret values.
    """


__all__ = [
    "MitoCodeCliError",
    "MitoCodeCliConfigError",
]
