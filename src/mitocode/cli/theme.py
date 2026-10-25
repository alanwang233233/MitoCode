"""Semantic Rich theme for the MitoCode command line interface.

Every style the CLI uses has a semantic name here, so colour literals never
appear in rendering code. Consumers create their console with
:data:`MITOCODE_THEME` and then refer to names such as ``mito.accent``.
"""

from __future__ import annotations

from rich.theme import Theme

THEME_STYLE_NAMES: tuple[str, ...] = (
    "mito.accent",
    "mito.success",
    "mito.error",
    "mito.warn",
    "mito.muted",
    "mito.code",
    "mito.diff.add",
    "mito.diff.remove",
)
"""Semantic style names the CLI may reference, in documentation order."""

ACCENT_COLOR = "#D97757"
"""Warm brand colour used for headings, tool names and links."""

SUCCESS_COLOR = "#7FB77E"
"""Green used for successful outcomes and added diff lines."""

ERROR_COLOR = "#E06C75"
"""Red used for failures and removed diff lines."""

WARN_COLOR = "#E5C07B"
"""Amber used for recoverable warnings such as unadvertised models."""

MUTED_COLOR = "#6B7280"
"""Grey used for secondary text, hints and truncation markers."""

CODE_COLOR = "#CBA6F7"
"""Purple used for inline code, commands and parameters."""

MITOCODE_THEME = Theme(
    {
        "mito.accent": ACCENT_COLOR,
        "mito.success": SUCCESS_COLOR,
        "mito.error": ERROR_COLOR,
        "mito.warn": WARN_COLOR,
        "mito.muted": MUTED_COLOR,
        "mito.code": CODE_COLOR,
        "mito.diff.add": SUCCESS_COLOR,
        "mito.diff.remove": ERROR_COLOR,
    }
)
"""Theme passed to every :class:`rich.console.Console` the CLI creates."""

__all__ = [
    "ACCENT_COLOR",
    "CODE_COLOR",
    "ERROR_COLOR",
    "MITOCODE_THEME",
    "MUTED_COLOR",
    "SUCCESS_COLOR",
    "THEME_STYLE_NAMES",
    "WARN_COLOR",
]
