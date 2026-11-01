"""Command line interface package for MitoCode.

Importing this package must stay side-effect free: the application module is
imported lazily inside :func:`main`.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive CLI.

    Args:
        argv: Argument list without the program name; ``None`` reads
            :data:`sys.argv`.

    Returns:
        Process exit code, ``0`` on a clean shutdown.
    """
    # Imported lazily so that importing ``mitocode.cli`` stays side-effect free.
    from .app import main as _main  # pylint: disable=import-outside-toplevel

    return _main(argv)
