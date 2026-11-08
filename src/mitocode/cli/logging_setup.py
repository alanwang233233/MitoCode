"""Loguru takeover for the MitoCode CLI.

The library's default handler writes to ``sys.stderr``, which would corrupt the
inline REPL as soon as any dependency emits a warning. :func:`configure_logging`
therefore removes every existing sink and installs a single file sink.

The log directory is private to the user (``0o700``) because debug lines may
contain file paths and provider errors; the file itself is appended to and
rotated by size so a long session cannot fill the disk. Tracebacks are written
with ``diagnose=False`` so loguru never renders the local variable values of
traceback frames into the file: those frames can hold raw credentials such as
the provider API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from loguru import logger as loguru_logger

LOG_FILENAME = "cli.log"
"""Name of the log file inside the log directory."""

LOG_LEVEL = "DEBUG"
"""Minimum level written to the file; the file is always the verbose one."""

ROTATION = "10 MB"
"""Rotate the file once it grows past ten megabytes."""

RETENTION = 5
"""Keep at most this many rotated files."""

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - {message}"
)
"""Plain-text format for the file sink; markup would only add noise."""

DIRECTORY_MODE = 0o700
"""Permission bits applied to the log directory."""

PathLike = Union[str, Path]


def configure_logging(
    *,
    log_dir: PathLike,
    debug: bool = False,
) -> Path:
    """Remove every loguru sink and route logging to a private workspace file.

    Call this once, before anything else logs, so that no library warning can
    reach the terminal while the REPL owns the screen. The file sink always
    runs at :data:`LOG_LEVEL`; *debug* only records that the CLI was started in
    debug mode, which the application later uses for its exit hint.

    Args:
        log_dir: Directory for the log file, normally
            ``<workspace>/.mitocode/logs``. Required: the CLI never falls back
            to a location outside the workspace.
        debug: Whether the CLI was started with ``--debug``.

    Returns:
        Path of the log file, ``<log_dir>/cli.log``.

    Raises:
        OSError: The directory cannot be created, secured or written to.
    """
    # Remove sinks before touching the filesystem: a run whose workspace is
    # not writable must fail without also writing anything to stderr, and the
    # caller renders its own error panel.
    loguru_logger.remove()
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(DIRECTORY_MODE)

    log_path = directory / LOG_FILENAME

    loguru_logger.add(
        str(log_path),
        rotation=ROTATION,
        retention=RETENTION,
        level=LOG_LEVEL,
        encoding="utf-8",
        format=LOG_FORMAT,
        backtrace=True,
        # Tracebacks stay in the log, but loguru's variable-value diagnostics
        # must never render frame locals: they can hold the raw API key.
        diagnose=False,
    )

    if debug:
        loguru_logger.debug("CLI debug mode enabled; file logging is {}.", LOG_LEVEL)

    return log_path


__all__ = [
    "DIRECTORY_MODE",
    "LOG_FILENAME",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "RETENTION",
    "ROTATION",
    "configure_logging",
]
