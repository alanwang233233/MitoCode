"""Loguru takeover for the MitoCode CLI.

The library's default handler writes to ``sys.stderr``, which would corrupt the
inline REPL as soon as any dependency emits a warning. :func:`configure_logging`
therefore removes every existing sink and installs a single file sink.

The log directory is private to the user (``0o700``) because debug lines may
contain file paths and provider errors; the file itself is appended to and
rotated by size so a long session cannot fill the disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from loguru import logger as loguru_logger

DEFAULT_LOG_SUBDIR = Path(".mitocode") / "logs"
"""Directory below the home directory that holds CLI logs."""

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


def default_log_dir() -> Path:
    """Return the default log directory below the current user's home.

    Returns:
        ``~/.mitocode/logs`` resolved at call time, so tests and embedders can
        redirect it by changing ``HOME``.
    """
    return Path.home() / DEFAULT_LOG_SUBDIR


def configure_logging(
    *,
    debug: bool = False,
    log_dir: Optional[PathLike] = None,
) -> Path:
    """Remove every loguru sink and route logging to a private file.

    Call this once, before anything else logs, so that no library warning can
    reach the terminal while the REPL owns the screen. The file sink always
    runs at :data:`LOG_LEVEL`; *debug* only records that the CLI was started in
    debug mode, which the application later uses for its exit hint.

    Args:
        debug: Whether the CLI was started with ``--debug``.
        log_dir: Directory for the log file. Defaults to
            :func:`default_log_dir`; tests pass a temporary directory.

    Returns:
        Path of the log file, ``<log_dir>/cli.log``.

    Raises:
        OSError: The directory cannot be created, secured or written to.
    """
    directory = default_log_dir() if log_dir is None else Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(DIRECTORY_MODE)

    log_path = directory / LOG_FILENAME

    # Removing every sink is the point of the takeover: a stale stderr handler
    # would print warnings into the REPL.
    loguru_logger.remove()
    loguru_logger.add(
        str(log_path),
        rotation=ROTATION,
        retention=RETENTION,
        level=LOG_LEVEL,
        encoding="utf-8",
        format=LOG_FORMAT,
        backtrace=True,
    )

    if debug:
        loguru_logger.debug("CLI debug mode enabled; file logging is {}.", LOG_LEVEL)

    return log_path


__all__ = [
    "DEFAULT_LOG_SUBDIR",
    "DIRECTORY_MODE",
    "LOG_FILENAME",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "RETENTION",
    "ROTATION",
    "configure_logging",
    "default_log_dir",
]
