"""Locate the configuration file a caller should open.

The manager answers one question: for a given name, is the project copy or the
global copy the one to use?  It never parses a file, never merges values and
never caches the answer -- every call inspects the disk, so two processes and
two workspaces cannot disagree about which file is current.

Paths are handled with :mod:`pathlib` throughout.  Nothing here joins strings
with a separator, calls ``chmod`` or relies on a POSIX-only primitive, so the
same code runs unchanged on Windows, macOS and Linux.
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import IO, Mapping, Optional, Union

from mitocode import MITOCODE_CONFIG_PATH
from mitocode.core.exception.mitocode_exception import (
    MitoCodeConfigNotFoundError,
    MitoCodeConfigPathError,
    MitoCodeConfigWriteError,
)

GLOBAL_DIRNAME = ".mitocode"
"""Directory name of the global configuration root on every platform."""

XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"
"""Environment variable holding the XDG configuration root."""


class ConfigScope(str, Enum):
    """Which copy of a configuration file a call refers to."""

    AUTO = "auto"
    """Project copy first, then the global copy; neither exists is an error."""

    PROJECT = "project"
    """The workspace's own copy, whether or not it exists yet."""

    GLOBAL = "global"
    """The user-level copy, whether or not it exists yet."""


def default_global_dir(
    *,
    platform: Optional[str] = None,
    home: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the platform's global configuration directory.

    Windows uses ``~/.mitocode``; macOS and Linux use ``~/.config/.mitocode``,
    with ``XDG_CONFIG_HOME`` taking precedence on Linux when it holds an
    absolute path (the XDG specification says a relative value must be
    ignored).

    Args:
        platform: Value overriding :data:`sys.platform`; for tests.
        home: Value overriding :meth:`pathlib.Path.home`; for tests.
        env: Mapping overriding :data:`os.environ`; for tests.

    Returns:
        The global configuration directory, not created on disk.
    """
    platform_name = sys.platform if platform is None else platform
    home_path = Path.home() if home is None else Path(home)
    environment = os.environ if env is None else env
    if platform_name.startswith("win"):
        return home_path / GLOBAL_DIRNAME
    base = home_path / ".config"
    xdg_root = environment.get(XDG_CONFIG_HOME_ENV)
    if xdg_root:
        candidate = Path(xdg_root).expanduser()
        if candidate.is_absolute():
            base = candidate
    return base / GLOBAL_DIRNAME


def _relative_name(name: str) -> Path:
    """Validate *name* as a path relative to a configuration directory.

    The check is deliberately structural rather than a string comparison:
    :meth:`pathlib.Path.parts`, :meth:`~pathlib.Path.is_absolute` and
    :attr:`~pathlib.Path.drive` already understand the separators, drive
    letters and UNC prefixes of the current platform.

    Args:
        name: Configuration file name, for example ``"provider.json"`` or
            ``"secret/provider.json"``.

    Returns:
        The validated relative path.

    Raises:
        MitoCodeConfigPathError: The name is empty, absolute, carries a drive
            or contains a ``..`` segment.
    """
    if not isinstance(name, str) or not name.strip():
        raise MitoCodeConfigPathError(
            "A configuration name must be a non-empty relative path."
        )
    candidate = Path(name)
    if not candidate.parts or candidate.is_absolute() or candidate.drive:
        raise MitoCodeConfigPathError(
            f"The configuration name '{name}' must be a relative path."
        )
    if ".." in candidate.parts:
        raise MitoCodeConfigPathError(
            f"The configuration name '{name}' must not contain '..'."
        )
    return candidate


class ConfigManager:
    """Resolve project-first, global-second configuration paths.

    Attributes:
        global_dir: The user-level configuration directory.
    """

    def __init__(
        self,
        workspace: Optional[Union[str, Path]] = None,
        *,
        global_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Prepare a manager for one workspace.

        Args:
            workspace: Workspace root, or ``None`` for a global-only manager.
            global_dir: Override for the platform's global directory; tests use
                it to keep the real home directory out of the run.
        """
        self._workspace = Path(workspace) if workspace is not None else None
        self._global_dir = (
            Path(global_dir) if global_dir is not None else default_global_dir()
        )

    @classmethod
    def for_workspace(
        cls,
        workspace: Optional[Union[str, Path]],
        *,
        global_dir: Optional[Union[str, Path]] = None,
    ) -> "ConfigManager":
        """Build a manager bound to *workspace*, or to the global copy only.

        Args:
            workspace: Workspace root, or ``None``.
            global_dir: Override for the platform's global directory.

        Returns:
            A manager for that workspace.
        """
        return cls(workspace, global_dir=global_dir)

    @property
    def global_dir(self) -> Path:
        """The global configuration directory."""
        return self._global_dir

    @property
    def project_dir(self) -> Path:
        """The workspace's configuration directory.

        Raises:
            MitoCodeConfigPathError: The manager was built without a workspace.
        """
        if self._workspace is None:
            raise MitoCodeConfigPathError(
                "No workspace is configured, so the project configuration "
                "directory is unavailable."
            )
        return self._workspace / MITOCODE_CONFIG_PATH

    def project_path(self, name: str) -> Path:
        """Return the project path for *name* without checking existence.

        Args:
            name: Relative configuration file name.

        Returns:
            ``<workspace>/.mitocode/<name>``.

        Raises:
            MitoCodeConfigPathError: The name is not a safe relative path, or
                no workspace was supplied.
        """
        return self.project_dir / _relative_name(name)

    def global_path(self, name: str) -> Path:
        """Return the global path for *name* without checking existence.

        Args:
            name: Relative configuration file name.

        Returns:
            ``<global_dir>/<name>``.

        Raises:
            MitoCodeConfigPathError: The name is not a safe relative path.
        """
        return self._global_dir / _relative_name(name)

    def path(self, name: str, *, scope: ConfigScope = ConfigScope.AUTO) -> Path:
        """Return the file a caller should use.

        ``AUTO`` prefers the project copy, falls back to the global copy and
        raises when neither exists.  ``PROJECT`` and ``GLOBAL`` are pure path
        computations so a file that is about to be created can be addressed.

        Args:
            name: Relative configuration file name.
            scope: Which copy to resolve.

        Returns:
            The resolved path.

        Raises:
            MitoCodeConfigPathError: The name is not a safe relative path, or
                ``PROJECT`` was requested without a workspace.
            MitoCodeConfigNotFoundError: ``AUTO`` found neither copy.
        """
        relative = _relative_name(name)
        if scope is ConfigScope.PROJECT:
            return self.project_dir / relative
        if scope is ConfigScope.GLOBAL:
            return self._global_dir / relative
        searched = []
        if self._workspace is not None:
            candidate = self.project_dir / relative
            if candidate.exists():
                return candidate
            searched.append(candidate)
        candidate = self._global_dir / relative
        if candidate.exists():
            return candidate
        searched.append(candidate)
        locations = " or ".join(f"'{item}'" for item in searched)
        raise MitoCodeConfigNotFoundError(
            f"No configuration file named '{relative.as_posix()}' exists at "
            f"{locations}."
        )

    def exists(self, name: str) -> bool:
        """Report whether either copy of *name* exists.

        Args:
            name: Relative configuration file name.

        Returns:
            ``True`` when the project or the global copy is present.

        Raises:
            MitoCodeConfigPathError: The name is not a safe relative path.
        """
        relative = _relative_name(name)
        candidates = [self._global_dir / relative]
        if self._workspace is not None:
            candidates.append(self.project_dir / relative)
        return any(candidate.exists() for candidate in candidates)

    def open(
        self,
        name: str,
        mode: str = "r",
        *,
        scope: ConfigScope = ConfigScope.AUTO,
        encoding: str = "utf-8",
    ) -> IO[str]:
        """Open the selected copy, ready for a ``with`` statement.

        A writing mode with ``AUTO`` targets the project copy and creates its
        parent directory; only an explicit ``GLOBAL`` writes the user-level
        copy, so a save never silently changes a file shared by every
        workspace.

        Args:
            name: Relative configuration file name.
            mode: Mode passed to :func:`open`.
            scope: Which copy to open.
            encoding: Text encoding, UTF-8 by default on every platform.

        Returns:
            An open text handle owned by the caller.

        Raises:
            MitoCodeConfigPathError: The name is not a safe relative path, or
                a project path was requested without a workspace.
            MitoCodeConfigNotFoundError: A read found neither copy.
            MitoCodeConfigWriteError: The parent directory or the file could
                not be created.
        """
        relative = _relative_name(name)
        if any(flag in mode for flag in "wax+"):
            target_scope = ConfigScope.PROJECT if scope is ConfigScope.AUTO else scope
            target = self._base(target_scope) / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError as exc:
                raise MitoCodeConfigWriteError(
                    f"Cannot create the configuration directory "
                    f"'{target.parent}': {exc.strerror or 'I/O error'}."
                ) from exc
            try:
                # The handle is handed to the caller, which closes it with
                # `with`; opening it here is the whole point of the method.
                return open(  # pylint: disable=consider-using-with
                    target, mode, encoding=encoding
                )
            except OSError as exc:
                raise MitoCodeConfigWriteError(
                    f"Cannot write the configuration file '{target}': "
                    f"{exc.strerror or 'I/O error'}."
                ) from exc
        target = self.path(name, scope=scope)
        return open(  # pylint: disable=consider-using-with
            target, mode, encoding=encoding
        )

    def _base(self, scope: ConfigScope) -> Path:
        """Return the directory a concrete scope resolves against.

        Args:
            scope: ``PROJECT`` or ``GLOBAL``.

        Returns:
            The corresponding configuration directory.

        Raises:
            MitoCodeConfigPathError: ``AUTO`` reached this helper.
        """
        if scope is ConfigScope.PROJECT:
            return self.project_dir
        if scope is ConfigScope.GLOBAL:
            return self._global_dir
        raise MitoCodeConfigPathError(f"'{scope}' is not a concrete scope.")


__all__ = [
    "ConfigManager",
    "ConfigScope",
    "GLOBAL_DIRNAME",
    "XDG_CONFIG_HOME_ENV",
    "default_global_dir",
]
