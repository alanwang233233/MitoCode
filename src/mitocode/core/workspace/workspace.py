"""Workspace manager module for MitoCode."""

import asyncio
import json
import shutil
import stat
from pathlib import Path
from typing import Union, Awaitable

from mitocode import MITOCODE_CONFIG_PATH, APP_VERSION
from mitocode.core.exception.mitocode_exception import (
    MitoCodePathConvertError,
    MitoCodeWorkspaceInitError,
    MitoCodeWorkspaceDumpError,
    MitoCodeWorkspacePathError,
    MitoCodeWorkspaceTooOldError,
    MitoCodeAppTooOldError,
)


# noinspection DuplicatedCode
class Workspace:
    """Manages MitoCode workspace configurations and file operations.

    The Workspace class handles initialization, configuration persistence.

    Attributes:
        path (Path): The root path of the workspace.
        config_path (Path): The path to the .mitocode configuration directory.
        config_file (Path): The path to the config.json file.
        config (dict): The workspace configuration dictionary containing version, name,
            description.
    """

    def __init__(
        self,
        workspace_path: Union[str, Path],
        name: str,
        version: int,
        description: str = "",
    ) -> None:
        """Initialize a Workspace instance.

        Args:
            workspace_path: The root path of the workspace, either as a string or Path object.
            name: The name of the workspace.
            description: Optional description of the workspace. Defaults to empty string.
            version: Required version of the workspace configuration as an integer.
        """
        self.path: Path = self._convert_to_path(workspace_path)
        self.config_path: Path = self.path / MITOCODE_CONFIG_PATH
        self.config_file = self.config_path / "config.json"
        self.config: dict = {
            "name": name,
            "description": description,
            "version": version,
        }

    @classmethod
    def new_workspace(
        cls,
        workspace_path: Union[str, Path],
        name: str,
        description: str = "",
        override=True,
    ) -> "Workspace | Awaitable[Workspace]":
        """Create a new workspace at the given path.

        Sync-or-async wrapper around :meth:`_new_workspace_async`: returns the
        coroutine when called inside a running event loop, otherwise runs it
        to completion. See :meth:`_new_workspace_async` for details.
        """
        if cls._has_running_loop():
            return cls._new_workspace_async(
                workspace_path=workspace_path,
                name=name,
                description=description,
                override=override,
            )
        return asyncio.run(
            cls._new_workspace_async(
                workspace_path=workspace_path,
                name=name,
                description=description,
                override=override,
            )
        )

    @classmethod
    async def _new_workspace_async(
        cls,
        workspace_path: Union[str, Path],
        name: str,
        description: str = "",
        override=True,
    ) -> "Workspace":
        """Create a new workspace at the specified path.

        This class method initializes a new MitoCode workspace by creating the configuration
        directory and saving the initial configuration. If a workspace already exists at
        the path, it can either be overridden or re-initialized based on the override parameter.

        Args:
            cls: The class reference.
            workspace_path: The root path where the workspace should be created.
            name: The name of the workspace.
            description: Optional description of the workspace. Defaults to empty string.
            override: If True, removes existing workspace configuration before creating new one.
                If False and workspace exists, initializes from existing path instead.
                Defaults to True.

        Returns:
            A new Workspace instance if successful, or an existing Workspace instance if
            override is False and workspace already exists.

        Raises:
            MitoCodeWorkspaceInitError: If the workspace path does not exist.
        """
        workspace_path_resolved: Path = cls._convert_to_path(workspace_path)
        if not await cls._path_exists(workspace_path_resolved):
            raise MitoCodeWorkspaceInitError(
                "Workspace path does not exist: " + str(workspace_path_resolved)
            )

        config_path = workspace_path_resolved / MITOCODE_CONFIG_PATH
        if await cls._path_exists(config_path):
            if not override:
                return await cls._init_from_path_async(workspace_path_resolved)
            await cls._rmtree(config_path)

        await cls._mkdir(config_path, parents=False, exist_ok=True)
        new_workspace = cls(workspace_path_resolved, name, APP_VERSION, description)
        await new_workspace._dump_async()
        return new_workspace

    @classmethod
    def init_from_path(
        cls, path: Union[str, Path]
    ) -> "Workspace | Awaitable[Workspace]":
        """Load an existing workspace from the given path.

        Sync-or-async wrapper around :meth:`_init_from_path_async`: returns
        the coroutine when called inside a running event loop, otherwise runs
        it to completion. See :meth:`_init_from_path_async` for details.
        """
        if cls._has_running_loop():
            return cls._init_from_path_async(path)
        return asyncio.run(cls._init_from_path_async(path))

    @classmethod
    async def _init_from_path_async(cls, path: Union[str, Path]) -> "Workspace":
        """Initialize a Workspace instance from an existing workspace path.

        This class method loads an existing workspace configuration from the specified path.
        It validates that the required configuration files exist and are properly formatted.

        Args:
            path: The root path of the existing workspace, either as a string or Path object.

        Returns:
            A Workspace instance initialized with the existing configuration.

        Raises:
            MitoCodeWorkspaceInitError: If configuration files are not found or if there
                are errors reading/parsing the configuration.
        """
        try:
            path_resolved: Path = cls._convert_to_path(path)
            config_path: Path = path_resolved / MITOCODE_CONFIG_PATH
            if (
                await cls._path_exists(path_resolved)
                and await cls._path_exists(path_resolved / MITOCODE_CONFIG_PATH)
                and await cls._path_is_file(config_path / "config.json")
            ):
                config: dict = await cls._read_json(config_path / "config.json")
                if not isinstance(config, dict) or not config:
                    raise MitoCodeWorkspaceInitError("config is empty")
                version_raw = config.get("version")
                if version_raw is None:
                    raise MitoCodeWorkspaceInitError(
                        "Configuration file is invalid: 'version' field is missing"
                    )
                try:
                    version = int(version_raw)
                except (ValueError, TypeError) as e:
                    raise MitoCodeWorkspaceInitError(
                        f"Configuration file is invalid: 'version' must be an integer, "
                        f"got {version_raw}"
                    ) from e
                if version > APP_VERSION:
                    raise MitoCodeAppTooOldError(
                        "App need upgrade to open the workspace"
                    )
                if version < APP_VERSION:
                    raise MitoCodeWorkspaceTooOldError(
                        "Workspace need migrate to be opened"
                    )
                if not config.get("name"):
                    raise MitoCodeWorkspaceInitError("config invalid")
                workspace = Workspace(
                    path_resolved,
                    config["name"],
                    version,
                    config.get("description", ""),
                )
                workspace.config["version"] = config.get("version")
                return workspace
            else:
                raise MitoCodeWorkspaceInitError(
                    "Workspace configuration not found at: " + str(path_resolved)
                )

        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise MitoCodeWorkspaceInitError(str(e)) from e

    def dump(self) -> bool | Awaitable[bool]:
        """Save the workspace configuration to disk.

        Sync-or-async wrapper around :meth:`_dump_async`: returns the
        coroutine when called inside a running event loop, otherwise runs it
        to completion. See :meth:`_dump_async` for details.
        """
        if self._has_running_loop():
            return self._dump_async()
        return asyncio.run(self._dump_async())

    async def _dump_async(self) -> bool:
        """Save the workspace configuration to disk.

        Writes the current configuration dictionary to the config.json file
        in the .mitocode directory.

        Returns:
            True if the configuration was successfully saved.

        Raises:
            MitoCodeWorkspaceDumpError: If there is an error writing the
                configuration file or the required directories do not exist.
        """
        if await self._path_is_dir(self.config_path) and await self._path_is_dir(
            self.path
        ):
            try:
                await self._write_json(self.config_path / "config.json", self.config)
                return True
            except (OSError, TypeError, ValueError) as e:
                raise MitoCodeWorkspaceDumpError(str(e)) from e
        raise MitoCodeWorkspaceDumpError("path not found or not a directory")

    @staticmethod
    def _convert_to_path(str_path: Union[str, Path]) -> Path:
        """Convert a string or Path object to a Path.

        This static method ensures that the input is converted to a pathlib.Path object.

        Args:
            str_path: The path to convert, either as a string or Path object.

        Returns:
            A Path object representing the input path.

        Raises:
            MitoCodePathConvertError: If the input is None or empty.
        """
        if str_path:
            if isinstance(str_path, str):
                return Path(str_path)
            return str_path
        raise MitoCodePathConvertError("Fail :" + str(str_path))

    def resolve_within(self, target: Union[str, Path]) -> Path:
        """Resolve *target* against the workspace root and verify it stays within bounds.

        This is the workspace's public containment barrier against path
        traversal: any ``..`` components that climb out of the root, and any
        symlink pointing outside it, are rejected here.

        Resolution rules, in order:

        1. An **absolute** target is taken as-is.  If it lies inside the root
           it is returned; if it lies outside, it raises.  It is deliberately
           *not* re-rooted against the workspace — a tool such as ``Glob``
           hands the model absolute paths, and ``Read`` must honour exactly
           what it was given rather than silently resolving a different file.
        2. A **relative** target is joined to the workspace root.

        Both forms are normalised with :meth:`pathlib.Path.resolve` before the
        containment check, so ``..`` segments and symlinks are collapsed first
        and cannot smuggle a path past the check.

        Args:
            target: A relative or absolute path to resolve against ``self.path``.

        Returns:
            The fully resolved absolute path if it lies within the workspace.

        Raises:
            MitoCodeWorkspacePathError: If the resolved path escapes the
                workspace root.  The class also inherits ``ValueError``.

        Note:
            "Escapes the root" is the only thing checked here.  A symlink loop
            behaves differently by version — ``RuntimeError`` on 3.10-3.12,
            returned unresolved on 3.13+ — so a caller that needs a yes/no
            answer should use :meth:`is_within`, which handles both.
        """
        candidate = Path(target)
        if candidate.is_absolute():
            full_path = candidate.resolve()
        else:
            full_path = (self.path / candidate).resolve()
        workspace_root = self.path.resolve()
        if not full_path.is_relative_to(workspace_root):
            raise MitoCodeWorkspacePathError(
                f"Path traversal detected: '{target}' resolves to '{full_path}', "
                f"which is outside the workspace root '{workspace_root}'"
            )
        return full_path

    def is_within(self, target: Union[str, Path]) -> bool:
        """Report whether *target* resolves inside the workspace root, never raising.

        A predicate rather than an assertion, so callers that only need the
        answer do not have to catch.  Anything that cannot be resolved into a
        usable path — a traversal, a symlink leaving the root, a symlink loop,
        an embedded NUL byte, a non-path argument — simply counts as
        "not within".

        True is reported only when *target* is both inside the root **and
        fully resolved**.  The second half is not redundant: a caller that
        trusts this verdict goes on to ``open()`` the path, and a partially
        resolved path cannot be opened at all, so "inside the root" alone
        would hand it something useless and label it trusted.

        Symlink loops are the case that forces this, and the case where the
        behaviour of the underlying call is version-dependent.
        ``Path.resolve()`` on a loop changed in Python 3.13:

        * On 3.10-3.12 it raises ``RuntimeError`` ("Symlink loop from ..."),
          which this method catches and reports as not-within.
        * On 3.13+ it no longer raises.  It returns the path with the looping
          component left unresolved, and because that string still sits under
          the root the containment check alone would answer True and pass on a
          path that only ever yields ELOOP.

        Catching ``RuntimeError`` therefore fixes 3.10-3.12 and leaves CI's
        3.14 broken.  Both are covered without sniffing the interpreter version
        by :meth:`_is_fully_resolved`: a correctly resolved path has no symlink
        left anywhere in it, and a loop is precisely a resolution that stopped
        early and left one behind.  That check is one ``lstat`` per component, a
        syscall whose behaviour has not changed across any supported version
        — unlike ``resolve()``, which is why the check exists rather than the
        catch alone.  Do not "simplify" this back to a bare containment test.

        Args:
            target: A relative or absolute path to check against ``self.path``.

        Returns:
            True if *target* is inside the workspace root and fully resolvable;
            False otherwise, including for every input that cannot even be
            turned into a path.
        """
        try:
            resolved = self.resolve_within(target)
        except (ValueError, TypeError, OSError, RuntimeError):
            return False
        return self._is_fully_resolved(resolved)

    @staticmethod
    def _is_fully_resolved(path: Path) -> bool:
        """Report whether *path* is the complete resolution of itself.

        Walks *path* and its ancestors with :meth:`pathlib.Path.lstat` looking
        for any component that is still a symbolic link.  ``resolve()`` follows
        every symlink it can, so a finished resolution has none left; a
        resolution that stopped early — because it hit a loop — has exactly
        one.

        Deliberately uses ``lstat()`` rather than the ``is_symlink()``
        convenience method: the latter swallows ``OSError`` and answers False,
        which would collapse "this component is a symlink", "this component
        does not exist" and "this component cannot be traversed" into one
        indistinguishable answer, and the first of those three is the one being
        hunted.  Distinguishing them is the whole point, so this catches
        ``FileNotFoundError`` and ``OSError`` separately.

        Args:
            path: An absolute path, normally the result of
                :meth:`resolve_within`, so already normalised against the root.

        Returns:
            True if no component of *path* is an unresolved symlink, False if
            one is, or if a component cannot be inspected (a loop in a middle
            component raises ``ELOOP`` here, and ``EACCES`` means the path
            cannot be traversed at all — neither is usable).
        """
        for component in (path, *path.parents):
            try:
                if stat.S_ISLNK(component.lstat().st_mode):
                    return False
            except FileNotFoundError:
                # Nothing exists here, so nothing below can exist either and
                # none of it can be a symlink.  This is the Write-a-new-file
                # case, and stopping here keeps it to a single lstat.
                return True
            except OSError:
                # ELOOP on a looping middle component, or EACCES on a directory
                # we may not traverse.  Either way the path cannot be opened.
                return False
        return True

    def _validate_within_workspace(self, target: Path) -> Path:
        """Delegate to :meth:`resolve_within`; retained as the private alias.

        Args:
            target: A relative or absolute path to resolve against ``self.path``.

        Returns:
            The fully resolved absolute path if it lies within the workspace.

        Raises:
            MitoCodeWorkspacePathError: If the resolved path escapes the
                workspace root.
        """
        return self.resolve_within(target)

    def validate(self) -> bool | Awaitable[bool]:
        """Validate the workspace configuration on disk.

        Sync-or-async wrapper around :meth:`_validate_async`: returns the
        coroutine when called inside a running event loop, otherwise runs it
        to completion. See :meth:`_validate_async` for details.
        """
        if self._has_running_loop():
            return self._validate_async()
        return asyncio.run(self._validate_async())

    async def _validate_async(self) -> bool:
        """Validate the workspace configuration.

        This method checks if the workspace is still valid by verifying:
        1. The workspace path exists
        2. The .mitocode configuration directory exists
        3. The config.json file exists and is readable
        4. The configuration contains required fields

        Returns:
            True if the workspace is valid.

        Raises:
            MitoCodeWorkspaceInitError: If any validation step fails.
        """
        try:
            if not await self._path_exists(self.path):
                raise MitoCodeWorkspaceInitError(
                    f"Workspace path does not exist: {self.path}"
                )
            if not await self._path_exists(self.config_path):
                raise MitoCodeWorkspaceInitError(
                    f"Configuration directory does not exist: {self.config_path}"
                )
            if not await self._path_is_file(self.config_file):
                raise MitoCodeWorkspaceInitError(
                    f"Configuration file does not exist: {self.config_file}"
                )
            config = await self._read_json(self.config_file)
            if not isinstance(config, dict) or not config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is empty or invalid"
                )
            if "name" not in config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is missing required 'name' field"
                )
            if "version" not in config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is missing required 'version' field"
                )
            return True
        except (OSError, json.JSONDecodeError) as e:
            raise MitoCodeWorkspaceInitError(str(e)) from e

    def reload(self) -> bool | Awaitable[bool]:
        """Reload the workspace configuration from disk.

        Sync-or-async wrapper around :meth:`_reload_async`: returns the
        coroutine when called inside a running event loop, otherwise runs it
        to completion. See :meth:`_reload_async` for details.
        """
        if self._has_running_loop():
            return self._reload_async()
        return asyncio.run(self._reload_async())

    async def _reload_async(self) -> bool:
        """Reload the workspace configuration from disk.

        This method re-reads the config.json file and updates the in-memory
        configuration dictionary. It also validates the configuration before
        applying changes.

        Returns:
            True if the configuration was successfully reloaded.

        Raises:
            MitoCodeWorkspaceInitError: If the configuration cannot be read or is invalid.
        """
        try:
            if not await self._path_is_file(self.config_file):
                raise MitoCodeWorkspaceInitError(
                    f"Configuration file does not exist: {self.config_file}"
                )
            config = await self._read_json(self.config_file)
            if not isinstance(config, dict) or not config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is empty or invalid"
                )
            if "name" not in config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is missing required 'name' field"
                )
            if "version" not in config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is missing required 'version' field"
                )
            self.config = config
            return True
        except (OSError, json.JSONDecodeError) as e:
            raise MitoCodeWorkspaceInitError(str(e)) from e

    def is_outdated(self) -> bool | Awaitable[bool]:
        """Check whether the in-memory config differs from disk.

        Sync-or-async wrapper around :meth:`_is_outdated_async`: returns the
        coroutine when called inside a running event loop, otherwise runs it
        to completion. See :meth:`_is_outdated_async` for details.
        """
        if self._has_running_loop():
            return self._is_outdated_async()
        return asyncio.run(self._is_outdated_async())

    async def _is_outdated_async(self) -> bool:
        """Check if the in-memory configuration is outdated compared to disk.

        This method compares the current in-memory configuration with the
        configuration stored on disk to determine if they differ.

        Returns:
            True if the in-memory configuration differs from the disk configuration,
            False otherwise.

        Raises:
            MitoCodeWorkspaceInitError: If the configuration cannot be read from disk.
        """
        try:
            if not await self._path_is_file(self.config_file):
                raise MitoCodeWorkspaceInitError(
                    f"Configuration file does not exist: {self.config_file}"
                )
            disk_config = await self._read_json(self.config_file)
            if not isinstance(disk_config, dict) or not disk_config:
                raise MitoCodeWorkspaceInitError(
                    "Configuration file is empty or invalid"
                )
            return self.config != disk_config
        except (OSError, json.JSONDecodeError) as e:
            raise MitoCodeWorkspaceInitError(str(e)) from e

    @staticmethod
    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    @staticmethod
    async def _path_exists(path: Path) -> bool:
        return await asyncio.to_thread(path.exists)

    @staticmethod
    async def _path_is_dir(path: Path) -> bool:
        return await asyncio.to_thread(path.is_dir)

    @staticmethod
    async def _path_is_file(path: Path) -> bool:
        return await asyncio.to_thread(path.is_file)

    @staticmethod
    async def _mkdir(
        path: Path, *, parents: bool = False, exist_ok: bool = False
    ) -> None:
        await asyncio.to_thread(path.mkdir, parents=parents, exist_ok=exist_ok)

    @staticmethod
    async def _rmtree(path: Path) -> None:
        await asyncio.to_thread(shutil.rmtree, path)

    @staticmethod
    async def _read_json(path: Path) -> dict:
        def _read() -> dict:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

    @staticmethod
    async def _write_json(path: Path, content: dict) -> None:
        def _write() -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f)

        await asyncio.to_thread(_write)
