"""
Workspace manager module for MitoCode.
"""
import json
import shutil
from pathlib import Path
from typing import Union

from mitocode import MITOCODE_CONFIG_PATH, APP_VERSION
from mitocode.backend.exception.mitocode_exception import MitoCodePathConvertError, MitoCodeWorkspaceInitError, \
    MitoCodeWorkspaceDumpError, MitoCodeWorkspaceTooOldError, MitoCodeAppTooOldError


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

    def __init__(self, workspace_path: Union[str, Path], name: str, version: int, description: str = "") -> None:
        """Initialize a Workspace instance.

        Args:
            workspace_path: The root path of the workspace, either as a string or Path object.
            name: The name of the workspace.
            description: Optional description of the workspace. Defaults to empty string.
            version: Optional version of the workspace configuration. Defaults to "0.0.1".
        """
        workspace_path: Path = self._convert_to_path(workspace_path)
        self.path: Path = workspace_path
        self.config_path: Path = Path(workspace_path) / MITOCODE_CONFIG_PATH
        self.config_file = self.config_path / "config.json"
        self.config: dict = {
            'name': name,
            'description': description,
            'version': version
        }

    @classmethod
    def new_workspace(cls, workspace_path: Union[str, Path], name: str, description: str = "",
                      override=True) -> 'Workspace':
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
        workspace_path: Path = cls._convert_to_path(workspace_path)
        if not workspace_path.exists():
            raise MitoCodeWorkspaceInitError("Workspace path does not exist: " + str(workspace_path))

        if (workspace_path / MITOCODE_CONFIG_PATH).exists():
            if not override:
                return cls.init_from_path(workspace_path)
            else:
                # log
                shutil.rmtree(workspace_path / MITOCODE_CONFIG_PATH)

        (workspace_path / MITOCODE_CONFIG_PATH).mkdir(parents=False, exist_ok=True)
        new_workspace = cls(workspace_path, name, APP_VERSION, description)
        new_workspace.dump()
        return new_workspace

    @classmethod
    def init_from_path(cls, path: Union[str, Path]) -> 'Workspace':
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
            path: Path = cls._convert_to_path(path)
            config_path: Path = path / MITOCODE_CONFIG_PATH
            if path.exists() and (path / MITOCODE_CONFIG_PATH).exists() and (config_path / "config.json").is_file():
                with open(path / MITOCODE_CONFIG_PATH / "config.json", "r") as f:
                    config: dict = json.load(f)
                    if config:
                        version_raw = config.get("version")
                        if version_raw is None:
                            raise MitoCodeWorkspaceInitError(
                                "Configuration file is invalid: 'version' field is missing")
                        try:
                            version = int(version_raw)
                        except (ValueError, TypeError):
                            raise MitoCodeWorkspaceInitError(
                                f"Configuration file is invalid: 'version' must be an integer, got {version_raw}")
                        if version > APP_VERSION:
                            raise MitoCodeAppTooOldError("App need upgrade to open the workspace")
                        elif version < APP_VERSION:
                            raise MitoCodeWorkspaceTooOldError("Workspace need migrate to be opened")
                        if config.get('name'):
                            workspace = Workspace(path, config["name"], config.get("version"),
                                                  config.get("description", ""))
                            workspace.config['version'] = config.get('version')
                            return workspace
                        else:
                            raise MitoCodeWorkspaceInitError("config invalid")
                    else:
                        raise MitoCodeWorkspaceInitError("config is empty")
            else:
                raise MitoCodeWorkspaceInitError("Workspace configuration not found at: " + str(path))

        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise MitoCodeWorkspaceInitError(str(e))

    def dump(self) -> bool:
        """Save the workspace configuration to disk.

        Writes the current configuration dictionary to the config.json file
        in the .mitocode directory.

        Returns:
            True if the configuration was successfully saved.

        Raises:
            MitoCodeWorkspaceDumpError: If there is an error writing the
                configuration file or the required directories do not exist.
        """
        if self.config_path.is_dir() and self.path.is_dir():
            try:
                with open(self.config_path / "config.json", "w") as f:
                    json.dump(self.config, f)
                return True
            except (OSError, TypeError, ValueError) as e:
                raise MitoCodeWorkspaceDumpError(str(e))
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

    def _validate_within_workspace(self, target: Path) -> Path:
        """Resolve the given path against the workspace root and verify it stays within bounds.

        This is a security barrier against path traversal attacks — any ``..`` components,
        absolute symlinks pointing outside the workspace are rejected here.

        Args:
            target: A relative path to resolve against ``self.path``.

        Returns:
            The fully resolved absolute path if it lies within the workspace.

        Raises:
            ValueError: If the resolved path escapes the workspace root.
        """
        target = Path(str(target).lstrip("/"))
        full_path = (self.path / target).resolve()
        workspace_root = self.path.resolve()
        if not full_path.is_relative_to(workspace_root):
            raise ValueError(
                f"Path traversal detected: '{target}' resolves to '{full_path}', "
                f"which is outside the workspace root '{workspace_root}'"
            )
        return full_path

    def validate(self):
        # TODO
        return True

    def reload(self):
        # TODO
        return True
