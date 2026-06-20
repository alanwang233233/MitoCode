import json
import shutil
from io import TextIOWrapper, FileIO, BufferedRandom, BufferedWriter, BufferedReader
from os import mkdir
from pathlib import Path
from typing import Union, BinaryIO, IO

from mitocode.backend.exception.MitoCodeException import MitoCodePathConvertError, MitoCodeWorkspaceInitError, \
    MitoCodeWorkspaceDumpError

"""Workspace manager module for MitoCode.

This module provides the Workspace class for managing MitoCode workspace configurations,
including directory mappings, file mappings, and configuration persistence.
"""

MITOCODE_CONFIG_PATH = ".mitocode"
"""Default path for MitoCode configuration directory within a workspace."""

DEFAULT_MAP = {
    "dir_map": {
        "agent": "/agent",
        "context": "/context"
    },
    "file_map": {
        "mcpServers": "/mcpServers.json",
        "config": "/config.json",
        "skills": "/skills.json",
        "allow_commands": "/allow_command.json"
    }
}
"""Default directory and file mappings for workspace structure.

The dir_map defines logical directory names to actual paths, while file_map
defines logical file names to actual file paths within the workspace.
"""

class Workspace:
    """Manages MitoCode workspace configurations and file operations.

    The Workspace class handles initialization, configuration persistence,
    and path resolution for MitoCode workspaces using directory and file mappings.

    Attributes:
        path (Path): The root path of the workspace.
        config_path (Path): The path to the .mitocode configuration directory.
        config_file (Path): The path to the config.json file.
        config (dict): The workspace configuration dictionary containing name,
            description, dir_map, and file_map.
    """

    def __init__(self, workspace_path: Union[str, Path], name: str, description: str = "", dir_map: dict = None, file_map: dict = None) -> None:
        """Initialize a Workspace instance.

        Args:
            workspace_path: The root path of the workspace, either as a string or Path object.
            name: The name of the workspace.
            description: Optional description of the workspace. Defaults to empty string.
            dir_map: Optional dictionary mapping logical directory names to actual paths.
                If None, uses DEFAULT_MAP["dir_map"].
            file_map: Optional dictionary mapping logical file names to actual paths.
                If None, uses DEFAULT_MAP["file_map"].
        """
        workspace_path: Path = self._convert_to_path(workspace_path)
        self.path: Path = workspace_path
        self.config_path: Path = Path(workspace_path) / MITOCODE_CONFIG_PATH
        self.config_file = self.config_path / "config.json"
        self.config: dict = {
            'name': name,
            'description': description,
            'dir_map': dir_map or DEFAULT_MAP.get("dir_map"),
            'file_map': file_map or DEFAULT_MAP.get("file_map")
        }

    @classmethod
    def new_workspace(cls, workspace_path: Union[str, Path], name: str, description: str = "", dir_map: dict = None, file_map: dict = None, override = True) -> 'Workspace | None':
        """Create a new workspace at the specified path.

        This class method initializes a new MitoCode workspace by creating the configuration
        directory and saving the initial configuration. If a workspace already exists at
        the path, it can either be overridden or re-initialized based on the override parameter.

        Args:
            cls: The class reference.
            workspace_path: The root path where the workspace should be created.
            name: The name of the workspace.
            description: Optional description of the workspace. Defaults to empty string.
            dir_map: Optional custom directory mapping. Defaults to DEFAULT_MAP["dir_map"].
            file_map: Optional custom file mapping. Defaults to DEFAULT_MAP["file_map"].
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

        if (workspace_path / MITOCODE_CONFIG_PATH).exists() :
            if not override:
                return cls.init_from_path(workspace_path)
            else:
                # log
                shutil.rmtree(workspace_path / MITOCODE_CONFIG_PATH)

        mkdir(Path(workspace_path) / MITOCODE_CONFIG_PATH)
        new_workspace = cls(workspace_path, name, description, dir_map, file_map)
        new_workspace.dump()
        return new_workspace

    @classmethod
    def init_from_path(cls, path: Union[str, Path]) -> 'Workspace | None':
        """Initialize a Workspace instance from an existing workspace path.

        This class method loads an existing workspace configuration from the specified path.
        It validates that the required configuration files exist and are properly formatted.

        Args:
            path: The root path of the existing workspace, either as a string or Path object.

        Returns:
            A Workspace instance initialized with the existing configuration, or None if
            initialization fails.

        Raises:
            MitoCodeWorkspaceInitError: If configuration files are not found or if there
                are errors reading/parsing the configuration.
        """
        try:
            path: Path = cls._convert_to_path(path)
            config_path: Path = path / MITOCODE_CONFIG_PATH
            if path.exists() and (path / MITOCODE_CONFIG_PATH).exists() and (config_path / "config.json").is_file() and (config_path / "config.json").exists():
                with open(path / MITOCODE_CONFIG_PATH / "config.json", "r") as f:
                    config: dict = json.load(f)
                    if config:
                        if config.get('name') and config.get('dir_map') and config.get('file_map'):
                            return Workspace(path, config["name"], config["description"], config["dir_map"], config["file_map"])

        except FileNotFoundError as e:
            raise MitoCodeWorkspaceInitError("File not found: " + str(e))
        except Exception as e:
            raise MitoCodeWorkspaceInitError(str(e))

    def dump(self) -> bool:
        """Save the workspace configuration to disk.

        Writes the current configuration dictionary to the config.json file
        in the .mitocode directory.

        Returns:
            True if the configuration was successfully saved, False if the
            required directories do not exist.

        Raises:
            MitoCodeWorkspaceDumpError: If there is an error writing the
                configuration file.
        """
        if self.config_path.is_dir() and self.path.is_dir():
            try:
                with open(self.config_path / "config.json", "w") as f:
                    json.dump(self.config, f)
                return True
            except Exception as e:
                raise MitoCodeWorkspaceDumpError(str(e))
        return False


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

    def open(self, file, *args, **kwargs) -> TextIOWrapper | FileIO | BufferedRandom | BufferedWriter | BufferedReader | BinaryIO | IO:
        """Open a file within the workspace using mapped paths.

        Resolves the file path using the workspace's directory and file mappings,
        then opens the file with the specified arguments.

        Args:
            file: The logical file identifier. If it starts with '/', it is treated
                as a directory mapping (e.g., '/agent/file.txt'). Otherwise, it is
                treated as a file mapping key (e.g., 'config' for config.json).
            *args: Additional positional arguments to pass to the built-in open().
            **kwargs: Additional keyword arguments to pass to the built-in open().

        Returns:
            A file object opened at the resolved path.
        """
        file = self.get_map(file)
        return open(file, *args, **kwargs)

    def _validate_within_workspace(self, target: Path) -> Path:
        """Resolve the given path against the workspace root and verify it stays within bounds.

        This is a security barrier against path traversal attacks — any ``..`` components,
        absolute symlinks pointing outside the workspace, or malicious map entries are
        rejected here.

        Args:
            target: A relative path to resolve against ``self.path``.

        Returns:
            The fully resolved absolute path if it lies within the workspace.

        Raises:
            ValueError: If the resolved path escapes the workspace root.
        """
        full_path = (self.path / target).resolve()
        workspace_root = self.path.resolve()
        if not full_path.is_relative_to(workspace_root):
            raise ValueError(
                f"Path traversal detected: '{target}' resolves to '{full_path}', "
                f"which is outside the workspace root '{workspace_root}'"
            )
        return full_path

    def get_map(self, input_str: str) -> Path:
        """Resolve a logical file or directory identifier to an absolute path.

        This method interprets the input string according to the workspace's mapping rules:
        - If the input starts with '/', it is treated as a directory mapping reference.
          The first part after '/' is looked up in dir_map, and remaining parts are
          appended as subdirectories/files.
        - Otherwise, it is treated as a file mapping key and looked up in file_map.

        All resolved paths are validated to prevent path traversal outside the workspace.

        Args:
            input_str: The logical identifier to resolve. Examples:
                - '/agent/tasks/task1.txt' -> resolves 'agent' in dir_map, then appends 'tasks/task1.txt'
                - 'config' -> looks up 'config' in file_map to get the actual filename

        Returns:
            A Path object representing the resolved absolute path within the workspace.

        Raises:
            ValueError: If the resolved path escapes the workspace root.
        """
        # begin with / means dir_map
        if input_str.startswith("/"):
            path_parts = input_str.strip("/").split("/")
            if not path_parts or path_parts == [""]:
                return self.path
            dir_key = path_parts[0]
            rest_parts = path_parts[1:]
            mapped_dir = self.config["dir_map"].get(dir_key, dir_key)
            mapped_dir_path = Path(mapped_dir.lstrip("/"))
            full_relative = mapped_dir_path.joinpath(*rest_parts)
            return self._validate_within_workspace(full_relative)
        else:
            # or it means file map
            file_name = self.config["file_map"].get(input_str, input_str)
            return self._validate_within_workspace(Path(file_name))