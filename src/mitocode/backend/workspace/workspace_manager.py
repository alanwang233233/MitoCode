import json
import shutil
from io import TextIOWrapper, FileIO, BufferedRandom, BufferedWriter, BufferedReader
from os import mkdir
from pathlib import Path
from typing import Union, BinaryIO, IO

from mitocode.backend.exception.MitoCodeException import MitoCodePathConvertError, MitoCodeWorkspaceInitError, \
    MitoCodeWorkspaceDumpError

MITOCODE_CONFIG_PATH = ".mitocode"
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

class Workspace:
    def __init__(self, workspace_path: Union[str, Path], name: str, description: str = "", dir_map: dict = None, file_map: dict = None) -> None:
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
    def new_workspace(cls, workspace_path: Union[str, Path], name: str, description: str = "", override = True) -> 'Workspace | None':
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
        new_workspace = cls(workspace_path, name, description)
        new_workspace.dump()
        return new_workspace

    @classmethod
    def init_from_path(cls, path: Union[str, Path]) -> 'Workspace | None':
        try:
            path: Path = cls._convert_to_path(path)
            config_path: Path = path / MITOCODE_CONFIG_PATH
            if path.exists() and (path / MITOCODE_CONFIG_PATH).exists() and (config_path / "config.json").is_file() and (config_path / "config.json").exists():
                with open(path / MITOCODE_CONFIG_PATH / "config.json", "r") as f:
                    config: dict = json.load(f)
                    if config:
                        if config.get('name') and config.get('dir_map') and config.get('file_map'):
                            return Workspace(path, config["name"], config["description"])

        except FileNotFoundError as e:
            raise MitoCodeWorkspaceInitError("File not found: " + str(e))
        except Exception as e:
            raise MitoCodeWorkspaceInitError(str(e))

    def dump(self) -> bool:
        if self.config_path.is_dir() and self.path.is_dir():
            try:
                with open(str(self.config_path / "config.json"), "w") as f:
                    json.dump(self.config, f)
                return True
            except Exception as e:
                raise MitoCodeWorkspaceDumpError(str(e))
        return False


    @staticmethod
    def _convert_to_path(str_path: Union[str, Path]) -> Path:
        if str_path:
            if isinstance(str_path, str):
                return Path(str_path)
            return str_path
        raise MitoCodePathConvertError("Fail :" + str(str_path))

    def open(self, file, *args, **kwargs) -> TextIOWrapper | FileIO | BufferedRandom | BufferedWriter | BufferedReader | BinaryIO | IO:
        file = self.get_map(file)
        return open(file, *args, **kwargs)

    def get_map(self, input_str: str) -> Path:
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
            return self.path / full_relative
        else:
            # or it means file map
            file_name = self.config["file_map"].get(input_str, input_str)
            return self.path / file_name