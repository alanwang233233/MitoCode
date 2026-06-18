from os import mkdir
from pathlib import Path
from typing import Union
from mitocode.backend.exception.MitoCodeException import MitoCodePathConvertError, MitoCodeWorkspaceInitError
import json
from jsonschema import validate, ValidationError
import shutil

MITOCODE_CONFIG_PATH = ".mitocode"

class Workspace:
    def __init__(self, workspace_path: Union[str, Path], name: str, description: str = "") -> None:
        workspace_path: Path = self._convert_to_path(workspace_path)
        self.path: Path = workspace_path
        self.config_path: Path = Path(workspace_path) / MITOCODE_CONFIG_PATH
        self.config_file = self.config_path / "config.json"
        self.config: dict = {}
        self.config.name  = name
        self.config.description = description

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
        # dump to file system
        return None

    @classmethod
    def init_from_path(cls, path: Union[str, Path]) -> 'Workspace | None':
        try:
            path: Path = cls._convert_to_path(path)
            config_path: Path = path / MITOCODE_CONFIG_PATH
            if path.exists() and (path / MITOCODE_CONFIG_PATH).exists() and (config_path / "config.json").is_file() and (config_path / "config.json").exists():
                with open(path / MITOCODE_CONFIG_PATH / "config.json", "r") as f:
                    config: dict = json.load(f)
                    if config:
                        if config["name"] and config["description"] and config["dir_map"] and config["file_map"]:
                            return Workspace(path, config["name"], config["description"])

        except FileNotFoundError as e:
            raise MitoCodeWorkspaceInitError("File not found: " + str(e))
        except Exception as e:
            raise MitoCodeWorkspaceInitError(str(e))

    def dump(self) -> bool:
        with open(Path(self.path) / MITOCODE_CONFIG_PATH / "config.json", "rw") as f:
            config = json.load(f)

    @staticmethod
    def _convert_to_path(str_path: Union[str, Path]) -> Path:
        if str_path:
            if isinstance(str_path, str):
                return Path(str_path)
            return str_path
        raise MitoCodePathConvertError("Fail :" + str(str_path))

    def get_workspace_path(self) -> Path:
        return self.path

    def set_workspace_path(self, workspace_path: Union[Path, str]) -> bool:
        if isinstance(workspace_path, str):
            workspace_path: Path = Path(workspace_path)
        if workspace_path.exists():
            if not (workspace_path / MITOCODE_CONFIG_PATH).exists():
                mkdir(Path(workspace_path) / MITOCODE_CONFIG_PATH)
                self.path = workspace_path
                return True
        return False

    def get_workspace_name(self) -> str:
        return self.name

    def set_workspace_name(self, name: str) -> None:
        pass
