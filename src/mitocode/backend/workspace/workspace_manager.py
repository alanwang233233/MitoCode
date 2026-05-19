from os import mkdir
from pathlib import Path
from typing import Union
import shutil
from mitocode.backend.Exception.MitoCodeException import MitoCodePathConvertError

MITOCODE_PATH = ".mitocode"


class Workspace:
    def __init__(self, workspace_path: Union[str , Path], name: str, description: str = "") -> None:
        if isinstance(workspace_path, str):
            workspace_path: Path = Path(workspace_path)
        if not (Path(workspace_path) / MITOCODE_PATH).exists():
            mkdir(Path(workspace_path) / MITOCODE_PATH)
        self.path: Path = Path(workspace_path) / ".mitocode"
        self.name: str = name
        self.description: str = description

    @staticmethod
    def _convert_to_path(str_path : str):
        if str_path:
            if isinstance(str_path, str):
                return Path(str_path)
        raise MitoCodePathConvertError("Failed to convert workspace path to Path object : {}".format(str_path))

    def get_workspace_path(self) -> Path:
        return self.path

    def set_workspace_path(self, workspace_path: Union[Path , str]) -> bool:
        if isinstance(workspace_path, str):
            workspace_path: Path = Path(workspace_path)
        if workspace_path.exists():
            if not (workspace_path / MITOCODE_PATH).exists():
                mkdir(Path(workspace_path) / MITOCODE_PATH)
                self.path = workspace_path
                return True
        return False

    def change_workspace_path(self, workspace_path: Union[Path , str]) -> bool:
        if isinstance(workspace_path, str):
            workspace_path: Path = Path(workspace_path)
            if workspace_path.exists():
                if (not (workspace_path / MITOCODE_PATH).exists()) and self.path.exists():
                    pass


    def get_workspace_name(self) -> str:
        return self.name
