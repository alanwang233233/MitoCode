from os import mkdir
from pathlib import Path

MITOCODE_PATH = ".mitocode"


class Workspace:
    def __init__(self, workspace_path: Path, name:) -> None:
        if not (Path(workspace_path) / MITOCODE_PATH).exists():
            mkdir(Path(workspace_path) / MITOCODE_PATH)
        self.path: Path = Path(workspace_path) / ".mitocode"
        self.name: str = "MitoCodeProject"

    def get_workspace_path(self) -> Path:
        return self.path

    def
