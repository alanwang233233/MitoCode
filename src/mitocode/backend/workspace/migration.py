import json
from pathlib import Path
from typing import Union, Callable
from backend.exception.mitocode_exception import MitoCodeWorkspaceInitError, MitoCodePathConvertError, \
    MitocodeMigrateError
from workspace_manager import MITOCODE_CONFIG_PATH, Workspace

class BasicMigration:
    def __init__(self, path: Union[Path, str]):
        self.path : Path = Workspace._convert_to_path(path)
        try:
            with open(self.path / MITOCODE_CONFIG_PATH, "r") as f:
                self.config = json.load(f)
                self.version = self.config["version"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, MitoCodePathConvertError) as e:
            raise MitoCodeWorkspaceInitError(f"Failed to load configuration from {self.path / MITOCODE_CONFIG_PATH}: {str(e)}")

    def migrate(self):
        i = self.version
        while i > 0:
            tmp = i
            # run _migrate_from_tmp_to_tmp-1
            i = i-1
            print("migrate from version: ", tmp, " to version: ", i)

    def _get_migrate_func(self, target_version) -> Callable:
        func_name = f"_migrate_from_v{target_version}_to_v{target_version - 1}"
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise MitocodeMigrateError(f"No migration function found for version {self.version}")

    def _migrate_from_1_to_2(self) -> bool:
        # version 1 and 2 are the same, so no migration is not needed
        return True