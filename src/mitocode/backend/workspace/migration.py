import json
from pathlib import Path
from typing import Union, Callable
from backend.exception.mitocode_exception import MitoCodeWorkspaceInitError, MitoCodePathConvertError, \
    MitocodeMigrateError
from workspace import MITOCODE_CONFIG_PATH, Workspace

LATEST_VERSION = 2

class BasicMigration:
    @classmethod
    def migrate(cls, path: Union[Path, str]):
        path: Path = Workspace._convert_to_path(path)
        try:
            with open(path / MITOCODE_CONFIG_PATH / "config.json", "r") as f:
                config = json.load(f)
                version = config["version"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, MitoCodePathConvertError) as e:
            raise MitoCodeWorkspaceInitError(
                f"Failed to load configuration from {path / MITOCODE_CONFIG_PATH}: {str(e)}")
        i = LATEST_VERSION
        while i > 1:
            tmp = i
            migrate_func = cls._get_migrate_func(i)
            i = i-1
            try:
                migrate_func()
                print("migrate from version: ", tmp, " to version: ", i)
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, MitoCodePathConvertError) as e:
                raise MitocodeMigrateError(str(e))


    @classmethod
    def _get_migrate_func(cls, target_version) -> Callable:
        if not (target_version -1 > 0 and target_version > 0):
            raise MitocodeMigrateError("invalid target version" + str(target_version))
        func_name = f"_migrate_from_{target_version-1}_to_{target_version}"
        if hasattr(cls, func_name):
            return getattr(cls, func_name)
        else:
            raise MitocodeMigrateError("No migration function found for version " + str(target_version))

    @classmethod
    def _migrate_from_1_to_2(cls) -> bool:
        # it's a test, and it's just a test
        # version 1 and 2 are the same, so no migration is not needed
        # we can simply change version from 1 to 2
        print("pending migration...")
        return True
