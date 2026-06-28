import json
from pathlib import Path
from typing import Union, Callable
from mitocode import APP_VERSION, MITOCODE_CONFIG_PATH
from mitocode.backend.exception.mitocode_exception import MitoCodeWorkspaceInitError, MitoCodePathConvertError, \
    MitoCodeMigrateError
from .workspace import Workspace

LATEST_VERSION = APP_VERSION


class BasicMigration:
    @classmethod
    def migrate(cls, path: Union[Path, str]) -> bool:
        try:
            # noinspection PyProtectedMember
            path: Path = Workspace._convert_to_path(path)
            with open(path / MITOCODE_CONFIG_PATH / "config.json", "r") as f:
                config: dict = json.load(f)
                version: int = int(config["version"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, MitoCodePathConvertError) as e:
            raise MitoCodeWorkspaceInitError(
                f"Failed to load configuration from {path / MITOCODE_CONFIG_PATH}: {str(e)}")
        if version == LATEST_VERSION:
            # no need to migrate
            return True
        elif version > LATEST_VERSION:
            raise MitoCodeMigrateError(
                f"Current version {version} is higher than the latest version {LATEST_VERSION}! Maybe you need to upgrade Mitocode!.")
        current_version : int= version
        while current_version < LATEST_VERSION:
            next_version : int = current_version + 1
            migrate_func : Callable = cls._get_migrate_func(next_version)
            try:
                migrate_func(path)
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, MitoCodePathConvertError) as e:
                raise MitoCodeMigrateError(str(e))
            current_version = next_version
        try:
            config_file_path = path / MITOCODE_CONFIG_PATH / "config.json"
            if not config_file_path.exists():
                raise MitoCodeMigrateError("Configuration file 'config.json' was lost or deleted during migration.")
            with open(config_file_path, "r") as f:
                config = json.load(f)
            with open(config_file_path, "w") as f:
                config["version"] = LATEST_VERSION
                json.dump(config, f)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            raise MitoCodeWorkspaceInitError(str(e))
        # print("success")
        return True

    @classmethod
    def _get_migrate_func(cls, target_version: int) -> Callable:
        if not (target_version - 1 > 0 and target_version > 0):
            raise MitoCodeMigrateError("invalid target version" + " " + str(target_version))
        func_name = f"_migrate_from_{target_version - 1}_to_{target_version}"
        if hasattr(cls, func_name):
            return getattr(cls, func_name)
        else:
            raise MitoCodeMigrateError("No migration function found for version " + str(target_version))

    @classmethod
    def _migrate_from_1_to_2(cls, path) -> bool:
        # it's a test, and it's just a test
        # version 1 and 2 are the same, so no migration is not needed
        # we can simply change version from 1 to 2
        # print(path)
        return True
