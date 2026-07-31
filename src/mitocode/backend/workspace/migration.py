import asyncio
import json
from pathlib import Path
from typing import Union, Callable, Awaitable
from mitocode import APP_VERSION, MITOCODE_CONFIG_PATH
from mitocode.backend.exception.mitocode_exception import (
    MitoCodeWorkspaceInitError,
    MitoCodePathConvertError,
    MitoCodeMigrateError,
)
from .workspace import Workspace

LATEST_VERSION = APP_VERSION


class BasicMigration:
    @staticmethod
    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

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

    @classmethod
    def migrate(cls, path: Union[Path, str]) -> bool | Awaitable[bool]:
        if cls._has_running_loop():
            return cls._migrate_async(path)
        return asyncio.run(cls._migrate_async(path))

    @classmethod
    async def _migrate_async(cls, path: Union[Path, str]) -> bool:
        try:
            # noinspection PyProtectedMember
            path: Path = Workspace._convert_to_path(path)
            config: dict = await cls._read_json(
                path / MITOCODE_CONFIG_PATH / "config.json"
            )
            version: int = int(config["version"])
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            MitoCodePathConvertError,
        ) as e:
            raise MitoCodeWorkspaceInitError(
                f"Failed to load configuration from {path / MITOCODE_CONFIG_PATH}: {str(e)}"
            ) from e
        if version == LATEST_VERSION:
            # no need to migrate
            return True
        elif version > LATEST_VERSION:
            raise MitoCodeMigrateError(
                f"Current version {version} is higher than the latest version {LATEST_VERSION}! Maybe you need to upgrade Mitocode!."
            )
        current_version: int = version
        while current_version < LATEST_VERSION:
            next_version: int = current_version + 1
            migrate_func: Callable = cls._get_migrate_func(next_version)
            try:
                migrate_ret = migrate_func(path)
                if asyncio.iscoroutine(migrate_ret):
                    await migrate_ret
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
                TypeError,
                MitoCodePathConvertError,
            ) as e:
                raise MitoCodeMigrateError(str(e)) from e
            current_version = next_version
        try:
            config_file_path = path / MITOCODE_CONFIG_PATH / "config.json"
            if not await asyncio.to_thread(config_file_path.exists):
                raise MitoCodeMigrateError(
                    "Configuration file 'config.json' was lost or deleted during migration."
                )
            config = await cls._read_json(config_file_path)
            config["version"] = LATEST_VERSION
            await cls._write_json(config_file_path, config)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            raise MitoCodeWorkspaceInitError(str(e)) from e
        # print("success")
        return True

    @classmethod
    def _get_migrate_func(cls, target_version: int) -> Callable:
        if not (target_version - 1 > 0 and target_version > 0):
            raise MitoCodeMigrateError(
                "invalid target version" + " " + str(target_version)
            )
        func_name = f"_migrate_from_{target_version - 1}_to_{target_version}"
        if hasattr(cls, func_name):
            return getattr(cls, func_name)
        else:
            raise MitoCodeMigrateError(
                "No migration function found for version " + str(target_version)
            )

    @classmethod
    def _migrate_from_1_to_2(cls, path) -> bool:
        # it's a test, and it's just a test
        # version 1 and 2 are the same, so no migration is not needed
        # we can simply change version from 1 to 2
        # print(path)
        return True
