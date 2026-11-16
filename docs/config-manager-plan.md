# 全局配置管理器实施计划

状态：已完成（2026-11-13）

基线（本分支实测）：`pytest` 1884 passed；`pylint` 9.88/10；`pyright` 63 errors。

执行纪律：每个任务先写会失败的测试（RED）→ 最小实现（GREEN）→ 再改下一个。
任务之间跑对应测试文件，任务 5 跑全量。

## 任务 1：新增 4 个配置异常

- 文件：`src/mitocode/core/exception/mitocode_exception.py`
- 在 `MitoCodePathConvertError` 之后插入：

```python
class MitoCodeConfigError(MitoCodeBaseError):
    """Base class for configuration-location errors."""

class MitoCodeConfigPathError(MitoCodeConfigError):
    """Raised when a configuration name cannot be turned into a safe path."""

class MitoCodeConfigNotFoundError(MitoCodeConfigError, builtins.FileNotFoundError):
    """Raised when neither the project nor the global copy of a file exists."""

class MitoCodeConfigWriteError(MitoCodeConfigError):
    """Raised when a configuration file cannot be created or opened for writing."""
```

- 同步 `src/mitocode/core/exception/__init__.py` 的导入与 `__all__`。
- 验证：`test/test_config_manager.py::test_missing_config_error_is_a_file_not_found_error`
  先因 `ImportError` 红，再绿。

## 任务 2：实现路径解析器

- 文件：`src/mitocode/core/config/__init__.py`、`src/mitocode/core/config/manager.py`（新增）
- 先在 `test/test_config_manager.py` 写以下用例（全部先红）：

| 用例 | 断言 |
| --- | --- |
| `test_windows_global_dir_is_home_dot_mitocode` | `default_global_dir(platform="win32", home=tmp, env={})` == `tmp/".mitocode"` |
| `test_macos_global_dir_is_under_dot_config` | == `tmp/".config"/".mitocode"` |
| `test_linux_global_dir_honours_absolute_xdg` | `XDG_CONFIG_HOME=tmp/"xdg"` → `tmp/"xdg"/".mitocode"` |
| `test_linux_global_dir_ignores_relative_xdg` | `XDG_CONFIG_HOME="rel/path"` → `tmp/".config"/".mitocode"` |
| `test_linux_global_dir_falls_back_when_xdg_is_unset` | == `tmp/".config"/".mitocode"` |
| `test_manager_exposes_both_directories` | `project_dir` / `global_dir` 正确 |
| `test_project_copy_wins_over_global` | 两层都有 → 项目 |
| `test_global_copy_is_used_when_project_has_none` | 只有全局 → 全局 |
| `test_missing_file_raises_not_found` | 两层都没有 → `MitoCodeConfigNotFoundError` |
| `test_forced_project_scope_does_not_check_existence` | 文件不存在仍返回项目路径 |
| `test_forced_global_scope_returns_global_path` | 项目存在也返回全局路径 |
| `test_exists_reports_either_copy` | 任一层存在 → True |
| `test_open_reads_the_selected_copy` | `with manager.open(...)` 读到项目内容 |
| `test_open_write_creates_the_project_copy` | 写后文件在项目层，父目录被创建 |
| `test_open_write_can_target_global_explicitly` | `scope=GLOBAL` 写全局 |
| `test_absolute_name_is_refused` | `MitoCodeConfigPathError` |
| `test_parent_traversal_is_refused` | `f"..{os.sep}evil.json"` → `MitoCodeConfigPathError` |
| `test_empty_name_is_refused` | `""` → `MitoCodeConfigPathError` |
| `test_project_path_requires_a_workspace` | 无 workspace → `MitoCodeConfigPathError` |
| `test_not_found_is_a_file_not_found_error` | 双继承可用 |
| `test_windows_drive_letter_is_refused` | `skipif(sys.platform != "win32")`；`"C:provider.json"` |
| `test_windows_unc_path_is_refused` | `skipif`；`r"\\\\server\\share\\provider.json"` |
| `test_windows_backslash_traversal_is_refused` | `skipif`；`r"secret\\..\\provider.json"` |

- 实现（完整代码）：

```python
"""Locate the configuration file a caller should open."""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import IO, Mapping, Optional, Union

from mitocode import MITOCODE_CONFIG_PATH
from mitocode.core.exception.mitocode_exception import (
    MitoCodeConfigNotFoundError,
    MitoCodeConfigPathError,
    MitoCodeConfigWriteError,
)

GLOBAL_DIRNAME = ".mitocode"
XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"


class ConfigScope(str, Enum):
    """Which copy of a configuration file a call refers to."""

    AUTO = "auto"
    PROJECT = "project"
    GLOBAL = "global"


def default_global_dir(
    *,
    platform: Optional[str] = None,
    home: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the platform's global configuration directory."""
    platform_name = sys.platform if platform is None else platform
    home_path = Path.home() if home is None else Path(home)
    environment = os.environ if env is None else env
    if platform_name.startswith("win"):
        return home_path / GLOBAL_DIRNAME
    base = home_path / ".config"
    xdg_root = environment.get(XDG_CONFIG_HOME_ENV)
    if xdg_root:
        candidate = Path(xdg_root).expanduser()
        if candidate.is_absolute():
            base = candidate
    return base / GLOBAL_DIRNAME


def _relative_name(name: str) -> Path:
    """Validate *name* as a path relative to a configuration directory."""
    if not isinstance(name, str) or not name.strip():
        raise MitoCodeConfigPathError(
            "A configuration name must be a non-empty relative path."
        )
    candidate = Path(name)
    if not candidate.parts or candidate.is_absolute() or candidate.drive:
        raise MitoCodeConfigPathError(
            f"The configuration name '{name}' must be a relative path."
        )
    if ".." in candidate.parts:
        raise MitoCodeConfigPathError(
            f"The configuration name '{name}' must not contain '..'."
        )
    return candidate


class ConfigManager:
    """Resolve project-first, global-second configuration paths."""

    def __init__(
        self,
        workspace: Optional[Union[str, Path]] = None,
        *,
        global_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._workspace = Path(workspace) if workspace is not None else None
        self._global_dir = (
            Path(global_dir) if global_dir is not None else default_global_dir()
        )

    @classmethod
    def for_workspace(
        cls,
        workspace: Optional[Union[str, Path]],
        *,
        global_dir: Optional[Union[str, Path]] = None,
    ) -> "ConfigManager":
        """Build a manager bound to *workspace* (or to the global copy only)."""
        return cls(workspace, global_dir=global_dir)

    @property
    def global_dir(self) -> Path:
        """The global configuration directory."""
        return self._global_dir

    @property
    def project_dir(self) -> Path:
        """The workspace's configuration directory."""
        if self._workspace is None:
            raise MitoCodeConfigPathError(
                "No workspace is configured, so the project configuration "
                "directory is unavailable."
            )
        return self._workspace / MITOCODE_CONFIG_PATH

    def project_path(self, name: str) -> Path:
        """Return the project path for *name* without checking existence."""
        return self.project_dir / _relative_name(name)

    def global_path(self, name: str) -> Path:
        """Return the global path for *name* without checking existence."""
        return self._global_dir / _relative_name(name)

    def path(self, name: str, *, scope: ConfigScope = ConfigScope.AUTO) -> Path:
        """Return the file to use, raising when AUTO finds no copy."""
        relative = _relative_name(name)
        if scope is ConfigScope.PROJECT:
            return self.project_dir / relative
        if scope is ConfigScope.GLOBAL:
            return self._global_dir / relative
        searched = []
        if self._workspace is not None:
            candidate = self.project_dir / relative
            if candidate.exists():
                return candidate
            searched.append(candidate)
        candidate = self._global_dir / relative
        if candidate.exists():
            return candidate
        searched.append(candidate)
        locations = " or ".join(f"'{item}'" for item in searched)
        raise MitoCodeConfigNotFoundError(
            f"No configuration file named '{relative.as_posix()}' exists at "
            f"{locations}."
        )

    def exists(self, name: str) -> bool:
        """Report whether either copy of *name* exists."""
        relative = _relative_name(name)
        candidates = [self._global_dir / relative]
        if self._workspace is not None:
            candidates.append(self.project_dir / relative)
        return any(candidate.exists() for candidate in candidates)

    def open(
        self,
        name: str,
        mode: str = "r",
        *,
        scope: ConfigScope = ConfigScope.AUTO,
        encoding: str = "utf-8",
    ) -> IO[str]:
        """Open the selected copy; writing defaults to the project copy."""
        relative = _relative_name(name)
        if any(flag in mode for flag in "wax+"):
            target_scope = ConfigScope.PROJECT if scope is ConfigScope.AUTO else scope
            target = self._base(target_scope) / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError as exc:
                raise MitoCodeConfigWriteError(
                    f"Cannot create the configuration directory "
                    f"'{target.parent}': {exc.strerror or 'I/O error'}."
                ) from exc
            try:
                return open(target, mode, encoding=encoding)
            except OSError as exc:
                raise MitoCodeConfigWriteError(
                    f"Cannot write the configuration file '{target}': "
                    f"{exc.strerror or 'I/O error'}."
                ) from exc
        target = self.path(name, scope=scope)
        return open(target, mode, encoding=encoding)

    def _base(self, scope: ConfigScope) -> Path:
        """Return the directory a concrete scope resolves against."""
        if scope is ConfigScope.PROJECT:
            return self.project_dir
        if scope is ConfigScope.GLOBAL:
            return self._global_dir
        raise MitoCodeConfigPathError(f"'{scope}' is not a concrete scope.")
```

- `__init__.py` 只导出 `ConfigManager`、`ConfigScope`、`default_global_dir`。
- 注意：
  - `manager.open` 内的 `open(...)` 是内置函数；若 pylint 报
    `consider-using-with`，加行内 disable 并注明「句柄由调用方的 with 关闭」。
  - 所有比较用 `Path` 对象；测试里不要出现 `str(path) == ".../..."`。
  - 不要 monkeypatch `HOME` 来测平台目录，直接给 `platform=` / `home=` / `env=`。
- 验证：`.venv/bin/python -m pytest test/test_config_manager.py -q` 全绿。

## 任务 3：`cli/config.py` 接入

- 文件：`src/mitocode/cli/config.py`
- 新增导入：`ConfigManager`。
- `SECRET_NAME = SECRET_RELATIVE_PATH.as_posix()`（模块内常量）。
- `secret_path(workspace)` 改为：
  `return ConfigManager.for_workspace(workspace).project_path(SECRET_NAME)`。
- `resolve_provider_config(workspace, *, global_dir=None)`：构造管理器，
  用 `manager.path(SECRET_NAME)` 选中（项目优先、全局兜底），再
  `with open(path, encoding="utf-8")` 读取。`MitoCodeConfigNotFoundError`
  保持现有提示文案（缺失时指出项目路径并给出 `api_key`/`model` 引导）。
- `write_secret_file`：只把 `path` 的来源换成 `manager.project_path(...)`；
  临时文件 + `os.replace` + `0o600` + 拒绝覆盖的逻辑原样保留。
- 新增测试（`test/test_cli_config.py`）：
  - `test_provider_config_falls_back_to_a_global_copy`：项目无文件、注入的
    `global_dir` 有合法配置 → 能解析。
  - `test_project_provider_config_wins_over_the_global_copy`：两份都在 → 用项目。
- 验证：`.venv/bin/python -m pytest test/test_cli_config.py -q`

## 任务 4：文档同步

- `AGENTS.md`：删除「The CLI writes nothing under `~/.mitocode`」这条已失真的
  不变量，改为「配置路径由 `core/config` 解析：项目优先，`provider.json`
  可回退到全局目录」。
- `README.md`：若出现 provider 配置位置的描述，补一句「项目优先、全局兜底」。
- 验证：`grep -rn "writes nothing under" AGENTS.md README.md docs/` 无输出。

## 任务 5：全量验证

```bash
.venv/bin/python -m pytest test/ -q
.venv/bin/python -m black src test --target-version=py310
.venv/bin/python -m pylint --output-format=text src/mitocode | tail -3
.venv/bin/python -m pyright src | tail -1
```

通过标准：

- pytest：1884 + 新增用例全绿；
- pylint ≥ 9.88；pyright ≤ 63 errors；
- `grep -rn 'Path(".mitocode")' src/mitocode/cli` 无输出（`cli/config.py` 不再拼路径）；
- Windows 专属用例在 macOS 上 skip，不产生失败。
