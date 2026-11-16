# 全局配置管理器设计

状态：已批准并实现（2026-11-13）
分支：`feat/global-config-manager`

## 1. 职责

**只决定「用哪个配置文件」，不解析内容。**

- 项目级优先；只有调用方允许时，才回退到全局级；
- 读取时自动选中命中的那一份，调用方不需要知道是哪一个；
- 不做分层值合并、不做点号 key、不做 schema 校验、不新增配置项。

## 2. 目录与 Windows 兼容性

| 平台 | 全局目录 |
| --- | --- |
| Windows | `~/.mitocode` |
| macOS | `~/.config/.mitocode` |
| Linux | `$XDG_CONFIG_HOME/.mitocode`，未设置或非绝对路径时 `~/.config/.mitocode` |

项目目录固定为 `<workspace>/.mitocode`。`XDG_CONFIG_HOME` 只接受绝对路径（XDG 规范），相对值按未设置处理。

Windows 兼容性约束（实现与测试都必须满足）：

1. 路径一律用 `pathlib` 计算，不手拼 `/` 或 `\`；返回 `Path` 对象，不返回字符串。
2. 名字校验拒绝盘符路径（`C:provider.json`）、UNC 路径（`\\\\server\\share\\x` 在 Windows 上是绝对路径）、以及任何 `..` 段；这三个判断都基于当前平台的 `Path.parts` / `is_absolute` / `drive`，不做字符串匹配。
3. `mkdir(mode=0o700)` 在 Windows 上 mode 被忽略，不作为错误；不依赖 `chmod`、`fcntl`、符号链接语义。
4. 文本读写显式 `encoding="utf-8"`，不依赖 Windows 默认代码页（GBK）。
5. 测试断言比较 `Path` 对象，不比较分隔符字符串；仅 Windows 成立的用例用 `skipif(sys.platform != "win32")`，POSIX 专属用例同理。
6. CI 现有的三平台矩阵（ubuntu / macos / windows）直接覆盖，不新增平台分支代码。

## 3. 接口

```python
class ConfigScope(str, Enum):
    AUTO = "auto"        # 项目优先，其次全局；都没有 → 抛异常
    PROJECT = "project"  # 只算项目路径，不查存在性
    GLOBAL = "global"    # 只算全局路径，不查存在性

class ConfigManager:
    def __init__(self, workspace: Path | None = None, *, global_dir: Path | None = None)
    @classmethod
    def for_workspace(cls, workspace, *, global_dir=None) -> "ConfigManager"

    project_dir -> Path          # 无工作区时抛 MitoCodeConfigPathError
    global_dir  -> Path

    path(name, *, scope=AUTO) -> Path
    project_path(name) -> Path
    global_path(name) -> Path
    exists(name) -> bool
    open(name, mode="r", *, scope=AUTO, encoding="utf-8") -> IO
```

语义：

- `AUTO` 做存在性判断：项目 → 全局 → 抛 `MitoCodeConfigNotFoundError`。
- `PROJECT` / `GLOBAL` 是纯路径计算，不查存在性；否则写新文件前无法取得目标路径。
- `open(..., "w")` + `AUTO` 落项目层；只有显式 `scope=GLOBAL` 才写全局。
- 写模式自动创建父目录（POSIX 上 `0o700`，Windows 上尽力而为）。
- `name` 必须是相对路径：拒绝空串、绝对路径、盘符、任何 `..` 段。
- 无缓存、非单例：每次调用看磁盘当前状态；测试可注入 `global_dir`。

## 4. 异常

全部直接抛出，无 `except: pass`，无「找不到就返回空配置」。

| 异常 | 基类 | 触发 |
| --- | --- | --- |
| `MitoCodeConfigError` | `MitoCodeBaseError` | 子系统基类 |
| `MitoCodeConfigPathError` | `MitoCodeConfigError` | 名字非法；需要项目路径但无工作区 |
| `MitoCodeConfigNotFoundError` | `MitoCodeConfigError` + `FileNotFoundError` | `AUTO` 下两层都不存在 |
| `MitoCodeConfigWriteError` | `MitoCodeConfigError` | 建目录/打开写入失败，包装底层 `OSError` |

消息只含配置名与路径，不含文件内容。内容解析错误（如 JSON 非法）由调用方的 `json.load` 抛原生异常，管理器不代为决定。

## 5. 改造清单

| 文件 | 改动 |
| --- | --- |
| `core/config/manager.py`（新增） | `ConfigManager`、`ConfigScope`、`default_global_dir` |
| `core/exception/mitocode_exception.py` | 新增 4 个异常并同步 `__init__.py` 导出 |
| `cli/config.py` | `secret_path()` 返回项目路径（写入/存在性检查用）；读取走 `AUTO`（项目优先、全局兜底）；写入保持临时文件 + `os.replace` 的原子逻辑，仅路径来自管理器 |
| `test/test_config_manager.py`（新增） | 路径解析与异常用例 |
| `test/test_cli_config.py` | 新增全局兜底用例 |
| `docs/config-manager.md` / `AGENTS.md` | 同步路径规则，删除「CLI 不写 `~/.mitocode`」这条已失效的描述 |

**不接入**：`core/workspace/workspace.py`、`core/workspace/migration.py`、`cli/bootstrap.py`。它们处理的是工作区身份与项目状态，没有全局语义，接入只增加间接层、不产生行为收益。

## 6. 非目标

- 不做分层值合并、不提供 `get("ui.theme")` 这类点号 key。
- 不做配置内容校验，不定义任何配置键。
- 不提供 `/config` 斜杠命令。
- 不做缓存、文件监听、跨进程读改写锁。
- 不接管 `provider.json` 的写入机制（原子写与 `0600` 仍由 `cli/config.py` 负责）。
- 不改造 workspace / migration / bootstrap 的路径拼接。

## 7. 测试与验收

新增 `test/test_config_manager.py`：

- 平台默认目录：Windows / macOS / Linux；`XDG_CONFIG_HOME` 绝对、相对、未设置。
- 解析：项目优先、全局兜底、两层都没有抛 `MitoCodeConfigNotFoundError`。
- 纯路径：`PROJECT` / `GLOBAL` 在文件不存在时仍返回路径。
- 写入：`AUTO` 落项目并创建父目录；显式 `GLOBAL` 才写全局。
- 非法名字：绝对路径、`..`、空串；Windows 专属：盘符、UNC、反斜杠 `..`。
- 无工作区：`project_dir` / `PROJECT` 抛 `MitoCodeConfigPathError`。
- 兼容：`MitoCodeConfigNotFoundError` 可被 `except FileNotFoundError` 捕获。
- 所有路径断言比较 `Path` 对象，不做分隔符字符串匹配。

`test/test_cli_config.py` 新增：全局目录存在 `secret/provider.json` 时能读取；项目级同时存在时项目优先。

验收：现有 1884 个测试 + 新增用例全绿；`black` / `pylint >= 9.88` / `pyright <= 63 errors` 不退化；`cli/config.py` 不再手工拼接 `.mitocode`（`grep 'Path(".mitocode")' src/mitocode/cli` 无输出）。
