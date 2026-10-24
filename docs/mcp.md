# MitoCode MCP 客户端

MitoCode 作为 **MCP（Model Context Protocol）客户端**接入外部 MCP server，把它们提供的工具
注册进既有的 `ToolManager`，供 agent 调用。

支持两类 server：

- **远程** —— 通过 Streamable HTTP 访问的独立服务
- **本地** —— 由 MitoCode 启动为子进程、通过 stdio 通信的脚本

> **当前状态：核心类已完成，尚未接线。** `__main__.py` 和 TUI 里没有任何生产调用点，MCP
> 功能对用户不可达。见文末「下一步」。

---

## 目录

- [架构](#架构)
- [快速上手](#快速上手)
- [配置格式](#配置格式)
- [API](#api)
- [不变量](#不变量)
- [调用时序约束](#调用时序约束)
- [安全：凭据不进入消息](#安全凭据不进入消息)
- [已知限制](#已知限制)
- [测试](#测试)
- [下一步](#下一步)

---

## 架构

```
              config dict / List[McpServerConfig]
                          │
              ┌───────────▼────────────┐
              │  McpServerConfig.parse │  校验 url / command 互斥、类型、超时
              │        build_transport │  → StreamableHttpTransport | StdioTransport
              └───────────┬────────────┘
                          │  每个 server 一个
              ┌───────────▼────────────┐
              │       McpClient        │  connect(): 建连 + 缓存工具目录
              │                        │  call_tool(): 调用 + 渲染成字符串
              │                        │  disconnect(): 幂等拆解
              └───────────┬────────────┘
                          │  server_name + tools
              ┌───────────▼────────────┐
              │      McpManager        │  并发连接、失败隔离、精确注销、状态
              └───────────┬────────────┘
                          │  register_mcp_client() / unregister()
              ┌───────────▼────────────┐
              │      ToolManager       │  工具名 mcp__<server>__<tool>
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │     AgentSession       │  已有的 tool_manager= 路径，无需改动
              └────────────────────────┘
```

**依赖方向是单向的**：`tool_manager` → `mcp` 包 → `fastmcp`。

`mcp` 包**绝不**在运行时 import `tool_manager`；`tool_manager` 只在 `if TYPE_CHECKING:`
块里引用 `McpClient`，所以它能在没装 fastmcp 的环境里 import。这条约束由测试守护
（子解释器屏蔽 `fastmcp` 和独立 `mcp` SDK 后仍能导入）。

### 为什么每个 server 一个 Client，而不是 `ClientGroup`

FastMCP 的 `ClientGroup.__aenter__` 会连接全部 client，任何一个失败整组就进入不了。对
harness 来说不可接受 —— 一个坏掉的 stdio server 不该让所有工具消失。自管连接换来三样东西：

- per-server 失败隔离
- per-server 重连与刷新
- TUI 能标注工具来源

代价是发现循环要自己写，约 50 行。

---

## 快速上手

```python
import asyncio

from mitocode.core.mcp import McpManager
from mitocode.core.tool.tool_manager import ToolManager

tool_manager = ToolManager()
tool_manager.register(read_file, write_file)  # 内置工具照常注册

manager = McpManager(
    {
        "mcpServers": {
            "weather": {"url": "https://weather.example.com/mcp"},
            "files": {"command": "npx", "args": ["-y", "@mcp/files"]},
        }
    },
    tool_manager=tool_manager,
)

asyncio.run(manager.start())  # 连不上的 server 只丢自己的工具，不抛异常

print(manager.status())
# {'weather': McpServerStatus(state='connected', error='', tool_count=12),
#  'files':   McpServerStatus(state='failed', error='MCPError: ...', tool_count=0)}

# agent 从这里开始就能看到 mcp__weather__get_forecast 之类的工具
session = AgentSession(provider=provider, tool_manager=tool_manager)
```

`start()` **不会因为任何一个 server 失败而抛异常** —— 这正是这个模块存在的理由。

---

## 配置格式

采用 Claude Desktop 的通用格式（FastMCP 也认）：

```jsonc
{
  "mcpServers": {
    "weather": {
      "url": "https://weather.example.com/mcp",
      "headers": {"Authorization": "Bearer <token>"},   // 静态令牌
      "timeout": 60,                                     // 单次调用超时（秒）
      "connect_timeout": 30                              // 建连超时（秒）
    },
    "files": {
      "command": "npx",
      "args": ["-y", "@mcp/files"],
      "env": {"TOKEN": "..."},
      "cwd": "/srv/project",
      "enabled": false                                   // 跳过连接，status 报 "disabled"
    }
  }
}
```

**`url` 与 `command` 必须且只能给一个。**

| 字段 | 类型 | 默认 | 适用 |
|---|---|---|---|
| `url` | `str`（`http://` / `https://`） | — | HTTP |
| `command` | `str`（非空） | — | stdio |
| `args` | `list[str]` | `[]` | stdio |
| `env` | `dict[str, str]` | `{}` | stdio |
| `cwd` | `str` | `""` | stdio |
| `headers` | `dict[str, str]` | `{}` | HTTP |
| `timeout` | 正数 | `60.0` | 单次 `call_tool` |
| `connect_timeout` | 正数 | `30.0` | 建连 + 工具发现 |
| `enabled` | `bool` | `True` | 两者 |

`connect_timeout` 默认 30 秒是刻意的：`npx -y` 冷启动可能要几十秒，一个短的全局超时会
把健康 server 误判为失败。它只从 server 配置来，`McpManager` 上没有第二份，避免「哪层
优先」的歧义。

校验失败抛 `MitoCodeMcpConfigError`，消息里带 server 名和**位置**，不带值（见
[安全](#安全凭据不进入消息)）。

---

## API

### `McpServerConfig` — `mcp_config.py`

冻结 dataclass，`dataclasses.replace()` 可用于派生。

```python
cfg = McpServerConfig.parse("weather", {"url": "https://x/mcp"})
cfg = McpServerConfig.parse_config({"mcpServers": {...}})   # -> list[McpServerConfig]
transport = cfg.build_transport()                            # fastmcp transport 对象
```

`parse_config` 只接受带 `mcpServers` 包装的字典；重复的 server 名会被拒绝（报位置，不报值）。

### `McpClient` — `mcp_client.py`

一个 server 一条连接。

```python
c = McpClient(cfg)
await c.connect()            # -> bool，失败只记录不抛
c.state                      # McpConnectionState: DISCONNECTED | CONNECTED | FAILED
c.last_error                 # 最近的失败原因（已脱敏），无错则 ""
c.tools                      # connect() 缓存的工具对象列表
await c.call_tool("get_forecast", {"city": "London"})   # -> str
await c.disconnect()         # 幂等
```

`call_tool` 接受的是 **server 侧的工具名**，不是 `mcp__server__tool`。返回字符串是因为
`AgentRuntime` 会对工具返回值调 `str()`。失败抛 `MitoCodeMcpToolCallError`。

### `McpManager` — `mcp_manager.py`

```python
m = McpManager(config, *, tool_manager, client_factory=McpClient)
await m.start()                       # 并发连接；只注册成功的 server
await m.stop()                        # 注销工具 + 断开全部
await m.refresh("weather")            # 刷新一个；refresh() 刷新全部
m.status()                            # -> {name: McpServerStatus(state, error, tool_count)}
```

`config` 接受 `{"mcpServers": {...}}` 或 `list[McpServerConfig]`。`client_factory` 是测试接缝。
`tool_manager` 是鸭子类型的对象 —— 任何有 `register_mcp_client` 和 `unregister` 的都行。

`refresh()` 在 `start()` 没跑过时也能工作（它会连上它点名的那个）。

### `ToolManager` 的新增 —— `tool_manager.py`

本特性唯一修改的既有模块，全部向后兼容。

```python
MCP_TOOL_PREFIX = "mcp__"
MCP_TOOL_SEPARATOR = "__"
MAX_TOOL_NAME_LENGTH = 64

tool_manager.build_mcp_tool_name("weather", "get_forecast")  # -> "mcp__weather__get_forecast"
tool_manager.register_mcp_client(client)   # -> list[ToolSpec]，全有或全无
tool_manager.unregister("mcp__weather__get_forecast")       # -> bool

# register() / tool() 新增可选关键字
tool_manager.register(func, name=..., parameters_schema=...)  # MCP schema 原样透传
```

`parameters_schema` 是必需的：MCP 工具的 schema 来自 server 的 `input_schema`，无法从
Python 注解推导。省略时行为与改动前完全一致。

名字被占用时**跳过并 warning，绝不覆盖**。`register_mcp_client` 是**全有或全无**的：先把
整份目录读完、归一化、拼好名，再动 `self._tools` —— 否则一个中途失败会让先注册的工具留在
注册表里、无人认领、任何 `stop()` 都清不掉。

同一个 server 重复注册会返回 `[]`（每个名字都撞上了），要重新注册得先 `unregister`。

### 异常

```
MitoCodeBaseError
└── MitoCodeMcpError
    ├── MitoCodeMcpConfigError      配置非法
    ├── MitoCodeMcpConnectionError  预留，当前无人抛出（connect 走记录而非抛出）
    └── MitoCodeMcpToolCallError    工具调用失败
```

---

## 不变量

这六条是本特性存在的理由，改动时必须保住：

1. **`start()` 不会因为某个 server 失败而抛异常。** 一个连不上的 server 只损失它自己的工具。
2. **`status()` 永远和注册表一致。** 已注册、可调用的工具不会被报成不存在；活着的 server
   不会被报成 `disabled`。
3. **没有工具能活过 `stop()`。** 也没有哪个 server 的失败能拖垮另一个的工具。
4. **凭据值不进入异常消息。** `status().error` 和所有异常都会进 TUI 错误面板和日志。
5. **导入方向单向。** `mcp` 包不在运行时 import `tool_manager`。
6. **取消是被传播的，不是被吞掉的** —— `start` / `stop` / `refresh` 都如此。

### 关于取消的细节

`McpClient.connect()` 和 `disconnect()` 都用 `asyncio.shield()` 包住拆解，并且**在 await
之前**就把状态/句柄落定。原因：FastMCP 的 `StdioTransport` 在一个**游离任务**里拉起子进程，
主协程只 `await _ready_event.wait()`。只取消外层 await 的话，子进程会活成孤儿。所以
shield 是必需的，而状态提前赋值同样必需 —— 否则 `connect()` 会因为「已经连接」而提前返回
`True`，把一个底下没有 transport 的客户端认领过去。

shield 的保证有边界：**只要事件循环比取消活得久**。进程循环本身被拆掉时，任何在途的拆解
都会被取消。

---

## 调用时序约束

> **`AgentSession` 在构造时快照工具注册表。**
> 所以 `McpManager.start()` 必须在创建任何 agent session **之前**完成，否则已存在的 session
> 看不到 MCP 工具。

这写在 `McpManager` 的 docstring 里。接线时（例如 `__main__.py`）：

```python
tool_manager = ToolManager()
_register_builtin_tools(tool_manager)

manager = McpManager(config, tool_manager=tool_manager)
asyncio.run(manager.start())          # ← 必须在这里完成

orchestrator = AgentOrchestrator(tool_manager=tool_manager, default_provider=provider)
asyncio.run(orchestrator.create_agent(name="Main"))
app = VibeFrontend(orchestrator=orchestrator)
app.run()
```

`stop()` 挂在 TUI 关闭路径上时，因为取消是被传播的，`finally` 里应该
`await asyncio.shield(manager.stop())`。

---

## 安全：凭据不进入消息

这条不变量被评审抓到过三次泄漏，都已修复：

- **配置校验期** —— `headers` / `env` / `url` / `args` 的**值**从不进入异常消息，只报位置
  和键名。（键名是用户可控的，所以放在键里的凭据仍会被回显 —— 这是明确记录的契约。）
- **连接期** —— `mcp_client._scrub_endpoint()` 剥掉 URL 的 userinfo、query、fragment，
  应用于 `connect()`、`_force_close()` 和 `mcp_manager._describe()`。
- **调用期** —— `_redact_arguments()` 剥掉 pydantic 错误里的 `input_value=` 载荷，
  `_scrub_endpoint()` 同时处理工具错误里引用的 endpoint。server 自己的文本保留 —— 那是
  诊断的关键，且是调用方选择信任哪个 server 的结果。

`_scrub_endpoint` 额外覆盖了**无 scheme** 的 `/mcp?token=…` 形态，因为那正是真实泄漏的形态
（FastMCP 4.1 的错误文本就是 `Internal error while proxying /mcp?…`）。代价是
`word?key=value` 这种散文里的问号也可能被当成 endpoint 处理掉。

---

## 已知限制

有意保留的取舍，都有记录：

**端点脱敏的边界**
- URL **path** 段的 token 保留（只认 query / userinfo / fragment）
- 跨行拆开的 token 会漏出尾巴
- `word?key=value` 形式的散文可能被误处理

**配置对象的表示**
- `frozen=True` 只挡字段**赋值**。`cfg.args` / `cfg.env` / `cfg.headers` 仍可变，且
  `build_transport()` 按引用交给 fastmcp，之后改动会影响下一次连接。
- dataclass 自动生成的 `repr` 打印所有字段（含 `url`），所以 `log(cfg)` 仍可能回显 URL 里的
  密码。禁泄漏保证只覆盖**异常消息**。
- `float("inf")` 会被当作合法超时（`inf > 0` 成立），然后永久挂住调用方。
- `"http://"` 无主机名会被接受 —— scheme 检查是 `startswith` 前缀测试，不是 URL 解析。

**工具名长度**
- `mcp__<server>__<tool>` 原样进 `tool_schemas()`，必须满足 OpenAI 的
  `^[a-zA-Z0-9_-]{1,64}$`。前缀和分隔符本身合规，但**长度**是 server 作者和工具作者花的 ——
  长 server 名 + 长工具名可能超 64 字符。**不会截断**（截断就是另一个工具了），只发 warning。

**子进程 stderr**
- `build_transport()` 没给 `StdioTransport` 传 `log_file`，所以子进程的 stderr 默认直通父
  进程终端。argv 里的 token、或某个 server 把自己 env 打进 stderr，会落到终端/日志 ——
  这条路径完全不经过 `_scrub_endpoint`。

---

## 测试

全离线，无需 API key 或网络。

```bash
.venv/bin/python -m pytest test/test_mcp_config.py -q      # 40 用例，纯校验
.venv/bin/python -m pytest test/test_mcp_client.py -q      # 73 用例，含 in-memory FastMCP server
.venv/bin/python -m pytest test/test_mcp_manager.py -q     # 70 用例，含真实 stdio 子进程
.venv/bin/python -m pytest test/ -q                        # 全量
```

| 模块 | 语句 | 分支 |
|---|---|---|
| `mcp_config.py` | 100% | 100% |
| `mcp_client.py` | 100% | 100% |
| `mcp_manager.py` | 100% | 100% |
| `tool_manager.py` | 97% | 98% |

两条真实 stdio 子进程测试（`test_mcp_manager.py`）在 `tmp_path` 写一个最小 FastMCP server
脚本，用 `sys.executable` 启动，是「本地 server 这条主路径真的通」的唯一证明。

测试大量使用变异测试（mutation testing）验证自身有效性：对源码的 scratch 副本逐个注入
缺陷，确认有测试会失败。`test_mcp_client.py` 的 redaction 边界、`disconnect` 的 shield、
`stop()` 的两阶段拆解都是这样钉住的。

---

## 下一步

**接线**（`todo.md` 里未勾选的那行）。按顺序要做：

1. 在 `__main__.py` 里构造 `config` 并 `await manager.start()`，**在创建任何
   `AgentSession` 之前**（见[调用时序约束](#调用时序约束)）
2. 决定配置来源 —— workspace 的 `.mitocode/config.json`（团队共享）还是全局配置（个人
   令牌），或两者合并
3. 在 TUI 里展示 `manager.status()`，并把 `stop()` 挂到关闭路径
4. （可选）子进程 stderr 改到受控 sink —— 见[已知限制](#已知限制)

`AgentSession` 目前没有刷新工具注册表的路径，所以 v1 是**启动时一次性发现**。server 增删工具
需要重启，或者手动调 `refresh()` 并新建 session。