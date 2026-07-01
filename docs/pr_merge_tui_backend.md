## 概述

将 TUI 前端与后端直接集成，移除 HTTP/SSE 通信层，引入 `AgentOrchestrator` 作为多 Agent 生命周期的统一管理中枢。

## 动机

TUI 原本通过 SSE（`/v1/sync`、`/v1/message`）连接后端服务，但该服务从未实现——仅有一个静态的 `dev_mock.py`。导致 TUI 只能展示模拟数据，无法发送消息、控制 Agent 生命周期、反映真实 Agent 状态。

## 架构变更

```
重构前:                      重构后:
TUI ──SSE──> [不存在的后端]    TUI ──直接调用──> AgentOrchestrator
                                                  ├── ManagedAgent[0] ── AgentRuntime
                                                  ├── ManagedAgent[1] ── AgentRuntime
                                                  └── ManagedAgent[N] ── AgentRuntime
```

数据流：`AgentRuntime.run() → EventBridge → asyncio.Queue → TUI 事件消费 worker`

## 新增模块

| 模块 | 用途 |
|------|------|
| `agent/agent_node.py` | Agent 树数据模型，含状态字段 |
| `agent/managed_agent.py` | AgentSession 包装器，生命周期状态与父子关系 |
| `agent/orchestrator.py` | 多 Agent 注册、消息分发、级联取消、`asyncio.Queue` 事件路由 |
| `bridge/event_adapter.py` | AgentRuntime 事件 → TUI 兼容格式，通过 ToolManager 富化标题元信息 |
| `chat_provider/scenario_provider.py` | 多轮工具调用模拟（bash、read_file、write_file 含冒泡排序代码），用于 TUI 开发测试 |

## TUI 重写

- 移除 `httpx`/`httpx-sse`，事件通过 orchestrator 事件队列由 worker 消费
- 新增消息输入框、快速创建 Agent、键盘导航
- Agent 树显示运行时状态图标（空闲/运行中/已完成/错误）
- 消息以独立 Panel 控件渲染，背景色区分类型
- 工具调用参数在流式过程中增量解析 JSON，解析成功后立即以格式化内容替换原始 JSON

## 其他

- `ToolSpec` 新增 `display_action`/`display_param` 字段
- 删除损坏的 `backend/__main__.py`
- 补全各 `__init__.py` 的导出
- 新增 8 个测试，总计 133 个全部通过
- `python -m mitocode` 启动，`--speed` 控制流式速度

## 已知问题

- `httpx`、`httpx-sse`、`fastapi`、`sse-starlette`、`uvicorn` 仍留在依赖中但核心代码不再引用；依赖它们的 `dev_mock.py` 暂保留
- `Workspace.validate()`/`Workspace.reload()`、`snapshot.py` 仍为 TODO 存根（既有问题）
- `ChatProviderOpenAI` 已接入但 CLI 尚未提供 `--openai` 入口
