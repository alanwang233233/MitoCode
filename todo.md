# TODO

由于本仓库维护者上下文窗口不足，经常出现上下文断片的情况，故创建此文件.

该文件用于记录当前仓库的 TODO 事项，按优先级排序，方便维护者快速定位问题。

## 高优先级

- [ ] 移除Herobrine
- [x] 完成Workspace.validate()，Workspace.reload()的具体实现
- [x] 重构Agent对外接口 (Agent → AgentSession/AgentRuntime/AgentOrchestrator 三层分离)
- [ ] 添加上下文管理功能
- [ ] 添加索引功能
- [x] 添加MCP功能
- [ ] 添加Agent Skills功能
- [x] 完成TUI (TUI直接调用backend，移除HTTP/SSE层)
- [x] 添加工具管理功能:PR #6
- [ ] 实现多Agent协作编排 (LLM通过工具调用创建/分派子Agent)
- [ ] TUI: 添加Toast通知组件 (错误/状态变化提示)
- [ ] TUI: AgentView中添加错误块渲染 (system.error事件)
- [ ] AgentTemplate 模版功能
- [ ] AgentHook钩子
- [ ] 插件系统
- [ ] 完善异常处理
- [ ] 还要干啥来着？算了想到了再说

## 中优先级

- [ ] 补充migration.py，workspace.py的单元测试
- [x] 补充orchestrator.py，managed_agent.py，event_adapter.py的单元测试
- [ ] 添加日志功能 (loguru已依赖但未全局接入)
- [ ] 完成快照功能 (snapshot.py为TODO存根)
- [x] 将所有raise ...改为raise ... from e，便于调试
- [x] 清理废弃代码 (backend/__main__.py已损坏，experiments/中多数过时)
- [ ] 清理过期依赖 (移除http/SSE相关的包: httpx, httpx-sse, sse-starlette, fastapi, uvicorn)

## 低优先级

- [x] 统一补齐所有模块、类、函数docstring
- [ ] TUI: 快速创建Agent时支持自定义provider配置
- [ ] TUI: 支持Agent重命名 (双击树节点编辑)
- [ ] TUI: 工具调用面板支持diff模式 (finish_tool_call时传入原始内容)
- [x] 将执行black，gitleaks，pip-audit，pytest-cov，pip-licenses，mutmut添加到CI，单独拆分出codeql（因为执行太慢，改成并行运行加快速度）
