# TODO

由于本仓库维护者上下文窗口不足，经常出现上下文断片的情况，故创建此文件.

该文件用于记录当前仓库的 TODO 事项，按优先级排序，方便维护者快速定位问题。

## 高优先级

- [ ] 移除Herobrine
- [ ] 完成Workspace.validate()，Workspace.reload()的具体实现
- [ ] 重构Agent对外接口
- [ ] 添加上下文管理功能
- [ ] 添加工具管理功能
- [ ] 添加索引功能
- [ ] 添加MCP功能
- [ ] 添加Agent Skills功能
- [ ] 完成TUI

## 中优先级

- [ ] 补充migration.py，workspace.py的单元测试
- [ ] 添加日志功能
- [ ] 完成快照功能
- [ ] 将所有raise ...改为raise ... from e，便于调试

## 低优先级

- [x] 全局执行black + isort，修复行过长、导入顺序
- [ ] 将执行black + isort添加到CI
- [ ] 统一补齐所有模块、类、函数docstring