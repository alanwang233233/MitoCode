# Agent Template 实施计划

> 依据：`docs/agent-template.md`（v2）。执行方式：TDD，每批先红后绿；批次之间跑一次全量回归。
> 基线：`.venv/bin/python -m pytest test/ -q --basetemp=.pytest-tmp --ignore=test/test_builtin_paths.py`
> → `1849 passed, 3 skipped`。`test_builtin_paths.py` 会删掉进程 cwd，单独跑，与本次改动无关。

## 批次 A：模板数据层（无 I/O）

### 任务 1：新增模板异常族
- 文件：`src/mitocode/core/exception/mitocode_exception.py`（`MitoCodeSkillReadError` 之后）
- 加：`MitoCodeTemplateError(MitoCodeBaseError)`、`MitoCodeTemplateDefinitionError`、
  `MitoCodeTemplateNotFoundError`、`MitoCodeTemplateReadError`，并进 `__all__`。
- 验证：`python -c "from mitocode.core.exception.mitocode_exception import MitoCodeTemplateDefinitionError"`

### 任务 2：写 RED 测试 `test/test_agent_template_json.py`
- 覆盖：最小合法模板；缺 `name`/`description`；`name` 与目录名不符；非法 `name` 字符；
  `session_autoload` 非数组；块非对象；未知 `kind`；`role` 非法；`arguments_string` 非 JSON；
  `parameter` 与 `arguments_string` 都缺；缺 id 自动分配；唯一未应答的 result 绑定；
  多个未应答 → 拒绝；一个坏块 → 整份拒绝。
- 验证：`pytest test/test_agent_template_json.py -q` 全红。

### 任务 3：实现 `template.py` + `template_json.py`
- `template.py`：`TemplateRoot`、`TemplateIssue`、`TemplateDocument`（name/description/autoload）、
  `AgentTemplate`（content + path/template_path/source）、`render_text(text, workspace)`、
  `system_prompt_of(...)`。
- `template_json.py`：`split/parse` 不复用 skills（JSON 不是 YAML frontmatter），
  `parse_template_json(text, expected_name=None) -> TemplateDocument`；
  `_validate_block`、`_resolve_call_ids`（配对规则）、`MAX_TEMPLATE_FILE_BYTES = 1 MiB`。
- 验证：任务 2 的测试转绿。

## 批次 B：发现层

### 任务 4：写 RED 测试 `test/test_agent_template_manager.py`
- 三根发现 + source 标签 + 排序；workspace > global > builtin 的 shadow 与 issue；
  缺 `template.json`；符号链接目录/文件；根是文件；根不存在；坏模板只损失自己；
  `get()` 未知名返回 None；构造期不读盘。
- 验证：全红。

### 任务 5：实现 `template_manager.py`
- `AgentTemplateManager(roots)`：`roots()`、`async discover()`、`list_templates()`、
  `get(name)`、`issues()`；扫描在 `asyncio.to_thread` 里跑；单条问题只记 issue。
- 验证：任务 4 转绿。

## 批次 C：注入层

### 任务 6：写 RED 测试 `test/test_agent_template_apply.py`
- 块按序注入且在 system prompt 之后；首块 system 时替换默认 system prompt；
  `build_standard_context()` 产出 tool_calls/tool 配对合法序列；
  空 autoload 不变；`{workspace}` 替换且其他花括号原样保留；
  真实 `AgentOrchestrator.create_session(template=...)` 端到端。
- 验证：全红。

### 任务 7：实现 `apply.py` + orchestrator 接入
- `apply_template(session, template, workspace=None)`：把块转成 `_PlainTextMessageBlock` /
  `_ToolCallMessageBlock` / `_ToolResultMessageBlock` 后写入 `session.messages`。
- `AgentOrchestrator.__init__(..., workspace=None)`；`create_session(..., template=None)`，
  注册前调用 applier。
- 验证：任务 6 转绿；`pytest test/test_orchestrator_and_bridge.py -q` 不回归。

## 批次 D：工具层

### 任务 8：写 RED 测试（追加到 `test/test_builtin_agent_tools.py`）
- `template` 透传为模板对象；未知模板 → 错误串列出可用模板；未挂 manager → 明确错误串；
  不传 template → 传给 orchestrator 的是 None。
- 验证：`pytest test/test_builtin_agent_tools.py -q` 新用例红。

### 任务 9：实现 context + `spawn_agent` + register doc
- `ToolContext.template_manager: Any = None`；
- `spawn_agent(..., template: Optional[str] = None)`；
- `register.py` 的 `Agent` agent doc 增补 `template`。
- 验证：任务 8 转绿。

## 批次 E：装配与数据

### 任务 10：内置模板 JSON
- `builtin/general/template.json`（原 `SYSTEM_PROMPT_TEMPLATE` 文案，`{workspace}` 占位）
- `builtin/reviewer/template.json`（review 人格 + 两轮示例对话；刻意不叫 `plan_agent`，
  避免与工作区里同名的本地模板互相 shadow）
- 验证：用解析器直接加载两个文件。

### 任务 11：工作区测试数据
- `.mitocode/agent_templates/deep_research_agent/template.json`（含一个缺 `id` 的 tool_call
  + 一个 id 对不上的 tool_result，专门压配对规则）；
- 校验现有 `.mitocode/agent_templates/plan_agent/template.json` 能被加载。
- 验证：脚本 `discover()` 输出 2 个模板、0 issue。

### 任务 12：bootstrap 装配 + 提示词组合
- `TEMPLATES_DIR_NAME`、`TEMPLATES_PROMPT_SECTION`、`template_roots()`、`build_template_manager()`；
- `build_system_prompt(workspace, skills=(), templates=())` 从 `general` 模板取人格；
- `bootstrap()` 发现模板、挂 context、传 `templates=`、给 orchestrator 传 `workspace=`；
- 删除 `SYSTEM_PROMPT_TEMPLATE` 与 `__all__` 条目。
- 测试：追加到 `test/test_cli_bootstrap.py`（有模板 → 含章节；无模板 → 不含；
  人格来自 general；context 挂上 manager）。
- 验证：`pytest test/test_cli_bootstrap.py -q`。

### 任务 13：打包内置模板
- `pyproject.toml` 加 `[tool.setuptools.package-data]`。
- 验证：`python -m build --wheel` 不做，改为 `python -c` 读 `setuptools` 配置的静态检查。

## 批次 F：质量门

### 任务 14：全量回归 + 静态检查
- `pytest ... --ignore=test/test_builtin_paths.py` → ≥1849 + 新增用例数，0 failed；
- `pytest test/test_builtin_paths.py -q --basetemp=.pytest-tmp` → 与基线一致（1 failed 已知）；
- `black src test --target-version=py310`；`pylint src/mitocode` ≥ 8.0；
  `pyright src` 不超过 52 errors 基线。

### 任务 15：代码评审
- 派一个只读子代理做两段式评审（规格符合度 → 代码质量），Critical 阻断合并。

## 任务 16：收尾
- 汇总 diff，给出合并 / PR / 保留分支三个选项；清理 `.pytest-tmp`。
