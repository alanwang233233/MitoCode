# Agent Template 设计文档

> 状态：已确认（v2，按评审意见修订）
> 修订要点：① 增加全局 `.mitocode` 根（复用 `ConfigManager`）；② `general` 成为默认模板，
> 代码中的硬编码提示词移入模板文件；③ 模板在 `AgentSession` 之外生成会话，`agent_session.py`
> 不加注入逻辑。

## 1. 背景与目标

`Agent` 工具现在只能派生子代理并给它一句 prompt：子代理从零开始，没有角色设定，
也没有任何预置上下文。`.mitocode/agent_templates/` 下已经有人手写了
`template.json`（一个 `{name, description, session_autoload: [...]}` 的消息块快照），
但代码里没有任何东西读它。

本特性让「创建 Agent 时带模板」成为一等能力：

1. `Agent` 工具新增可选参数 `template`，按名字选择模板；
2. 创建会话时模板里的上下文被原样注入，子代理一启动就带着这段上下文（含
   `tool_call` / `tool_result` 块），而不是只拿到一句系统提示；
3. 数据来源三个根：工作区 `.mitocode/agent_templates/`、全局
   `<config>/agent_templates/`、包内置 `core/agent/template/builtin/`；
4. 内置 `general` 是默认模板——`build_system_prompt` 的人格文案从模板读，
   Python 里不再硬编码整段提示词。

## 2. 非目标

- 不做模板的可视化编辑器、不做运行期热重载（重新 `discover()` 即可）；
- 不做 `/templates` 斜杠命令，也不新增 `ListTemplate` 工具——模板目录通过系统提示词
  的模板章节暴露给模型（见 §6）；
- 除 `{workspace}` 外不做变量插值：只做一次字面量替换，其余花括号原样保留
  （见 §5.3）；
- 不改 `session_autoload` 的数据形态（沿用现有样例）；
- 不改 `agent_session.py` 的构造逻辑：注入由模板包里的 applier 在会话构造完成后完成。

## 3. 模板格式

一个模板 = 一个目录 + 一个 `template.json`：

```
<root>/<模板名>/template.json
```

```json
{
  "name": "plan-agent",
  "description": "Plans a multi-step change before touching code.",
  "session_autoload": [
    {"kind": "text", "role": "system", "message": "You are a planning agent."},
    {"kind": "text", "role": "user", "message": "Plan the refactor."},
    {"kind": "tool_call", "name": "Read", "arguments_string": "{\"file_path\": \"a.py\"}"},
    {"kind": "tool_result", "tool_call_id": "call_1", "content": "print(1)"}
  ]
}
```

### 字段

| 字段 | 必填 | 规则 |
| --- | --- | --- |
| `name` | 是 | `^[a-z0-9]+([_-][a-z0-9]+)*$`，≤ 64 字符，且必须等于目录名。下划线是必须放开的：本特性落地前已有 `plan_agent`、`deep_research_agent` 两个目录 |
| `description` | 是 | 折叠为单行后 1..1024 字符 |
| `session_autoload` | 否 | 消息块数组；缺省/空数组表示「只有角色说明，没有预置上下文」 |

### 消息块

| `kind` | 必填字段 | 可选字段 |
| --- | --- | --- |
| `text` | `role` ∈ {`system`, `user`, `assistant`}、`message` (str) | `reasoning_content` (str) |
| `tool_call` | `name` (str)、`arguments_string`（合法 JSON 对象文本） | `id` (str)、`parameter`（对象）。两者同时存在时以 `arguments_string` 为准，`parameter` 由它反解——现有样例两个字段都写 |
| `tool_result` | `content` (str) | `tool_call_id` (str)、`success` (bool，默认 `true`) |

### 配对规则（容错点）

手写模板常漏 `id`。解析时按**位置**配对，规则只有一条：

- `tool_call` 没写 `id` → 解析器分配 `call_<块序号>`；
- `tool_result` 的 `tool_call_id` 与所有未应答的 `tool_call` 都对不上、且**恰好只有一个**
  未应答的 `tool_call` → 绑定到它（文件里写错/漏写的 id 被忽略）；
- 其余情况（多个未应答、或根本没有未应答）→ 整个模板视为非法。

这条规则让现有 `.mitocode/agent_templates/plan_agent/template.json`（首个 `tool_call`
缺 `id`）能正常加载，同时不会把真正错位的模板猜成一个错误的历史。

### 另外三条「整份非法」（provider 会拒绝的历史不放过）

- `tool_call` 的 id 重复（含手写 id 撞上自动分配的 `call_<位置>`）：两条 call 共用一个
  身份，请求要么被拒、要么结果错配；
- `tool_call` 与其 `tool_result` 之间夹了 `text` 块：provider 要求结果紧跟调用；
- 文件末尾留着没有被应答的 `tool_call`。

这三条同属「宁可在加载时说清楚，也不要留到第一次请求远端报一个看不懂的错」。

模板是**整体校验**的：任何一个块非法，整个模板被拒绝并记成 issue，不做部分加载——
半截上下文比没有上下文更糟，而且它只会在第一次调 provider 时炸掉，用户完全看不懂。

## 4. 发现与优先级

`AgentTemplateManager` 与 `SkillsManager` 同构（有序根 + 整份重扫 + issue 记录，
永不因单条坏数据抛出）。根由 wiring 层给出，manager 不硬编码任何路径：

```
根顺序（先到先得，同名后者被 shadow 并记录 issue）：
  1. ConfigManager.project_path("agent_templates")   label="workspace"
  2. ConfigManager.global_path("agent_templates")    label="global"
  3. core/agent/template/builtin                     label="builtin"
```

- 根不存在 → 贡献 0 个模板、0 个 issue；
- 目录缺 `template.json` → 一条 issue（该目录被跳过）；
- `template.json` 是符号链接 → 拒绝（沿用 skills 的防任意文件读取立场）；
- 解析/校验失败 → 一条 issue。

## 5. 注入语义

### 5.1 注入点

`core/agent/template/apply.py` 提供 `apply_template(session, template, workspace=None)`，
在 `AgentOrchestrator.create_session` **构造完会话之后、注册之前**调用。`agent_session.py`
不新增构造参数，只被当作数据容器使用。块由 applier **直接构造**成三个内部块类型后写入
`session.messages`，而不是走 `append_user_prompt` / `append_assistant_delta`：后者是给
流式增量用的，带副作用（重置取消事件、清空推理缓冲、合并相邻 assistant 文本），模板重放
不该触发其中任何一个。

| 块 | 落地方式 |
| --- | --- |
| `text` | `_PlainTextMessageBlock(role, message)`，`reasoning_content` 原样带上 |
| `tool_call` | `_ToolCallMessageBlock(name, id)` + 赋 `arguments_string` + `finish()` |
| `tool_result` | `_ToolResultMessageBlock(tool_call_id, content, success)` |

### 5.2 system 块的优先级

> 若 `session_autoload` 的**第一个块**是 `text` + `role="system"`，它**替换**会话默认的
> system prompt；否则默认 system prompt 保留，所有块追加在其后。

理由：模板自带 system 块时，作者表达的是「完整人格」，再叠一层默认提示词会得到两段
互相打架的角色说明。现有样例正是这种形态。

注意由此产生的等价关系：`general` 模板贡献默认人格，所以「不传 template」和
「传 template=general」得到的会话基本一致，只差技能/模板目录两节。这个差异是有意保留
的：显式点名一个模板，就等于用模板自带的人格替换默认人格。`Agent` 工具文档只说明
「省略 template 得到默认人格、没有预置上下文」，不把这层边角写进模型可见的说明。

### 5.3 `{workspace}` 替换

文本块里的字面量 `{workspace}` 在注入时替换为工作区绝对路径（`str.replace`，不是
`.format`，所以正文里的其他花括号不会炸）。默认人格原本就带工作区路径，靠这一条
从 Python 常量迁进模板文件。

## 6. 对模型的可见性

`build_system_prompt(workspace, skills, templates)` 变为组合器：

```
general 模板的 system 文本（替换 {workspace}）
  + SKILLS_PROMPT_SECTION（有技能时）
  + TEMPLATES_PROMPT_SECTION（有模板时）
```

- `general` 缺失（内置目录被破坏）→ 退化为一行兜底文案并告警，不让启动失败；
- 人格部分逐字节等于原 `SYSTEM_PROMPT_TEMPLATE` 的渲染（含结尾换行）；无技能时不追加
  技能节。**但提示词整体不再与旧版相等**：内置模板目录永远非空，所以每次启动都会多出
  模板章节——这是本方案的预期产物，不是回归；
- `Agent` 工具的 agent doc 增补 `template` 参数与「未知模板返回可用清单」的行为。

## 7. 代码改动清单

| 文件 | 改动 |
| --- | --- |
| `core/agent/template/template.py` | 新增 `AgentTemplate`、`TemplateRoot`、`TemplateIssue` 数据类 |
| `core/agent/template/template_json.py` | 新增 `parse_template_json(text, expected_name)`，含块校验与配对规则 |
| `core/agent/template/template_manager.py` | 新增 `AgentTemplateManager`（`discover`/`list_templates`/`get`/`issues`） |
| `core/agent/template/apply.py` | 新增 `apply_template(session, template, workspace=None)` |
| `core/agent/template/builtin/general/template.json` | 默认人格（原 `SYSTEM_PROMPT_TEMPLATE` 文案 + `{workspace}`） |
| `core/agent/template/builtin/reviewer/template.json` | 内置样例（system 块 + user/assistant 上下文）；刻意不与工作区已有的 `plan_agent` 重名，否则每次启动都会多一条 shadow issue |
| `core/agent/template/__init__.py` | 导出数据类型、解析器与管理器。**刻意不导出 `apply_template`**：它 import `agent_session`，由 orchestrator 显式导入 `template.apply`，这样「只想要模板格式」的调用方不会顺带拖进 agent 运行时 |
| `core/agent/agent_orchestrator.py` | `create_session(..., template=None)` 并在注册前 `apply_template` |
| `core/tool/builtin/context.py` | `ToolContext.template_manager: Any = None` |
| `core/tool/builtin/agent_tools.py` | `spawn_agent(..., template=None)`：解析、错误串、透传 |
| `core/tool/builtin/register.py` | `Agent` 的 agent doc 增补 `template` |
| `cli/bootstrap.py` | 构建 manager + 发现 + 挂 context + 组合提示词（删掉硬编码人格常量） |
| `pyproject.toml` | `[tool.setuptools.package-data]` 打包内置模板 JSON |

**依赖方向不变**：`tool` 包不 import `agent` 包；模板管理器像 `skills_manager` 一样
以 duck-typed 句柄挂在 `ToolContext` 上，由 `bootstrap` 注入。

## 8. 测试计划（TDD，先红后绿）

`test/test_agent_template_json.py`
- 最小合法模板 / 缺 name / 缺 description / name 与目录名不符 / 非法 name 字符
- `session_autoload` 非数组、块非对象、未知 `kind`、`role` 非法
- `arguments_string` 非 JSON、`parameter` 与 `arguments_string` 同时缺失
- 配对：缺 id 自动分配；唯一未应答的 result 绑定到它；多个未应答 → 拒绝
- 整份拒绝：一个坏块导致 `MitoCodeTemplateDefinitionError`

`test/test_agent_template_manager.py`
- 三根发现、source 标签、排序
- 同名 shadow（workspace > global > builtin）并记录 issue
- 缺 `template.json`、符号链接、非目录根、不存在根
- 解析失败只损失自己

`test/test_agent_template_apply.py`
- 块按序注入、位于 system prompt 之后
- 首块为 system 时替换默认 system prompt
- `build_standard_context()` 产出 provider 合法序列（tool_calls + tool 结果配对）
- 空 autoload / None → 行为与今天完全一致
- `{workspace}` 替换；其他花括号原样保留

`test/test_builtin_agent_tools.py`（追加）
- 传 `template` 时把模板对象转交 orchestrator
- 未知模板 → 错误串列出可用模板
- 未挂 manager → 明确错误串（而不是 `AttributeError`）

`test/test_cli_bootstrap.py`（追加）
- 提示词含模板章节 / 空目录时不含
- 提示词人格来自 `general` 模板
- bootstrap 把 manager 挂到 context，且根含 global

## 9. 风险与取舍

- **模板体积**：`session_autoload` 会被完整塞进每次子代理的首个请求，一个塞满工具
  结果的模板会让每次派发都变贵。管理端不设上限（作者自控），文档提示。
- **系统提示词替换**是唯一有「隐式」味道的规则，靠 docstring + 提示词章节 + 测试
  三处锚定。
- **`general` 成为默认人格的唯一来源**：内置目录损坏会让人格退化为一行兜底文案，
  有告警日志，不阻断启动。
