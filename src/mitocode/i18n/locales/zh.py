"""Simplified Chinese message catalog for MitoCode's user-facing strings.

Keys mirror :mod:`en` one for one; the fallback in
:func:`mitocode.i18n.translator.t` guarantees English output for any key this
catalog has not translated yet.
"""

CATALOG = {
    "cli.app_help": "MitoCode 命令行界面。",
    "cli.cli.help": "启动交互式 CLI 聊天模式。",
    "cli.webui.help": "Web 界面的占位命令。",
    "cli.not_implemented": "'{name}' 尚未实现。",
    "cli.tui.help": "启动终端界面。注意：TUI 已废弃，不再维护。",
    "cli.tui.deprecated_warning": "警告：内置 TUI 已废弃，不再维护。",
    "cli.tui.launching_in": "{seconds} 秒后启动……",
    "cli.tui.loaded_env_file": "已从 {path} 加载 provider 配置",
    "cli.api_key_missing": (
        "未找到 API key。请在环境变量中设置 MITOCODE_API_KEY"
        "（或 OPENAI_API_KEY），或在当前目录的 .env 文件中配置。"
    ),
    "cli.tui.invalid_timeout": (
        "MITOCODE_TIMEOUT 的值 '{value}' 无效，将使用默认值 {default} 秒。"
    ),
    "cli.tui.launch_error": "MitoCode 启动失败：{error}",
    "cli.tui.workspace_error": "在工作目录 {path} 准备 workspace 失败：{error}",
    # 交互式 CLI（mitocode cli）
    "cli.cli.welcome": "MitoCode CLI——输入 /help 查看可用命令。",
    "cli.cli.no_provider_hint": (
        "尚未配置 Provider。运行 /connect（或 /login）进行配置；/help 查看命令列表。"
    ),
    "cli.cli.exit_hint": "按 Ctrl+D 或输入 /exit 退出。",
    "cli.cli.interrupted": "已中断。",
    "cli.cli.queued": "已排队：{text}",
    "cli.cli.bye": "再见。",
    "cli.cli.agent_label": "#{id} {name}",
    "cli.cli.launch_error": "CLI 启动失败：{error}",
    "cli.cli.provider_swapped": "Provider 已切换为 {model}（{base_url}）。",
    "cli.cli.provider_busy": "Agent 仍在运行；本轮结束后再试。",
    "cli.cli.corrupt_secret": "忽略无法读取的 provider 配置 {path}：{error}",
    # git 引导
    "cli.git.missing": "PATH 中未找到 git；请安装以便 MitoCode 跟踪工作区变更。",
    "cli.git.initialized": "已在 {path} 初始化新的 git 仓库。",
    "cli.git.init_failed": "在 {path} 执行 'git init' 失败：{error}",
    # 斜杠命令
    "cli.cmd.help.help": "显示可用命令。",
    "cli.cmd.clear.help": "清空当前会话上下文。",
    "cli.cmd.exit.help": "退出 CLI。",
    "cli.cmd.connect.help": (
        "配置聊天 Provider 并保存到 .mitocode/secret/provider.json。"
    ),
    "cli.cmd.login.help": "/connect 的别名。",
    "cli.cmd.model.help": "查看或切换当前模型。",
    "cli.cmd.mcp.help": "管理 MCP 服务器：/mcp list、/mcp add、/mcp remove。",
    "cli.cmd.unknown": "未知命令：{name}——输入 /help 查看列表。",
    "cli.cmd.cleared": "上下文已清空。",
    "cli.cmd.connect.ask_api_key": "API key（留空保持当前/默认）： ",
    "cli.cmd.connect.ask_base_url": "Base URL（留空用 {default}）： ",
    "cli.cmd.connect.ask_model": "模型（留空用 {default}）： ",
    "cli.cmd.connect.ask_timeout": "超时秒数（留空用 {default}）： ",
    "cli.cmd.connect.saved": "Provider 配置已保存到 {path}。",
    "cli.cmd.connect.gitignored": "已将 {path} 加入 .gitignore，密钥不会进入 git。",
    "cli.cmd.model.current": "当前模型：{model}",
    "cli.cmd.model.list_header": "可用模型：",
    "cli.cmd.model.pick": "模型名或序号（留空取消）： ",
    "cli.cmd.model.switched": "已切换模型：{model}。",
    "cli.cmd.model.not_available": "模型 '{model}' 不在可用列表中。",
    "cli.cmd.model.session_only": (
        "尚无 provider.json，本次切换仅当前会话有效；运行 /connect 可持久化。"
    ),
    "cli.cmd.model.invalid_number": "'{value}' 不是有效序号。",
    "cli.cmd.model.fetch_failed": "获取模型列表失败：{error}",
    "cli.cmd.model.empty_list": "Provider 返回的模型列表为空。",
    # MCP 命令
    "cli.cmd.mcp.none": "未配置 MCP 服务器（{path}）。",
    "cli.cmd.mcp.list_header": "MCP 服务器：",
    "cli.cmd.mcp.ask_name": "服务器名称： ",
    "cli.cmd.mcp.ask_kind": "类型（stdio 或 http）： ",
    "cli.cmd.mcp.ask_command": "启动命令（stdio）： ",
    "cli.cmd.mcp.ask_url": "服务器 URL（http）： ",
    "cli.cmd.mcp.invalid_kind": "类型必须是 'stdio' 或 'http'。",
    "cli.cmd.mcp.added": "已将 MCP 服务器 '{name}' 添加到 {path}。",
    "cli.cmd.mcp.removed": "已移除 MCP 服务器 '{name}'。",
    "cli.cmd.mcp.not_found": "没有名为 '{name}' 的 MCP 服务器。",
    "cli.cmd.mcp.ask_remove": "要移除的服务器名称： ",
    "cli.cmd.mcp.status": "{name}：{status}",
    "cli.cmd.mcp.started": "MCP：{connected} 已连接，{failed} 失败，{disabled} 已禁用。",
    "cli.cmd.mcp.load_error": "无法加载 MCP 配置：{error}",
    # 审批提示
    "cli.approval.prompt": "{tool} 请求执行：{reason}",
    "cli.approval.choices": "[1] 允许一次   [2] 总是允许此类   [3] 拒绝",
    "cli.approval.pick": "选择 [1-3]： ",
    "cli.approval.invalid": "请输入 1、2 或 3。",
    "cli.approval.confirm_rule": "记住规则 '{rule}' 供以后使用？[y/N]： ",
    "cli.approval.rule_saved": "规则已保存：{rule}",
    "cli.approval.denied": "已拒绝。",
}
