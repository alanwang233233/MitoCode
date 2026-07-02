import json
import time
from difflib import unified_diff
from typing import Dict, List, Optional

from rich import box
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.reactive import reactive
from textual.widgets import Header, Footer, Input, Static, RichLog
from textual.widgets import Tree
from textual.widgets._tree import TreeNode

from mitocode.backend.agent.orchestrator import AgentOrchestrator

STATUS_ICONS = {
    "idle": "\u26aa",
    "running": "\U0001f7e1",
    "finished": "\U0001f7e2",
    "cancelled": "\u26aa",
    "error": "\U0001f534",
}

ACTION_LABELS = {
    "bash": "Bash",
    "file.read": "Read File",
    "file.write": "Write File",
}


class ToolCallPanelSyntax(Static):
    code = reactive("")
    panel_type = "tool_call"

    def __init__(self, title: str, lang: str, show_line_number: bool = False):
        super().__init__()
        self.lang = lang
        self.title: str = title
        self.code = ""
        self.show_line_number = show_line_number
        self.finished: bool = False
        self.finished_success: bool = True
        self.finished_result: str = ""
        self.finished_show_line_number: bool = True
        self.finished_show_diff: bool = False
        self.finished_diff_origin: str = ""
        self.finished_diff_tofile: str = ""
        self.finished_line_number_range = None
        self._buffer = ""
        self._formatted = False
        self._action = ""

    def watch_code(self, old_code: str, new_code: str) -> None:
        self.refresh(layout=True)

    def append_code(self, content):
        if self._formatted:
            return
        self._buffer += content
        try:
            args, _ = json.JSONDecoder().raw_decode(self._buffer)
            self._formatted = True
            file_path = args.get("file_path", "")
            code = args.get("content", "")
            if file_path:
                label = ACTION_LABELS.get(self._action, self._action)
                self.title = f"{label} → {file_path}"
            if code:
                self.code = code
            else:
                self.code = self._buffer
        except json.JSONDecodeError:
            pass

    def _finalize(self):
        if not self._formatted:
            try:
                args = json.loads(self._buffer)
            except (json.JSONDecodeError, TypeError):
                self.code = self._buffer
                return
            file_path = args.get("file_path", "")
            code = args.get("content", "")
            if file_path:
                label = ACTION_LABELS.get(self._action, self._action)
                self.title = f"{label} → {file_path}"
            self.code = code if code else self._buffer
            self._formatted = True

    def render(self):
        if self.finished:
            self._finalize()
            if self.finished_show_diff:
                old_code = self.finished_diff_origin.splitlines(keepends=True)
                new_code = self.code.splitlines(keepends=True)
                diff_lines = unified_diff(
                    old_code,
                    new_code,
                    fromfile="",
                    tofile=self.finished_diff_tofile,
                    lineterm="",
                )
                diff_out = "".join(diff_lines) or "No diff"
                syntax = Syntax(
                    diff_out,
                    "diff",
                    theme="monokai",
                    background_color="#282a36",
                    line_numbers=self.finished_show_line_number,
                    line_range=self.finished_line_number_range,
                )
                return Panel(
                    syntax,
                    title=(
                        f"[bold green]{self.title} ✓[/bold green]"
                        if self.finished_success
                        else f"[bold red]{self.title} ✗[/bold red]"
                    ),
                    title_align="left",
                    border_style="",
                    box=box.SIMPLE,
                    style="on #282a36",
                    padding=(1, 1, 0, 1),
                )
            syntax = Syntax(
                self.code,
                self.lang,
                theme="monokai",
                background_color="#282a36",
                line_numbers=self.finished_show_line_number,
                line_range=self.finished_line_number_range,
            )
            return Panel(
                syntax,
                title=(
                    f"[bold green]{self.title} ✓[/bold green]"
                    if self.finished_success
                    else f"[bold red]{self.title} ✗[/bold red]"
                ),
                title_align="left",
                border_style="",
                box=box.SIMPLE,
                style="on #282a36",
                padding=(1, 1, 0, 1),
            )
        if self._formatted:
            syntax = Syntax(
                self.code,
                self.lang,
                theme="monokai",
                background_color="#282a36",
                line_numbers=False,
            )
            return Panel(
                syntax,
                title=f"[bold cyan]{self.title}[/bold cyan]",
                title_align="left",
                border_style="",
                box=box.SIMPLE,
                style="on #282a36",
                padding=(1, 1, 0, 1),
            )
        return Panel(
            "...",
            title=f"[bold cyan]{self.title}[/bold cyan]",
            title_align="left",
            border_style="",
            box=box.SIMPLE,
            style="on #282a36",
            padding=(1, 1, 0, 1),
        )

    def end(self, success: bool = True, show_line_number: bool = False):
        self.finished = True
        self.refresh(layout=True)


class ToolCallPanel(Static):
    panel_type = "tool_call"

    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        self.content = ""
        self.finished = False
        self.finished_success = True
        self._buffer = ""
        self._formatted = False

    def append_code(self, delta: str):
        if self._formatted:
            return
        self._buffer += delta
        try:
            args, _ = json.JSONDecoder().raw_decode(self._buffer)
            self._formatted = True
            cmd = args.get("command", "")
            path = args.get("file_path", "")
            if cmd:
                self.content = f"$ {cmd}"
            elif path:
                self.content = path
            else:
                self.content = self._buffer
            self.refresh(layout=True)
        except json.JSONDecodeError:
            pass

    def _finalize(self):
        if not self._formatted:
            try:
                args = json.loads(self._buffer)
            except (json.JSONDecodeError, TypeError):
                self.content = self._buffer
                return
            cmd = args.get("command", "")
            path = args.get("file_path", "")
            if cmd:
                self.content = f"$ {cmd}"
            elif path:
                self.content = path
            else:
                self.content = self._buffer
            self._formatted = True

    def render(self):
        if self.finished:
            status = "✓" if self.finished_success else "✗"
            color = "green" if self.finished_success else "red"
            title = f"[bold {color}]{self.title_text} {status}[/bold {color}]"
        elif self._formatted:
            title = f"[bold cyan]{self.title_text}[/bold cyan]"
        else:
            return Panel(
                "...",
                title=f"[bold cyan]{self.title_text}[/bold cyan]",
                title_align="left",
                border_style="",
                box=box.SIMPLE,
                style="on #282a36",
                padding=(1, 1, 0, 1),
            )
        return Panel(
            self.content,
            title=title,
            title_align="left",
            border_style="",
            box=box.SIMPLE,
            style="on #282a36",
            padding=(1, 1, 0, 1),
        )

    def end(self, success: bool = True):
        self.finished = True
        self.finished_success = success
        self.refresh(layout=True)


class UserMessagePanel(Static):
    panel_type = "user"

    def __init__(self, text: str):
        super().__init__()
        self.user_text = text

    def render(self):
        return Panel(
            self.user_text,
            title="",
            border_style="",
            box=box.SIMPLE,
            style="on #1e2d3d",
            padding=(0, 1),
        )


class PlainTextPanel(Static):
    panel_type = "plain_text"
    content = reactive("", layout=True)

    def append(self, delta: str):
        self.content += delta

    def render(self):
        return Panel(
            RichMarkdown(self.content) if self.content else "",
            title="",
            border_style="",
            box=box.SIMPLE,
            padding=(0, 1),
        )


class ReasoningTextPanel(Static):
    panel_type = "reasoning_text"
    content = reactive("", layout=True)

    def append(self, delta: str):
        self.content += delta

    def render(self):
        return Panel(
            RichMarkdown(self.content) if self.content else "",
            title="",
            border_style="",
            box=box.SIMPLE,
            style="on #2e2e1a",
            padding=(0, 1),
        )


class AgentView(VerticalScroll):
    def __init__(self, agent_id: int, agent_name: str = ""):
        super().__init__()
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.border_title = f"Agent {agent_id}"
        self.styles.border = ("solid", "cyan")
        self._streaming_panel: Optional[Static] = None

    def on_mount(self):
        self.styles.overflow_y = "auto"
        self.styles.scrollbar_gutter = "stable"

    def set_active(self, active: bool):
        if active:
            self.styles.border = ("heavy", "gold")
        else:
            self.styles.border = ("solid", "cyan")

    @property
    def _last_panel_type(self) -> str:
        if self._streaming_panel is not None:
            return getattr(self._streaming_panel, "panel_type", "")
        return ""

    def _scroll_to_bottom(self):
        self.call_after_refresh(self.scroll_end, animate=False)

    def _check_scroll_before_append(self) -> bool:
        return self.scroll_offset.y >= self.max_scroll_y

    def _scroll_if_was_at_bottom(self, was_at_bottom: bool):
        if was_at_bottom:
            self._scroll_to_bottom()

    async def append_user_message(self, text: str) -> None:
        self._streaming_panel = None
        panel = UserMessagePanel(text)
        await self.mount(panel)
        self._scroll_to_bottom()

    async def append_plain_text(self, delta: str) -> None:
        if self._last_panel_type != "plain_text":
            self._streaming_panel = None
            panel = PlainTextPanel()
            await self.mount(panel)
            self._streaming_panel = panel
            self._scroll_to_bottom()
        if self._streaming_panel is not None:
            was_at_bottom = self._check_scroll_before_append()
            self._streaming_panel.append(delta)
            self._scroll_if_was_at_bottom(was_at_bottom)

    async def append_reasoning(self, delta: str) -> None:
        if self._last_panel_type != "reasoning_text":
            self._streaming_panel = None
            panel = ReasoningTextPanel()
            await self.mount(panel)
            self._streaming_panel = panel
            self._scroll_to_bottom()
        if self._streaming_panel is not None:
            was_at_bottom = self._check_scroll_before_append()
            self._streaming_panel.append(delta)
            self._scroll_if_was_at_bottom(was_at_bottom)

    async def start_tool_call(
        self, action: str, description: str, additional: str
    ) -> None:
        self._streaming_panel = None

        if action in ("file.write",):
            lang = "python"
            if additional and additional.endswith(".json"):
                lang = "json"
            elif additional and additional.endswith(".sh"):
                lang = "bash"
            elif additional and additional.endswith(".js"):
                lang = "javascript"
            elif additional and additional.endswith(".md"):
                lang = "markdown"
            title = (
                f"[{action}] {description} - {additional}"
                if additional
                else f"[{action}] {description}"
            )
            panel = ToolCallPanelSyntax(title=title, lang=lang)
            panel._action = action
        else:
            title = ACTION_LABELS.get(action, action)
            panel = ToolCallPanel(title=title)

        await self.mount(panel)
        self._streaming_panel = panel
        self._scroll_to_bottom()

    async def append_tool_content(self, content: str) -> None:
        if self._streaming_panel is not None and self._last_panel_type == "tool_call":
            self._streaming_panel.append_code(content)

    async def finish_tool_call(self, result: str) -> None:
        if self._streaming_panel is not None and self._last_panel_type == "tool_call":
            panel = self._streaming_panel
            success = "error" not in result.lower() and "fail" not in result.lower()
            if isinstance(panel, ToolCallPanelSyntax):
                panel._finalize()
                panel.finished_result = result
                panel.finished_success = success
                panel.end()
            else:
                panel._finalize()
                panel.content = result
                panel.end(success=success)
        self._streaming_panel = None
        self._scroll_to_bottom()


class DynamicGrid(Grid):
    def on_mount(self):
        self.styles.layout = "grid"

    async def set_widgets(self, widgets):
        await self.query(".ellipsis").remove()
        for w in self.query("AgentView"):
            w.styles.display = "none"

        n = len(widgets)
        if n == 0:
            return

        if n == 1:
            self.styles.grid_size_columns = 1
            self.styles.grid_size_rows = 1
            self.styles.grid_columns = "1fr"
            self.styles.grid_rows = "1fr"

            widget = widgets[0]
            if widget not in self.children:
                await self.mount(widget)
            widget.styles.display = "block"
            widget.styles.row_span = 1
            widget.styles.column_span = 1
        else:
            max_right_items = 3
            actual_right_rows = min(n - 1, max_right_items)

            self.styles.grid_size_columns = 2
            self.styles.grid_size_rows = actual_right_rows
            self.styles.grid_columns = "2fr 1fr"
            self.styles.grid_rows = " ".join(["1fr"] * actual_right_rows)

            left_widget = widgets[0]
            if left_widget not in self.children:
                await self.mount(left_widget)
            left_widget.styles.display = "block"
            left_widget.styles.row_span = actual_right_rows
            left_widget.styles.column_span = 1

            for i in range(1, actual_right_rows + 1):
                if i == actual_right_rows and n - 1 > actual_right_rows:
                    remaining = n - actual_right_rows
                    widget = Static(
                        f"+ {remaining} agents", classes="agent-view ellipsis"
                    )
                    await self.mount(widget)
                else:
                    widget = widgets[i]
                    if widget not in self.children:
                        await self.mount(widget)
                    widget.styles.display = "block"
                widget.styles.row_span = 1
                widget.styles.column_span = 1

        self.refresh(layout=True)


class AgentTree(Tree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_click_time = 0
        self._last_click_node = None

    def on_click(self, event) -> None:
        node = self.get_node_at_line(event.y)
        if node:
            self.select_node(node)
            current_time = time.time()
            if (
                current_time - self._last_click_time < 0.4
                and node == self._last_click_node
            ):
                node.toggle()
                self._last_click_time = 0
            else:
                self._last_click_time = current_time
                self._last_click_node = node
            event.stop()
            self.focus()


class VibeFrontend(App):
    TITLE = "MitoCode"

    CSS = """
    Screen {
        layout: vertical;
        background: #282a36;
    }

    Horizontal {
        height: 1fr;
    }

    .left-pane {
        width: 30;
        height: 100%;
        border-right: tall $accent;
    }

    #quick-create {
        dock: bottom;
        margin: 0 1;
        height: 3;
    }

    DynamicGrid {
        layout: grid;
        grid-gutter: 1;
        width: 1fr;
        height: 1fr;
        min-width: 50;
    }

    .right-pane {
        width: 1fr;
        height: 100%;
    }

    .chat-input {
        dock: bottom;
        margin: 0 1 1 1;
        height: 3;
    }

    AgentView {
        border: panel $primary;
        background: $surface;
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 1;
    }

    #log-panel {
        height: 30%;
    }

    ToolCallPanelSyntax {
        height: auto;
        width: 100%;
        margin-bottom: 0;
    }

    ToolCallPanel {
        height: auto;
        width: 100%;
        margin-bottom: 0;
    }

    UserMessagePanel {
        height: auto;
        width: 100%;
        margin: 1 0 1 0;
    }

    PlainTextPanel {
        height: auto;
        width: 100%;
        margin-bottom: 0;
    }

    ReasoningTextPanel {
        height: auto;
        width: 100%;
        margin-bottom: 0;
    }
    """

    BINDINGS = [
        ("ctrl+up", "focus_previous", "Prev Agent"),
        ("ctrl+down", "focus_next", "Next Agent"),
        ("escape", "cancel_agent", "Cancel Agent"),
        ("f5", "refresh_layout", "Refresh"),
        ("f12", "toggle_debug", "Debug Log"),
    ]

    debug_mode = reactive(False)

    def __init__(self, orchestrator: AgentOrchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.agent_tree: Optional[AgentTree] = None
        self.grid: Optional[DynamicGrid] = None
        self.chat_input: Optional[Input] = None
        self.quick_create: Optional[Input] = None
        self.agents: Dict[int, AgentView] = {}
        self.active_agent_id: Optional[int] = None
        self.agent_treenodes: Dict[int, TreeNode] = {}
        self.log_panel: Optional[RichLog] = None

    def print_log(self, text):
        if self.log_panel:
            self.log_panel.write(text)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Container(classes="left-pane"):
                self.agent_tree = AgentTree("Agent Tree")
                self.agent_tree.show_root = False
                yield self.agent_tree
                self.quick_create = Input(
                    placeholder="Quick create: name & Enter",
                    id="quick-create",
                )
                yield self.quick_create
            with Container(classes="right-pane"):
                self.grid = DynamicGrid()
                yield self.grid
                self.chat_input = Input(
                    placeholder="Send message...",
                    id="chat-input",
                    classes="chat-input",
                )
                yield self.chat_input
        self.log_panel = RichLog(id="log-panel")
        self.log_panel.styles.display = "none"
        yield self.log_panel
        yield Footer()

    def on_mount(self):
        self._update_and_render_tree(self.orchestrator.get_tree_snapshot())
        self.run_worker(self._consume_events(), group="workers", name="event_consumer")
        tree = self.agent_tree
        if tree.root and tree.root.children:
            first_child = tree.root.children[0]
            tree.select_node(first_child)
            tree.scroll_to_node(first_child)
        if self.chat_input:
            self.chat_input.focus()

    def action_toggle_debug(self) -> None:
        self.debug_mode = not self.debug_mode

    def action_refresh_layout(self) -> None:
        self.print_log("Refreshing layout...")
        self.refresh(layout=True)

    def action_cancel_agent(self) -> None:
        if self.chat_input and self.chat_input.has_focus and self.chat_input.value:
            self.chat_input.value = ""
            return
        if self.active_agent_id is not None:
            self.orchestrator.cancel_agent(self.active_agent_id, cascade=True)
            self.print_log(f"Cancelled Agent {self.active_agent_id}")

    def action_focus_previous(self) -> None:
        if not self.agent_tree:
            return
        visible = list(self.agent_treenodes.values())
        if not visible:
            return
        current = self.active_agent_id
        idx = next(
            (
                i
                for i, n in enumerate(visible)
                if n.data and n.data.get("id") == current
            ),
            0,
        )
        prev_idx = (idx - 1) % len(visible)
        self._select_tree_node(visible[prev_idx])

    def action_focus_next(self) -> None:
        if not self.agent_tree:
            return
        visible = list(self.agent_treenodes.values())
        if not visible:
            return
        current = self.active_agent_id
        idx = next(
            (
                i
                for i, n in enumerate(visible)
                if n.data and n.data.get("id") == current
            ),
            -1,
        )
        next_idx = (idx + 1) % len(visible)
        self._select_tree_node(visible[next_idx])

    def _select_tree_node(self, node: TreeNode) -> None:
        self.agent_tree.select_node(node)
        self.agent_tree.scroll_to_node(node)

    def watch_debug_mode(self, debug_mode: bool):
        if self.log_panel:
            self.log_panel.styles.display = "block" if debug_mode else "none"

    # ------------------------------------------------------------------
    # Event consumption
    # ------------------------------------------------------------------

    async def _consume_events(self):
        while True:
            agent_id, event = await self.orchestrator.event_queue.get()
            try:
                await self._handle_agent_event(agent_id, event)
            except Exception as e:
                self.print_log(f"Event error (agent={agent_id}): {e}")

    async def _handle_agent_event(self, agent_id: int, event: dict) -> None:
        event_type = event.get("type")

        if event_type == "sync":
            agents_delta = event.get("agents", [])
            if agents_delta:
                self._update_and_render_tree(agents_delta)
            return

        if agent_id < 0:
            return

        if agent_id not in self.agents:
            agent_data = event.get("agent_data", {})
            name = agent_data.get("name", "")
            self.print_log(f"Created AgentView for agent {agent_id}")
            self.agents[agent_id] = AgentView(agent_id, name)

        agent_view = self.agents[agent_id]

        if event_type == "token":
            await agent_view.append_plain_text(event.get("content", ""))
        elif event_type == "tool_call":
            if "action" in event:
                action = event.get("action", "Unknown")
                description = event.get("description", "Tool Call")
                additional = event.get("additional", "")
                await agent_view.start_tool_call(action, description, additional)
            elif "content" in event:
                await agent_view.append_tool_content(event.get("content", ""))
        elif event_type == "tool_result":
            result = event.get("result", "")
            await agent_view.finish_tool_call(result)

    # ------------------------------------------------------------------
    # Quick create & chat input
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        if event.input == self.quick_create:
            parent_id = self.active_agent_id
            self.orchestrator.create_agent(
                name=text,
                provider=None,
                description="",
                parent_id=parent_id,
            )
            self.quick_create.value = ""
            self.print_log(f"Created agent '{text}' (parent={parent_id})")

        elif event.input == self.chat_input:
            if self.active_agent_id is None:
                self.print_log("No agent selected.")
                self.chat_input.value = ""
                return

            if self.active_agent_id in self.agent_treenodes:
                node_data = self.agent_treenodes[self.active_agent_id].data
                if node_data and node_data.get("state") == "running":
                    self.print_log("Agent is busy, please wait.")
                    return

            if self.active_agent_id not in self.agents:
                name = ""
                if self.active_agent_id in self.agent_treenodes:
                    node = self.agent_treenodes[self.active_agent_id]
                    name = node.data.get("name", "") if node.data else ""
                self.agents[self.active_agent_id] = AgentView(
                    self.active_agent_id, name
                )

            view = self.agents[self.active_agent_id]
            await view.append_user_message(text)

            await self.orchestrator.send_message(self.active_agent_id, text)
            self.print_log(f"Sent to Agent {self.active_agent_id}: {text[:50]}...")
            self.chat_input.value = ""

    # ------------------------------------------------------------------
    # Tree rendering
    # ------------------------------------------------------------------

    def _update_and_render_tree(self, agents_list: list) -> None:
        if not self.agent_tree:
            return

        previous_active = self.active_agent_id

        agent_map: Dict[int, dict] = {}
        for a in agents_list:
            agent_map[a["id"]] = a

        self.agent_tree.clear()
        self.agent_treenodes.clear()

        all_child_ids = set()
        for a in agents_list:
            all_child_ids.update(a.get("sub_agent", []))

        root_candidates = [a for a in agents_list if a["id"] not in all_child_ids]

        for root_data in root_candidates:
            self._build_tree_recursive(self.agent_tree.root, root_data, agent_map)

        if previous_active is not None and previous_active in self.agent_treenodes:
            node = self.agent_treenodes[previous_active]
            self._select_tree_node(node)

    def _build_tree_recursive(self, parent_node, current_agent_data, agent_map):
        agent_id = current_agent_data["id"]
        state = current_agent_data.get("state", "idle")
        icon = STATUS_ICONS.get(state, STATUS_ICONS["idle"])
        name = current_agent_data.get("name", f"Agent {agent_id}")
        label = f"{icon} [{agent_id}] {name}"

        new_node = parent_node.add(label, data=current_agent_data)
        self.agent_treenodes[agent_id] = new_node

        for sub_id in current_agent_data.get("sub_agent", []):
            if sub_id in agent_map:
                self._build_tree_recursive(new_node, agent_map[sub_id], agent_map)

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    async def on_tree_node_selected(self, event: Tree.NodeSelected):
        node = event.node
        if not node or not node.data:
            return
        self.active_agent_id = node.data["id"]

        display_nodes = [node] + list(node.children)
        agent_views = []
        for n in display_nodes:
            agent_id = n.data["id"]
            if agent_id not in self.agents:
                name = n.data.get("name", "")
                self.agents[agent_id] = AgentView(agent_id, name)
            agent_views.append(self.agents[agent_id])

        await self.grid.set_widgets(agent_views)
        self._update_active_border()
        self._update_chat_placeholder()

    def _update_active_border(self):
        for agent_id, view in self.agents.items():
            view.set_active(agent_id == self.active_agent_id)

    def _update_chat_placeholder(self):
        if self.chat_input is None:
            return
        if (
            self.active_agent_id is not None
            and self.active_agent_id in self.agent_treenodes
        ):
            node = self.agent_treenodes[self.active_agent_id]
            name = node.data.get("name", "") if node.data else ""
            placeholder = f"Send to Agent {self.active_agent_id} ({name})..."
        else:
            placeholder = "Send message..."
        self.chat_input.placeholder = placeholder
