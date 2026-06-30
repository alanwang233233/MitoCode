import argparse
import json
import time
from difflib import unified_diff
from typing import Dict

import httpx
from httpx_sse import aconnect_sse
from rich.panel import Panel
from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Markdown, RichLog
from textual.widgets import Tree
from textual.widgets._tree import TreeNode


class ToolCallPanel(Container):
    def __init__(self, lang: str):
        super().__init__()
        pass


class ToolCallPanelSyntax(Static):
    code = reactive("")

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

    def watch_code(self, old_code: str, new_code: str) -> None:
        self.refresh(layout=True)

    def append_code(self, content):
        self.code = self.code + content

    def render(self):
        if self.finished:
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
                diff_out = "".join(diff_lines)
                if diff_out == "":
                    diff_out = "No diff"
                syntax = Syntax(
                    diff_out,
                    "diff",
                    theme="monokai",
                    line_numbers=self.finished_show_line_number,
                    line_range=self.finished_line_number_range,
                )
                return Panel(
                    syntax,
                    title=self.title + " " + self.finished_result,
                    border_style="green" if self.finished_success else "red",
                )
            syntax = Syntax(
                self.code,
                self.lang,
                theme="monokai",
                line_numbers=self.finished_show_line_number,
                line_range=self.finished_line_number_range,
            )
            return Panel(
                syntax,
                title=self.title + " " + self.finished_result,
                border_style="green" if self.finished_success else "red",
            )
        syntax = Syntax(
            self.code, self.lang, theme="monokai", line_numbers=self.show_line_number
        )
        return Panel(syntax, title=self.title, border_style="cyan")

    def end(self, success: bool = True, show_line_number: bool = False):
        self.finished = True
        self.refresh(layout=True)


class AgentView(VerticalScroll):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.border_title = f"Agent {agent_id}"
        self.styles.border = ("solid", "green")
        self._current_md_content = ""

    def compose(self) -> ComposeResult:
        yield Markdown("")

    def on_mount(self):
        self.styles.overflow_y = "auto"
        self.styles.scrollbar_gutter = "stable"

    async def append_token(self, token: str) -> None:
        was_at_bottom = self.scroll_offset.y >= self.max_scroll_y

        self._current_md_content += token
        md_widgets = self.query(Markdown)
        if md_widgets:
            await md_widgets.last().update(self._current_md_content)

            if was_at_bottom:
                self.call_after_refresh(self.scroll_end, animate=False)

    async def start_tool_call(
        self, action: str, description: str, additional: str
    ) -> None:
        was_at_bottom = self.scroll_offset.y >= self.max_scroll_y

        lang = "python"
        if additional.endswith(".json"):
            lang = "json"
        elif additional.endswith(".sh"):
            lang = "bash"
        elif additional.endswith(".js"):
            lang = "javascript"
        elif additional.endswith(".md"):
            lang = "markdown"

        title = f"[{action}] {description} - {additional}"

        tool_panel = ToolCallPanelSyntax(title=title, lang=lang)
        await self.mount(tool_panel)

        self._current_md_content = ""
        await self.mount(Markdown(""))

        if was_at_bottom:
            self.scroll_end(animate=False)

    async def append_tool_content(self, content: str) -> None:
        was_at_bottom = self.scroll_offset.y >= self.max_scroll_y

        tool_panels = self.query(ToolCallPanelSyntax)
        if tool_panels:
            last_panel = tool_panels.last()
            last_panel.append_code(content)

            if was_at_bottom:
                self.call_after_refresh(self.scroll_end, animate=False)

    async def finish_tool_call(self, result: str) -> None:
        was_at_bottom = self.scroll_offset.y >= self.max_scroll_y

        tool_panels = self.query(ToolCallPanelSyntax)
        if tool_panels:
            last_panel = tool_panels.last()
            last_panel.finished_result = result
            last_panel.finished_success = (
                "error" not in result.lower() and "fail" not in result.lower()
            )
            last_panel.end()

        if was_at_bottom:
            self.scroll_end(animate=False)


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
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("f12", "toggle_debug", "Toggle debug mode"),
        ("f5", "refresh_layout", "Refresh layout"),
    ]

    CSS = """
        Screen {
            layout: vertical;
        }

        Horizontal {
            height: 1fr;
        }

        .left-pane {
            width: 30;
            height: 100%;
            border-right: tall $accent;
        }

        DynamicGrid {
            layout: grid;
            grid-gutter: 1;
            width: 1fr;
            height: 100%;
            min-width: 50;
        }

        .agent-view {
            border: panel $primary;
            background: $surface;
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 1;
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
            margin-bottom: 1;
        }

        /* Markdown 也需要 auto */
        AgentView > Markdown {
            height: auto;
            width: 100%;
            margin-bottom: 1;
        }
        """

    debug_mode = reactive("")

    def __init__(self, sse_url, sync_url):
        super().__init__()
        self.agent_tree: AgentTree = None  # type: ignore
        self.grid: DynamicGrid = None  # type: ignore
        self.agents: Dict[int, AgentView] = {}
        self.sse_url: str = sse_url
        self.sync_url: str = sync_url
        self.workspace: str = ""
        self.agent_treenodes: Dict[int, TreeNode] = {}
        self.agent_data_store: Dict[int, dict] = {}
        self.log_panel = RichLog()

    def print_log(self, text):
        self.log_panel.write(text)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            self.agent_tree = AgentTree("Agent Tree", classes="left-pane")
            yield self.agent_tree
            self.grid = DynamicGrid(classes="right-pane")
            yield self.grid
        self.log_panel = RichLog(id="log-panel")
        yield self.log_panel
        yield Footer()

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def action_toggle_debug(self) -> None:
        self.debug_mode = "block" if self.debug_mode == "none" else "none"

    def action_refresh_layout(self) -> None:
        self.print_log("Refreshing layout ...")
        self.refresh(layout=True)

    def watch_debug_mode(self, debug_mode: bool):
        if debug_mode:
            self.log_panel.styles.display = "block"
        else:
            self.log_panel.styles.display = "none"

    async def handle_sse_event(self, event):
        action_type = event.get("type")
        agent_id = int(event["agent_id"])

        if agent_id not in self.agents:
            self.print_log(f"Temporary AgentView is created here for agent {agent_id}")
            self.agents[agent_id] = AgentView(agent_id)

        agent_view = self.agents[agent_id]

        if action_type == "token":
            await agent_view.append_token(event.get("content", ""))

        elif action_type == "tool_call":
            if "action" in event:
                action = event.get("action", "Unknown")
                description = event.get("description", "Tool Call")
                additional = event.get("additional", "")
                await agent_view.start_tool_call(action, description, additional)
            elif "content" in event:
                await agent_view.append_tool_content(event.get("content", ""))

        elif action_type == "tool_result":
            result = event.get("result", "")
            await agent_view.finish_tool_call(result)

    async def _sse_single_agent(self, agent_id):
        url = f"{self.sse_url}?id={agent_id}"

        try:
            self.print_log(f"Starting SSE for Agent {agent_id}: {url}")
            async with httpx.AsyncClient(timeout=None) as client:
                async with aconnect_sse(client, "GET", url) as event_source:
                    async for sse in event_source.aiter_sse():
                        if sse.data:
                            try:
                                event = json.loads(sse.data)
                                await self.handle_sse_event(event)
                            except json.JSONDecodeError:
                                continue
            self.print_log(f"SSE for Agent {agent_id} completed.")
        except Exception as e:
            self.print_log(f"❌ SSE Error for {agent_id}: {repr(e)}")

    async def _sse_global_sync(self):
        url = self.sync_url

        try:
            self.print_log(f"Starting Global Sync SSE on {url}")
            async with httpx.AsyncClient(timeout=None) as client:
                async with aconnect_sse(client, "GET", url) as event_source:
                    async for sse in event_source.aiter_sse():
                        if sse.data:
                            try:
                                payload = json.loads(sse.data)
                                if payload.get("event") == "sync":
                                    agents_delta = payload.get("agents", [])
                                    if agents_delta:
                                        self.print_log(
                                            f"Received {len(agents_delta)} deltas"
                                        )
                                        self._update_and_render_tree(agents_delta)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            self.print_log(f"❌ SSE Error: {repr(e)}")

    def on_mount(self):
        self.agent_tree.clear()
        self.agent_tree.show_root = False

        self.run_worker(
            self._sse_global_sync(), group="sse_workers", name="global_sync_worker"
        )

    def _update_and_render_tree(self, incoming_agents: list):
        for agent_update in incoming_agents:
            agent_id = agent_update["id"]
            if agent_id not in self.agent_data_store:
                self.agent_data_store[agent_id] = agent_update
            else:
                self.agent_data_store[agent_id].update(agent_update)

        self.agent_tree.clear()
        self.agent_treenodes.clear()

        agents_list = list(self.agent_data_store.values())
        if not agents_list:
            return

        all_child_ids = set()
        for a in agents_list:
            all_child_ids.update(a.get("sub_agent", []))

        root_candidates = [a for a in agents_list if a["id"] not in all_child_ids]

        for root_data in root_candidates:
            self._build_tree_recursive(
                self.agent_tree.root, root_data, self.agent_data_store
            )

    def _build_tree_recursive(self, parent_node, current_agent_data, agent_map):
        agent_id = current_agent_data["id"]

        new_node = parent_node.add(current_agent_data["name"], data=current_agent_data)
        self.agent_treenodes[agent_id] = new_node

        for sub_id in current_agent_data.get("sub_agent", []):
            if sub_id in agent_map:
                self._build_tree_recursive(new_node, agent_map[sub_id], agent_map)

    async def on_tree_node_highlighted(self, event: Tree.NodeHighlighted):
        node = event.node
        if not node or not node.data:
            return

        self.workers.cancel_group(self, "sse_workers")

        display_nodes = [node] + list(node.children)
        agent_views = []
        for n in display_nodes:
            self.print_log(str(n.data))
            agent_id = n.data["id"]
            if agent_id not in self.agents:
                self.agents[agent_id] = AgentView(agent_id)
            agent_views.append(self.agents[agent_id])
        self.print_log(f"self.agents: {self.agents}")
        await self.grid.set_widgets(agent_views)

        for view in agent_views[:4]:  # 只为当前显示的启动
            self.run_worker(
                self._sse_single_agent(view.agent_id),
                group="sse_workers",
                name=f"sse_{view.agent_id}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vibe Frontend Textual App")

    parser.add_argument(
        "--sse",
        default="http://127.0.0.1:8000/message",
        help="SSE endpoint URL for agent messages",
    )
    parser.add_argument(
        "--sync",
        default="http://127.0.0.1:8000/sync",
        help="SSE endpoint URL for global sync",
    )

    args = parser.parse_args()
    app = VibeFrontend(sse_url=args.sse, sync_url=args.sync)
    app.run()
