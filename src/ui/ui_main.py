import json
from typing import Dict

import httpx
from httpx_sse import aconnect_sse
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll, Grid
from textual.reactive import reactive
from textual.widgets import Header, Footer, Tree, Static, Markdown, RichLog
from textual.widgets._tree import TreeNode

log_panel: RichLog = None

def print_log(text):
    log_panel.write(text)

class ToolCallView(Static):
    def __init__(self, tool, args):
        super().__init__(f"🔧 {tool}\n{args}")


class ToolResultView(Static):
    def __init__(self, result):
        super().__init__(f"📦 {result}")


from textual.reactive import reactive


class AgentView(VerticalScroll):

    content = reactive("")

    def __init__(self, agent_id):
        super().__init__(classes="agent-view")
        self.agent_id = agent_id
        self.border_title = f"Agent {agent_id}"
        self.styles.border = ("solid", "green")

    def compose(self) -> ComposeResult:
        yield Markdown(self.content)

    def on_mount(self):
        print_log(f"on_mount {id(self)}: {True if self.query(Markdown) else False}")

    def watch_content(self, content: str) -> None:
        md_widgets = self.query(Markdown)

        if md_widgets:
            md_widgets.first().update(content)
            self.scroll_end(animate=False)
        else:
            pass

    def append_token(self, token: str) -> None:
        self.content += token


class DynamicGrid(Grid):
    def on_mount(self):
        self.styles.layout = "grid"

    async def set_widgets(self, widgets):
        await self.query("*").remove()

        n = len(widgets)
        if n == 0: return

        if n == 1:
            self.styles.grid_size_columns = 1
            self.styles.grid_size_rows = 1
            self.styles.grid_columns = "1fr"
            self.styles.grid_rows = "1fr"

            widgets[0].styles.row_span = 1
            widgets[0].styles.column_span = 1
            await self.mount(widgets[0])

        else:
            max_right_items = 3
            actual_right_rows = min(n - 1, max_right_items)

            self.styles.grid_size_columns = 2
            self.styles.grid_size_rows = actual_right_rows

            self.styles.grid_columns = "2fr 1fr"
            # 动态生成行高，例如 "1fr 1fr 1fr"
            self.styles.grid_rows = " ".join(["1fr"] * actual_right_rows)

            left_widget = widgets[0]
            left_widget.styles.row_span = actual_right_rows
            left_widget.styles.column_span = 1
            # 背景色用于调试，确认后可删除
            left_widget.styles.background = "#1f3e1f"
            await self.mount(left_widget)

            for i in range(1, actual_right_rows + 1):
                if i == actual_right_rows and n - 1 > actual_right_rows:
                    remaining = n - actual_right_rows
                    widget = Static(f"+ {remaining} agents", classes="agent-view ellipsis")
                else:
                    widget = widgets[i]

                widget.styles.row_span = 1
                widget.styles.column_span = 1
                widget.styles.background = "#1f1f3e"  # 调试背景色
                await self.mount(widget)

        self.refresh(layout=True)

class VibeFrontend(App):

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    CSS = """
        Screen {
            layout: vertical;
        }

        Horizontal {
            height: 1fr; /* 填满 Header 和 Footer 之间 */
        }

        .left-pane {
            width: 30;
            height: 100%;
            border-right: tall $accent;
        }

        DynamicGrid {
            layout: grid; /* 显式声明 */
            grid-gutter: 1;
            width: 1fr;
            height: 100%;
            min-width: 50; /* 确保有足够宽度开启两列布局 */
        }
        
        .agent-view {
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
        """

    def __init__(self):
        super().__init__()
        self.agent_tree: Tree = None
        self.grid: DynamicGrid = None
        self.agents: Dict[int, AgentView] = {}
        self.sse_url: str = "http://127.0.0.1:8000/message"
        self.sync_url: str = "http://127.0.0.1:8000/sync"
        self.workspace: str = ""
        self.agent_treenodes: Dict[int, TreeNode] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            self.agent_tree = Tree("Agent Tree", classes="left-pane")
            yield self.agent_tree
            self.grid = DynamicGrid(classes="right-pane")
            yield self.grid
        global log_panel
        log_panel = RichLog(id="log-panel")
        yield log_panel
        yield Footer()

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    async def handle_sse_event(self, event):
        action_type = event["type"]
        agent_id = int(event["agent_id"])

        if agent_id not in self.agents:
            print_log(f"Temporary AgentView is created here for agent {agent_id}")
            self.agents[agent_id] = AgentView(agent_id)

        agent_view = self.agents[agent_id]

        if action_type == "token":
            agent_view.append_token(event["content"])

    async def _sse_single_agent(self, agent_id):
        url = f"{self.sse_url}?id={agent_id}"

        try:
            print_log(f"Starting SSE for Agent {agent_id}: {url}")
            async with httpx.AsyncClient(timeout=None) as client:
                async with aconnect_sse(client, "GET", url) as event_source:
                    async for sse in event_source.aiter_sse():
                        if sse.data:
                            try:
                                event = json.loads(sse.data)
                                await self.handle_sse_event(event)
                            except json.JSONDecodeError:
                                continue
            print_log(f"SSE for Agent {agent_id} completed.")
        except Exception as e:
            print_log(f"❌ SSE Error for {agent_id}: {repr(e)}")

    def sync_global(self):
        url = self.sync_url
        r = httpx.get(url)
        if r.status_code != 200:
            return None
        response = r.json()["agents"]
        return response

    def on_mount(self):
        self.agent_tree.clear()
        self.agent_tree.show_root = False
        agents_list = self.sync_global()
        if not agents_list:
            return

        agent_map = {a["id"]: a for a in agents_list}

        all_child_ids = set()
        for a in agents_list:
            all_child_ids.update(a.get("sub_agent", []))

        root_candidates = [a for a in agents_list if a["id"] not in all_child_ids]

        for root_data in root_candidates:
            self._build_tree_recursive(self.agent_tree.root, root_data, agent_map)

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
            print_log(str(n.data))
            agent_id = n.data["id"]
            if agent_id not in self.agents:
                self.agents[agent_id] = AgentView(agent_id)
            agent_views.append(self.agents[agent_id])
        print_log(f"self.agents: {self.agents}")
        await self.grid.set_widgets(agent_views)

        for view in agent_views[:4]: # 只为当前显示的启动
             self.run_worker(
                self._sse_single_agent(view.agent_id),
                group="sse_workers",
                name=f"sse_{view.agent_id}"
            )

if __name__ == "__main__":
    app = VibeFrontend()
    app.run()
