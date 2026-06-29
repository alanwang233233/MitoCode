import threading
from concurrent.futures import ThreadPoolExecutor
from backend.agent.agent import Agent
from backend.agent import mcp_manager

executor = ThreadPoolExecutor(max_workers=4)
agents: dict[int, Agent] = {}
agents_cnt = 0
lock = threading.Lock()


def submit_exist(agent_id: int, message: str):
    with lock:
        agent = agents[agent_id]
    agent.append_str_message(message)
    return executor.submit(agent.run)


def submit_new(
    prompt: list[dict], globalmcpmanager: mcp_manager.MCPManager, **settings
):
    with lock:
        agent = Agent(prompt, globalmcpmanager, settings=settings)
        global agents_cnt
        agents[agents_cnt] = agent
        agents_cnt += 1
    return executor.submit(agent.run)
