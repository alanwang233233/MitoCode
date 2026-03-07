import threading
from concurrent.futures import ThreadPoolExecutor

import _single_agent as single_agent

executor = ThreadPoolExecutor(max_workers=4)
agents: dict[int, single_agent.SingleAgent] = {}
agents_cnt = 0
lock = threading.Lock()


def worker(agent: single_agent.SingleAgent):
    agent.run()


def submit_exist(agent_id: int, message: str):
    with lock:
        agent = agents[agent_id]
    agent.append_str_message(message)
    executor.submit(worker, agent)


def submit_new(system_prompt: str, api_base: str, api_key: str, initial_message: str, tools=None):
    with lock:
        agent = single_agent.SingleAgent(system_prompt, api_base, api_key, tools=tools)
        global agents_cnt
        agents[agents_cnt] = agent
        agents_cnt += 1
    agent.append_str_message(initial_message)
    executor.submit(worker, agent)
