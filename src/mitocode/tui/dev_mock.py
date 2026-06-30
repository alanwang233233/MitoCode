import asyncio
import json

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

# 模拟 Agent 结构数据
AGENTS_DATA = {
    "event": "sync",
    "agents": [
        {
            "id": 1,
            "name": "Main Orchestrator",
            "description": "The boss agent",
            "sub_agent": [2, 3]
        },
        {
            "id": 2,
            "name": "Coder Agent",
            "description": "Writes Python code",
            "sub_agent": []
        },
        {
            "id": 3,
            "name": "Reviewer Agent",
            "description": "Reviews the code",
            "sub_agent": []
        }
    ]
}


@app.get("/sync")
async def sync_agents():
    """模拟全局状态同步"""

    async def event_generator():
        # 发送初始树结构
        yield json.dumps(AGENTS_DATA)
        while True:
            # 每隔30秒维持一次心跳或发送伪增量更新
            await asyncio.sleep(30)
            yield json.dumps({"event": "ping"})

    return EventSourceResponse(event_generator())


@app.get("/message")
async def get_messages(id: int):
    """模拟流式消息和工具调用"""

    async def event_generator():
        # 1. 发送普通 Token
        intro_text = f"Hello! I am Agent {id}. I will now create a Python script for you.\n\n"
        for char in intro_text:
            yield json.dumps({
                "type": "token",
                "agent_id": id,
                "content": char
            })
            await asyncio.sleep(0.02)

        # 2. 发送工具调用 Header
        yield json.dumps({
            "type": "tool_call",
            "agent_id": id,
            "action": "file.write",
            "description": "Generating a Fibonacci script",
            "additional": "fibonacci.py"
        })
        await asyncio.sleep(0.5)

        # 3. 流式发送工具调用 Body (代码内容)
        code_content = (
            "def fib(n):\n"
            "    if n <= 1: return n\n"
            "    return fib(n-1) + fib(n-2)\n\n"
            "print(fib(10))"
        )
        for chunk in [code_content[i:i + 3] for i in range(0, len(code_content), 3)]:
            yield json.dumps({
                "type": "tool_call",
                "agent_id": id,
                "content": chunk
            })
            await asyncio.sleep(0.05)

        # 4. 发送工具执行结果
        await asyncio.sleep(0.8)
        yield json.dumps({
            "type": "tool_result",
            "agent_id": id,
            "result": "Success: File written to /tmp/fibonacci.py"
        })

        # 5. 最后一点总结文本
        outro = "\n\nThe tool has finished executing. Is there anything else you need?"
        for char in outro:
            yield json.dumps({
                "type": "token",
                "agent_id": id,
                "content": char
            })
            await asyncio.sleep(0.02)

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
