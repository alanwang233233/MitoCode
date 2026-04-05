from typing import Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import random

app = FastAPI()

agents = ["1", "2", "3", "4", "5"]

async def event_stream(agent_id):
    while True:
        await asyncio.sleep(0.01)

        event = {
            "type": random.choice(["token", "tool_call", "tool_result"]),
            "agent_id": agent_id,
        }

        if event["type"] == "token":
            event["content"] = random.choice("hello world ")

        elif event["type"] == "tool_call":
            event["tool"] = "search"
            event["args"] = {"q": "weather"}

        elif event["type"] == "tool_result":
            event["result"] = "sunny"

        yield f"data: {json.dumps(event)}\n\n"


@app.get("/message")
async def sse(id: Union[str, None] = None):
    return StreamingResponse(event_stream(id), media_type="text/event-stream")

@app.get("/sync")
async def sync():
    response = {
        "id": 666,
        "workspace": "test",
        "event": "sync",
        "agents": [
            {
                "id": 1,
                "name": "森国秀",
                "description": "三织统性己队温型明。极见则选上见线我。解道问。出知支议。才节立。响全经经采导。",
                "sub_agent": [
                    3
                ]
            },
            {
                "id": 2,
                "name": "乙雪",
                "description": "关代土术们物年千月派。当问报将工线引方改。再到么者极属义。干石制县向感。",
                "sub_agent": [
                    4, 5
                ]
            },
            {
                "id": 3,
                "name": "瞿梓睿",
                "description": "也少规东么积。子方常。",
                "sub_agent": [
                ]
            },
            {
                "id": 4,
                "name": "瞿梓睿",
                "description": "也少规东么积。子方常。",
                "sub_agent": [
                ]
            },
            {
                "id": 5,
                "name": "瞿梓睿",
                "description": "也少规东么积。子方常。",
                "sub_agent": [
                ]
            }
        ]
    }
    return JSONResponse(status_code=200, content=response)