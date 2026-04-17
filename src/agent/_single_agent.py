import asyncio
import json
from typing import List, AsyncGenerator, Any, Dict, Callable


class _MessageBlock:
    def __init__(self):
        self.finished = False


class _PlainTextMessageBlock(_MessageBlock):
    def __init__(self, role: str, message: str = ""):
        super().__init__()
        self.role = role
        self.message: str = message
        self.finished = True

    def append(self, delta: str):
        self.message += delta


class _ToolCallMessageBlock(_MessageBlock):
    def __init__(self, name: str, call_id: str):
        super().__init__()
        self.name = name
        self.id = call_id
        self.arguments_string = ""
        self.parameter = {}

    def append_argument_chunk(self, delta: str):
        self.arguments_string += delta

    def finish(self):
        self.finished = True
        if self.arguments_string:
            try:
                self.parameter = json.loads(self.arguments_string)
            except json.JSONDecodeError:
                self.parameter = {}

    def to_standard(self):
        if not self.finished:
            return {}
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_string,
            }
        }


class _ToolResultMessageBlock(_MessageBlock):
    """New block to store the execution results of tools"""

    def __init__(self, tool_call_id: str, result: str):
        super().__init__()
        self.role = "tool"
        self.tool_call_id = tool_call_id
        self.content = result
        self.finished = True


class _SingleAgent:
    def __init__(self, **settings):
        provider = settings.get("provider", None)
        tools_map = settings.get("tools_map", {})
        tool_schemas = settings.get("tool_schemas", [])
        on_end = settings.get("on_end", lambda agent, full_message: None)

        if not provider:
            raise ValueError("Provider is required.")

        self.provider = provider
        self.messages: List[_MessageBlock] = []
        self.tools_map: Dict[str, Callable] = tools_map
        self.tool_schemas = tool_schemas
        self.on_end = on_end

    def cancel(self):
        # TODO: Implementation for cancellation (e.g., setting a threading/asyncio Event)
        pass

    def append_user_prompt(self, prompt: str):
        self.cancel()
        self.messages.append(_PlainTextMessageBlock("user", prompt))

    def append_assistant_delta(self, delta: str):
        if (not self.messages or not isinstance(self.messages[-1], _PlainTextMessageBlock)
                or not self.messages[-1].role == "assistant"):  # type: ignore
            self.messages.append(_PlainTextMessageBlock("assistant", delta))
        else:
            self.messages[-1].append(delta)  # type: ignore

    def build_standard_context(self):
        context = []
        tool_calls = []

        for block in self.messages:
            if not block.finished:
                continue

            if isinstance(block, _PlainTextMessageBlock):
                if tool_calls:
                    context.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    })
                    tool_calls = []
                context.append({
                    "role": block.role,
                    "content": block.message
                })
            elif isinstance(block, _ToolCallMessageBlock):
                tool_calls.append(block.to_standard())
            elif isinstance(block, _ToolResultMessageBlock):
                if tool_calls:
                    context.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    })
                    tool_calls = []
                context.append({
                    "role": block.role,
                    "tool_call_id": block.tool_call_id,
                    "content": block.content
                })

        if tool_calls:
            context.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            })

        return context

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            standard_messages = self.build_standard_context()

            stream = await self.provider.chat(
                tools=self.tool_schemas if self.tool_schemas else None,
                messages=standard_messages
            )

            tool_calls_dict = {}
            active_tool_blocks = {}
            stream_finished_reason = None

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                if delta.content:
                    self.append_assistant_delta(delta.content)
                    yield {"event": "text", "delta": delta.content}

                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        index = tool_call_chunk.index

                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {"id": "", "name": ""}

                        if tool_call_chunk.id:
                            tool_calls_dict[index]["id"] += tool_call_chunk.id
                        if tool_call_chunk.function and tool_call_chunk.function.name:
                            tool_calls_dict[index]["name"] += tool_call_chunk.function.name

                            call_id = tool_calls_dict[index]["id"]
                            call_name = tool_calls_dict[index]["name"]

                            block = _ToolCallMessageBlock(name=call_name, call_id=call_id)
                            self.messages.append(block)
                            active_tool_blocks[index] = block

                            yield {"event": "tool_header", "id": call_id, "name": call_name}

                        if tool_call_chunk.function and tool_call_chunk.function.arguments:
                            arg_delta = tool_call_chunk.function.arguments
                            if index in active_tool_blocks:
                                active_tool_blocks[index].append_argument_chunk(arg_delta)
                            yield {"event": "tool_argument_delta", "id": tool_calls_dict[index].get("id"),
                                   "delta": arg_delta}

                if finish_reason:
                    stream_finished_reason = finish_reason

            if stream_finished_reason == "tool_calls":
                yield {"event": "system", "content": "\n--- Stream finished, processing final tool calls ---"}

                for block in active_tool_blocks.values():
                    block.finish()

                for block in active_tool_blocks.values():
                    func_name = block.name
                    func_args = block.parameter

                    yield {"event": "tool_execution", "name": func_name, "args": func_args}

                    if func_name in self.tools_map:
                        tool_func = self.tools_map[func_name]
                        try:
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(**func_args)
                            else:
                                result = tool_func(**func_args)
                            result_str = str(result)
                        except Exception as e:
                            result_str = f"Error executing tool: {str(e)}"
                    else:
                        result_str = f"Error: Tool '{func_name}' is not registered."

                    self.messages.append(_ToolResultMessageBlock(tool_call_id=block.id, result=result_str))
                    yield {"event": "tool_result", "id": block.id, "result": result_str}

                continue
            if stream_finished_reason in ["stop", "length", "content_filter"]:
                if self.on_end:
                    self.on_end(self, self.messages)
                break
