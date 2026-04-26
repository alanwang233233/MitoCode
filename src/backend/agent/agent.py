import asyncio
import json
import logging
from typing import List, AsyncGenerator, Any, Dict, Callable

logger = logging.getLogger(__name__)

class _MessageBlock:
    """Base class for all message elements in the conversation history."""
    def __init__(self):
        self.finished = False


class _PlainTextMessageBlock(_MessageBlock):
    """Represents standard text messages from the user or the assistant."""
    def __init__(self, role: str, message: str = ""):
        super().__init__()
        self.role = role
        self.message: str = message
        # Text blocks are considered valid for context building immediately
        self.finished = True

    def append(self, delta: str):
        self.message += delta


class _ToolCallMessageBlock(_MessageBlock):
    """Represents an assistant's request to call a tool/function."""
    def __init__(self, name: str, call_id: str):
        super().__init__()
        self.name = name
        self.id = call_id
        self.arguments_string = ""
        self.parameter: Dict[str, Any] = {}

        self.parse_error: str | None = None

    def append_argument_chunk(self, delta: str):
        self.arguments_string += delta

    def finish(self):
        """Finalizes the tool call by safely parsing the argument string."""
        self.finished = True
        if self.arguments_string:
            try:
                self.parameter = json.loads(self.arguments_string)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {self.name}: {e}")
                self.parse_error = f"JSON parsing failed: {e}"
                self.parameter = {}
        else:
            self.parameter = {}

    def to_standard(self) -> Dict[str, Any]:
        """Converts the internal block to the standard OpenAI tool call format."""
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
    """Stores the execution results returned from local tools."""
    def __init__(self, tool_call_id: str, result: str):
        super().__init__()
        self.role = "tool"
        self.tool_call_id = tool_call_id
        self.content = result
        self.finished = True


class Agent:
    """An asynchronous LLM agent capable of handling streaming text and tool execution."""
    def __init__(self, **settings):
        provider = settings.get("provider", None)
        tools_map = settings.get("tools_map", {})
        tool_schemas = settings.get("tool_schemas", [])
        on_end = settings.get("on_end", None)

        if not provider:
            raise ValueError("Provider is required.")

        self.provider = provider
        self.messages: List[_MessageBlock] = []
        self.tools_map: Dict[str, Callable] = tools_map
        self.tool_schemas = tool_schemas
        self.on_end = on_end

        # Asyncio event to handle cancellation signals safely
        self._cancel_event = asyncio.Event()

    def cancel(self):
        """Signals the agent to halt its current run stream."""
        self._cancel_event.set()

    def append_user_prompt(self, prompt: str):
        """Adds a user prompt to the context and clears any prior cancellation."""
        self._cancel_event.clear() # Reset cancellation state before new run
        self.messages.append(_PlainTextMessageBlock("user", prompt))

    def append_assistant_delta(self, delta: str):
        """Appends streaming text chunks to the latest assistant message."""
        if (not self.messages or
            not isinstance(self.messages[-1], _PlainTextMessageBlock) or
            self.messages[-1].role != "assistant"):
            self.messages.append(_PlainTextMessageBlock("assistant", delta))
        else:
            self.messages[-1].append(delta)

    def build_standard_context(self) -> List[Dict[str, Any]]:
        """Compiles internal message blocks into a provider-compatible JSON list."""
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

        # Catch any trailing tool calls at the end of the history
        if tool_calls:
            context.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            })

        return context

    async def run(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main interaction loop: streams responses, handles tools, and yields events.

        Yields:
            Dict[str, Any]: A dictionary containing the event type and associated data.
            
            Events:
            --------------------
            - system.cancelled:
                Sent if the run is manually cancelled. 
                Format: {"event": "system.cancelled"}
            
            - agent.text: 
                Streaming text delta from the assistant.
                Format: {"event": "agent.text", "delta": str}
            
            - agent.tool_call.header: 
                Initializes a tool call with its ID and name.
                Format: {"event": "agent.tool_call.header", "id": str, "name": str}
            
            - agent.tool_call.delta: 
                Streaming argument chunks for a specific tool call.
                Format: {"event": "agent.tool_call.delta", "id": str, "delta": str}
            
            - system.finished.tool_calls:
                Signal that the LLM has finished proposing tool calls and execution begins.
                Format: {"event": "system.finished.tool_calls"}
            
            - agent.tool_call.execution:
                Indicates a tool is currently being invoked.
                Format: {"event": "agent.tool_call.execution", "name": str, "args": dict}
            
            - agent.tool_call.result:
                The output/return value from a executed tool.
                Format: {"event": "agent.tool_call.result", "id": str, "success": bool, "result": str}
                
            - system.finished.standard:
                Signal that the agent has finished the current conversation turn (stop/length).
                Format: {"event": "system.finished.standard", "reason": str}
        """
        while True:
            if self._cancel_event.is_set():
                yield {"event": "system.cancelled"}
                break

            standard_messages = self.build_standard_context()
            stream = await self.provider.chat(
                tools=self.tool_schemas if self.tool_schemas else None,
                messages=standard_messages
            )

            tool_calls_dict = {}
            active_tool_blocks = {}
            stream_finished_reason = None

            async for chunk in stream:
                # Check for cancellation during the stream loop
                if self._cancel_event.is_set():
                    yield {"event": "system.cancelled"}
                    break

                if not getattr(chunk, 'choices', None):
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # 1. Handle Text
                if getattr(delta, 'content', None):
                    self.append_assistant_delta(delta.content)
                    yield {"event": "agent.text", "delta": delta.content}

                # 2. Handle Tools
                if getattr(delta, 'tool_calls', None):
                    for tool_call_chunk in delta.tool_calls:
                        index = tool_call_chunk.index

                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {"id": "", "name": ""}

                        if getattr(tool_call_chunk, 'id', None):
                            tool_calls_dict[index]["id"] += tool_call_chunk.id

                        if getattr(tool_call_chunk, 'function', None) and getattr(tool_call_chunk.function, 'name', None):
                            tool_calls_dict[index]["name"] += tool_call_chunk.function.name

                            if index not in active_tool_blocks:
                                call_id = tool_calls_dict[index]["id"]
                                call_name = tool_calls_dict[index]["name"]
                                block = _ToolCallMessageBlock(name=call_name, call_id=call_id)
                                self.messages.append(block)
                                active_tool_blocks[index] = block
                                yield {"event": "agent.tool_call.header", "id": call_id, "name": call_name}

                        if getattr(tool_call_chunk, 'function', None) and getattr(tool_call_chunk.function, 'arguments', None):
                            arg_delta = tool_call_chunk.function.arguments
                            if index in active_tool_blocks:
                                active_tool_blocks[index].append_argument_chunk(arg_delta)
                            yield {
                                "event": "agent.tool_call.delta",
                                "id": tool_calls_dict[index].get("id"),
                                "delta": arg_delta
                            }

                if finish_reason:
                    stream_finished_reason = finish_reason

            # Break out entirely if we hit a cancel event while streaming
            if self._cancel_event.is_set():
                break

            # 3. Handle End of Stream
            if stream_finished_reason == "tool_calls":
                yield {"event": "system.finished.tool_calls"}

                # Execute and gather results
                for block in active_tool_blocks.values():
                    block.finish()

                for block in active_tool_blocks.values():
                    func_name = block.name
                    func_args = block.parameter
                    success = False

                    yield {"event": "agent.tool_call.execution", "name": func_name, "args": func_args}

                    if block.parse_error:
                        result_str = f"Invalid JSON Format. {block.parse_error}"
                    elif func_name in self.tools_map:
                        tool_func = self.tools_map[func_name]
                        try:
                            result_raw = tool_func(**func_args)
                            if asyncio.iscoroutine(result_raw):
                                result = await result_raw
                            else:
                                result = result_raw
                            result_str = str(result)
                            success = True
                        except TypeError as e:
                            result_str = f"Invalid argument. {str(e)}"
                        except Exception as e:
                            result_str = f"Error executing tool: {str(e)}"
                    else:
                        result_str = f"Error: Tool '{func_name}' is not registered."

                    self.messages.append(_ToolResultMessageBlock(tool_call_id=block.id, result=result_str))
                    yield {"event": "agent.tool_call.result", "id": block.id, "success": success, "result": result_str}

                # Loop continues back to `while True` to submit the tool execution results back to the LLM
                continue

            # Standard Stops
            if stream_finished_reason in ["stop", "length", "content_filter"]:
                if self.on_end:
                    ret = self.on_end(self, self.messages)
                    if ret and asyncio.iscoroutine(ret):
                        await ret
                
                yield {"event": "system.finished.standard", "reason": stream_finished_reason}
                break
