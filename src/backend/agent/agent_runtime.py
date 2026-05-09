import asyncio
import logging
from typing import AsyncGenerator, Dict, Any

from .agent_session import (
    AgentSession,
    _ToolCallMessageBlock,
    _ToolResultMessageBlock,
)

logger = logging.getLogger(__name__)


class AgentRuntime:

    @staticmethod
    async def run(session: AgentSession) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main interaction loop for *session*: streams responses, handles tool
        calls, and yields events.

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
                Signal that the LLM has finished proposing tool calls and
                execution begins.
                Format: {"event": "system.finished.tool_calls"}

            - agent.tool_call.execution:
                Indicates a tool is currently being invoked.
                Format: {"event": "agent.tool_call.execution", "name": str, "args": dict}

            - agent.tool_call.result:
                The output/return value from an executed tool.
                Format: {"event": "agent.tool_call.result", "id": str, "success": bool, "result": str}

            - system.finished.standard:
                Signal that the agent has finished the current conversation
                turn (stop/length).
                Format: {"event": "system.finished.standard", "reason": str}
        """
        while True:
            if session._cancel_event.is_set():
                yield {"event": "system.cancelled"}
                break

            standard_messages = session.build_standard_context()
            stream = await session.provider.chat(
                tools=session.tool_schemas if session.tool_schemas else None,
                messages=standard_messages
            )

            tool_calls_dict = {}
            active_tool_blocks = {}
            stream_finished_reason = None

            async for chunk in stream:
                # Check for cancellation during the stream loop
                if session._cancel_event.is_set():
                    yield {"event": "system.cancelled"}
                    break

                if not getattr(chunk, 'choices', None):
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # 1. Handle Text
                if getattr(delta, 'content', None):
                    session.append_assistant_delta(delta.content)
                    yield {"event": "agent.text", "delta": delta.content}

                # 2. Handle Tools
                if getattr(delta, 'tool_calls', None):
                    for tool_call_chunk in delta.tool_calls:
                        index = tool_call_chunk.index

                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {"id": "", "name": ""}

                        if getattr(tool_call_chunk, 'id', None):
                            tool_calls_dict[index]["id"] += tool_call_chunk.id
                            # Keep the block in sync if it was already created with a
                            # partial ID (ID can be spread across multiple chunks).
                            if index in active_tool_blocks:
                                active_tool_blocks[index].id = tool_calls_dict[index]["id"]

                        if (getattr(tool_call_chunk, 'function', None) and
                                getattr(tool_call_chunk.function, 'name', None)):
                            tool_calls_dict[index]["name"] += tool_call_chunk.function.name

                            if index not in active_tool_blocks:
                                call_id = tool_calls_dict[index]["id"]
                                call_name = tool_calls_dict[index]["name"]
                                block = _ToolCallMessageBlock(name=call_name, call_id=call_id)
                                session.messages.append(block)
                                active_tool_blocks[index] = block
                                yield {"event": "agent.tool_call.header", "id": call_id, "name": call_name}
                            else:
                                # Name can also be spread across multiple chunks.
                                active_tool_blocks[index].name = tool_calls_dict[index]["name"]

                        if (getattr(tool_call_chunk, 'function', None) and
                                getattr(tool_call_chunk.function, 'arguments', None)):
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
            if session._cancel_event.is_set():
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
                    elif func_name in session.tools_map:
                        tool_func = session.tools_map[func_name]
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

                    session.messages.append(
                        _ToolResultMessageBlock(tool_call_id=block.id, result=result_str)
                    )
                    yield {
                        "event": "agent.tool_call.result",
                        "id": block.id,
                        "success": success,
                        "result": result_str,
                    }

                # Loop continues back to `while True` to submit the tool
                # execution results back to the LLM.
                continue

            # Standard Stops
            if stream_finished_reason in ["stop", "length", "content_filter"]:
                if session.on_end:
                    ret = session.on_end(session, session.messages)
                    if ret and asyncio.iscoroutine(ret):
                        await ret

                yield {"event": "system.finished.standard", "reason": stream_finished_reason}
                break

runtime = AgentRuntime.get_instance()
