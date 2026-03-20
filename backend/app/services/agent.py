import json
import logging
from typing import AsyncGenerator

from ..config import settings
from ..schemas.chat import ChatMessage, ToolCall, FunctionCall
from ..tools import tool_registry
from .generation import GenerationService
from .model_manager import ModelManager
from .tool_calling import ToolCallParser

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, generation_service: GenerationService):
        self.generation = generation_service
        self.max_iterations = settings.agent_max_iterations

    @property
    def manager(self) -> ModelManager:
        return self.generation.manager

    async def run(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[str, list[dict], int, int]:
        """Run the agent loop (non-streaming).

        Returns (final_text, tool_history, total_prompt_tokens, total_completion_tokens).
        """
        tool_defs = tool_registry.list_definitions()
        parser = ToolCallParser(self.manager.supports_native_tools())

        working_messages = list(messages)
        tool_history: list[dict] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for iteration in range(self.max_iterations):
            generated_text, prompt_tokens, completion_tokens = await self.generation.generate(
                messages=working_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                tools=tool_defs,
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            tool_calls, clean_text = parser.parse(generated_text)

            if not tool_calls:
                return clean_text, tool_history, total_prompt_tokens, total_completion_tokens

            # Append assistant message with tool calls
            assistant_msg = ChatMessage(
                role="assistant",
                content=clean_text or "",
                tool_calls=tool_calls,
            )
            working_messages.append(assistant_msg)

            # Execute each tool call and append results
            for tc in tool_calls:
                tool_history_entry = {
                    "tool_call": {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }

                result = await self._execute_tool_call(tc)
                tool_history_entry["result"] = result
                tool_history.append(tool_history_entry)

                tool_msg = ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    name=tc.function.name,
                )
                working_messages.append(tool_msg)

            logger.info(f"Agent iteration {iteration + 1}: executed {len(tool_calls)} tool call(s)")

        # Max iterations reached — do one final generation without tools
        final_text, prompt_tokens, completion_tokens = await self.generation.generate(
            messages=working_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            tools=None,
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        return final_text, tool_history, total_prompt_tokens, total_completion_tokens

    async def run_streaming(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> AsyncGenerator[dict, None]:
        """Run agent loop, yielding SSE-compatible events.

        Yields dicts with either:
          {"type": "tool_activity", "data": {...}}
          {"type": "content_delta", "data": str}
          {"type": "done"}
        """
        tool_defs = tool_registry.list_definitions()
        parser = ToolCallParser(self.manager.supports_native_tools())

        working_messages = list(messages)

        for iteration in range(self.max_iterations):
            # Intermediate turns: non-streaming to check for tool calls
            generated_text, prompt_tokens, completion_tokens = await self.generation.generate(
                messages=working_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                tools=tool_defs,
            )

            tool_calls, clean_text = parser.parse(generated_text)

            if not tool_calls:
                # Final turn — re-generate with streaming for the response
                # But we already have the text, so just yield it as deltas
                # to avoid double generation, yield the already-generated text
                if clean_text:
                    yield {"type": "content_delta", "data": clean_text}
                yield {"type": "done"}
                return

            # Append assistant message with tool calls
            assistant_msg = ChatMessage(
                role="assistant",
                content=clean_text or "",
                tool_calls=tool_calls,
            )
            working_messages.append(assistant_msg)

            # Execute tools, yielding activity events
            for tc in tool_calls:
                yield {
                    "type": "tool_activity",
                    "data": {
                        "type": "tool_call",
                        "name": tc.function.name,
                        "arguments": _safe_parse_json(tc.function.arguments),
                    },
                }

                result = await self._execute_tool_call(tc)

                yield {
                    "type": "tool_activity",
                    "data": {
                        "type": "tool_result",
                        "name": tc.function.name,
                        "content": result[:500] if result else "",
                    },
                }

                tool_msg = ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    name=tc.function.name,
                )
                working_messages.append(tool_msg)

            logger.info(f"Agent streaming iteration {iteration + 1}: executed {len(tool_calls)} tool call(s)")

        # Max iterations — final generation streamed
        generator = await self.generation.generate(
            messages=working_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            tools=None,
        )

        async for text in generator:
            yield {"type": "content_delta", "data": text}

        yield {"type": "done"}

    async def _execute_tool_call(self, tc: ToolCall) -> str:
        tool = tool_registry.get(tc.function.name)
        if not tool:
            return f"Error: Unknown tool '{tc.function.name}'"

        try:
            arguments = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            return f"Error: Invalid arguments for tool '{tc.function.name}'"

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            logger.error(f"Tool execution failed for {tc.function.name}: {e}")
            return f"Error executing tool: {str(e)}"


def _safe_parse_json(s: str) -> dict | str:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s
