import json
import re

from ...schemas.chat import ToolCall
from .base import ModelStrategy, dict_to_tool_call

PATTERN = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)


class MistralStrategy(ModelStrategy):
    """Parses Mistral-family tool calls: [TOOL_CALLS] [{"name": ..., "arguments": ...}]"""

    @property
    def name(self) -> str:
        return "mistral"

    def parse_tool_calls(self, text: str) -> tuple[list[ToolCall] | None, str]:
        match = PATTERN.search(text)
        if not match:
            return None, text

        try:
            items = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None, text

        calls = []
        for item in items:
            call = dict_to_tool_call(item)
            if call:
                calls.append(call)

        if not calls:
            return None, text

        clean = text[: match.start()].strip()
        return calls, clean
