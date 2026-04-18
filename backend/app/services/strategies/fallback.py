import re

from ...schemas.chat import ToolCall
from .base import ModelStrategy, parse_json_call

PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class FallbackStrategy(ModelStrategy):
    """Parses the <tool_call>{json}</tool_call> format injected via system prompt.

    Used for models that don't have native tool call support. The generation
    service teaches this format through FALLBACK_TOOL_SYSTEM_PROMPT.
    """

    @property
    def name(self) -> str:
        return "fallback"

    def parse_tool_calls(self, text: str) -> tuple[list[ToolCall] | None, str]:
        matches = PATTERN.findall(text)
        if not matches:
            return None, text

        calls = []
        for match in matches:
            call = parse_json_call(match)
            if call:
                calls.append(call)

        if not calls:
            return None, text

        clean = PATTERN.sub("", text).strip()
        return calls, clean
