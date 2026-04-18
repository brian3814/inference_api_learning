from ..schemas.chat import ToolCall
from .strategies.base import ModelStrategy


class ToolCallParser:
    def __init__(self, strategy: ModelStrategy):
        self._strategy = strategy

    def parse(self, text: str) -> tuple[list[ToolCall] | None, str]:
        """Parse tool calls from generated text.

        Returns (tool_calls, clean_text) where clean_text has tool-call markup stripped.
        """
        return self._strategy.parse_tool_calls(text)
