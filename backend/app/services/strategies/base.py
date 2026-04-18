import json
import logging
import uuid
from abc import ABC, abstractmethod

from ...schemas.chat import ToolCall, FunctionCall

logger = logging.getLogger(__name__)


class ModelStrategy(ABC):
    """Abstract strategy for parsing tool calls from model output.

    Each model family (Qwen, Mistral, Llama, Gemma 4, etc.) produces tool calls
    in a different text format. Concrete strategies know how to extract them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def skip_special_tokens(self) -> bool:
        """Whether to skip special tokens when decoding model output.

        Most models: True (default). Gemma 4: False, because its tool call
        markers (<|tool_call>, <|"|>, etc.) are special tokens that must be
        preserved for parsing.
        """
        return True

    @abstractmethod
    def parse_tool_calls(self, text: str) -> tuple[list[ToolCall] | None, str]:
        """Parse tool calls from raw model output.

        Returns (tool_calls, clean_text) where clean_text has tool-call
        markup stripped. Returns (None, text) if no tool calls found.
        """
        ...


# ── Shared utilities for JSON-based strategies ──────────────────────────


def parse_json_call(json_str: str) -> ToolCall | None:
    """Parse a JSON string into a ToolCall."""
    try:
        data = json.loads(json_str)
        return dict_to_tool_call(data)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool call JSON: {json_str[:100]}")
        return None


def dict_to_tool_call(data: dict) -> ToolCall | None:
    """Convert a dict with name/arguments/parameters keys into a ToolCall."""
    name = data.get("name")
    if not name:
        return None

    arguments = data.get("arguments", data.get("parameters", {}))
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        function=FunctionCall(name=name, arguments=arguments),
    )
