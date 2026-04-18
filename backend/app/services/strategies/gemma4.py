import json
import re
import uuid

from ...schemas.chat import ToolCall, FunctionCall
from .base import ModelStrategy

CALL_PATTERN = re.compile(
    r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL
)
ARG_PATTERN = re.compile(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))')


def _coerce_value(raw: str):
    """Coerce a bare (non-string-delimited) value to int, float, or bool."""
    stripped = raw.strip()
    if not stripped:
        return stripped
    low = stripped.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


class Gemma4Strategy(ModelStrategy):
    """Parses Gemma 4 tool calls.

    Format: <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

    String values are wrapped in <|"|> delimiters instead of quotes.
    Numeric and boolean values appear bare.
    """

    @property
    def name(self) -> str:
        return "gemma4"

    @property
    def skip_special_tokens(self) -> bool:
        # Gemma 4's tool call markers are special tokens — must be preserved.
        return False

    def parse_tool_calls(self, text: str) -> tuple[list[ToolCall] | None, str]:
        matches = CALL_PATTERN.findall(text)
        if not matches:
            return None, text

        calls = []
        for func_name, raw_args in matches:
            arguments = {}
            for key, str_val, bare_val in ARG_PATTERN.findall(raw_args):
                if str_val is not None and str_val != "":
                    arguments[key] = str_val
                else:
                    arguments[key] = _coerce_value(bare_val)

            calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments),
                    ),
                )
            )

        clean = CALL_PATTERN.sub("", text).strip()
        return calls, clean
