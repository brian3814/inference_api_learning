import json
import re
import uuid

from ...schemas.chat import ToolCall, FunctionCall
from .base import ModelStrategy

PATTERN = re.compile(
    r'\{"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)


class LlamaStrategy(ModelStrategy):
    """Parses Llama 3.1+ tool calls: {"name": "...", "parameters": {...}}"""

    @property
    def name(self) -> str:
        return "llama"

    def parse_tool_calls(self, text: str) -> tuple[list[ToolCall] | None, str]:
        matches = PATTERN.findall(text)
        if not matches:
            return None, text

        calls = []
        for match in matches:
            try:
                data = json.loads(match)
            except json.JSONDecodeError:
                continue
            if "name" in data and "parameters" in data:
                calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function=FunctionCall(
                            name=data["name"],
                            arguments=json.dumps(data["parameters"]),
                        ),
                    )
                )

        if not calls:
            return None, text

        clean = text
        for match in matches:
            clean = clean.replace(match, "")
        return calls, clean.strip()
