import json
import logging
import re
import uuid

from ..schemas.chat import ToolCall, FunctionCall

logger = logging.getLogger(__name__)


class ToolCallParser:
    def __init__(self, native_support: bool):
        self.native_support = native_support

    def parse(self, text: str) -> tuple[list[ToolCall] | None, str]:
        """Parse tool calls from generated text.

        Returns (tool_calls, clean_text) where clean_text has tool-call markup stripped.
        """
        if self.native_support:
            result = self._parse_native(text)
            if result is not None:
                return result

        return self._parse_fallback(text)

    def _parse_native(self, text: str) -> tuple[list[ToolCall] | None, str] | None:
        """Try native model tool call patterns."""

        # Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        qwen_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        qwen_matches = re.findall(qwen_pattern, text, re.DOTALL)
        if qwen_matches:
            calls = []
            for match in qwen_matches:
                call = self._parse_json_call(match)
                if call:
                    calls.append(call)
            if calls:
                clean = re.sub(qwen_pattern, "", text, flags=re.DOTALL).strip()
                return calls, clean

        # Mistral: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
        mistral_pattern = r"\[TOOL_CALLS\]\s*(\[.*?\])"
        mistral_match = re.search(mistral_pattern, text, re.DOTALL)
        if mistral_match:
            try:
                items = json.loads(mistral_match.group(1))
                calls = []
                for item in items:
                    call = self._dict_to_tool_call(item)
                    if call:
                        calls.append(call)
                if calls:
                    clean = text[:mistral_match.start()].strip()
                    return calls, clean
            except json.JSONDecodeError:
                pass

        # Llama 3.1+: {"name": "...", "parameters": {...}} at end of output
        # Look for JSON object with name and parameters keys
        json_pattern = r'\{"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}\s*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        if json_matches:
            calls = []
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if "name" in data and "parameters" in data:
                        call = ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=data["name"],
                                arguments=json.dumps(data["parameters"]),
                            ),
                        )
                        calls.append(call)
                except json.JSONDecodeError:
                    continue
            if calls:
                clean = text
                for match in json_matches:
                    clean = clean.replace(match, "")
                return calls, clean.strip()

        return None

    def _parse_fallback(self, text: str) -> tuple[list[ToolCall] | None, str]:
        """Parse our injected <tool_call> format."""
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return None, text

        calls = []
        for match in matches:
            call = self._parse_json_call(match)
            if call:
                calls.append(call)

        if not calls:
            return None, text

        clean = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return calls, clean

    def _parse_json_call(self, json_str: str) -> ToolCall | None:
        try:
            data = json.loads(json_str)
            return self._dict_to_tool_call(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call JSON: {json_str[:100]}")
            return None

    def _dict_to_tool_call(self, data: dict) -> ToolCall | None:
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
