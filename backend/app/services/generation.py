import asyncio
import json
import logging
import threading
from typing import AsyncGenerator, Union

import torch
from transformers import TextIteratorStreamer

from ..schemas.chat import ChatMessage
from ..config import settings
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

FALLBACK_TOOL_SYSTEM_PROMPT = """You have access to the following tools. To call a tool, output a tool_call block:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

Available tools:
{tool_descriptions}

When you need information from the web, use the appropriate tool. After receiving tool results, provide your final answer to the user. Only call tools when necessary."""


class GenerationService:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self._shutdown_event = threading.Event()

    def shutdown(self):
        self._shutdown_event.set()

    def _format_messages(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> str:
        if self.manager.tokenizer is None:
            raise RuntimeError("No tokenizer loaded")

        native_tools = self.manager.supports_native_tools() and tools

        message_dicts = []
        for m in messages:
            d: dict = {"role": m.role, "content": m.content}
            if m.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
            message_dicts.append(d)

        if native_tools:
            try:
                return self.manager.tokenizer.apply_chat_template(
                    message_dicts,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.warning("Native tool template failed, falling back")

        # Fallback: inject tool descriptions as system prompt
        if tools:
            tool_descriptions = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']} "
                f"Parameters: {json.dumps(t['function']['parameters'])}"
                for t in tools
            )
            tool_system = FALLBACK_TOOL_SYSTEM_PROMPT.format(
                tool_descriptions=tool_descriptions
            )

            # Convert tool-role messages to text for fallback
            fallback_dicts = []
            for d in message_dicts:
                if d["role"] == "tool":
                    name = d.get("name", "unknown")
                    fallback_dicts.append({
                        "role": "user",
                        "content": f"Tool Result ({name}): {d['content']}",
                    })
                elif d["role"] == "assistant" and d.get("tool_calls"):
                    # Represent the assistant's tool call in the fallback format
                    tc_text = d.get("content", "") or ""
                    for tc in d["tool_calls"]:
                        tc_text += f'\n<tool_call>\n{{"name": "{tc["function"]["name"]}", "arguments": {tc["function"]["arguments"]}}}\n</tool_call>'
                    fallback_dicts.append({
                        "role": "assistant",
                        "content": tc_text.strip(),
                    })
                else:
                    fallback_dicts.append({"role": d["role"], "content": d["content"]})

            # Prepend or merge tool system prompt
            if fallback_dicts and fallback_dicts[0]["role"] == "system":
                fallback_dicts[0]["content"] = tool_system + "\n\n" + fallback_dicts[0]["content"]
            else:
                fallback_dicts.insert(0, {"role": "system", "content": tool_system})

            message_dicts = fallback_dicts

        if hasattr(self.manager.tokenizer, "apply_chat_template"):
            try:
                return self.manager.tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Final fallback: plain text
        prompt_parts = []
        for d in message_dicts:
            role = d["role"]
            content = d["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                name = d.get("name", "tool")
                prompt_parts.append(f"Tool Result ({name}): {content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def _get_generation_kwargs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.manager.tokenizer.pad_token_id,
            "eos_token_id": self.manager.tokenizer.eos_token_id,
        }

        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        return kwargs

    async def generate(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        tools: list[dict] | None = None,
    ) -> Union[tuple[str, int, int], AsyncGenerator[str, None]]:
        if not self.manager.is_loaded():
            raise RuntimeError("No model loaded")

        max_tokens = max_tokens or settings.max_new_tokens
        prompt = self._format_messages(messages, tools=tools)

        inputs = self.manager.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.manager.device)

        prompt_tokens = inputs["input_ids"].shape[1]

        if stream:
            return self._stream_generate(inputs, max_tokens, temperature, top_p, prompt_tokens)
        else:
            return await self._generate(inputs, max_tokens, temperature, top_p, prompt_tokens)

    async def _generate(
        self,
        inputs: dict,
        max_tokens: int,
        temperature: float,
        top_p: float,
        prompt_tokens: int,
    ) -> tuple[str, int, int]:
        generation_kwargs = self._get_generation_kwargs(
            inputs["input_ids"],
            inputs["attention_mask"],
            max_tokens,
            temperature,
            top_p,
        )

        def _run():
            with torch.no_grad():
                return self.manager.model.generate(**generation_kwargs)

        outputs = await asyncio.to_thread(_run)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion_tokens = len(new_tokens)

        generated_text = self.manager.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return generated_text.strip(), prompt_tokens, completion_tokens

    async def _stream_generate(
        self,
        inputs: dict,
        max_tokens: int,
        temperature: float,
        top_p: float,
        prompt_tokens: int,
    ) -> AsyncGenerator[str, None]:
        streamer = TextIteratorStreamer(
            self.manager.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generation_kwargs = self._get_generation_kwargs(
            inputs["input_ids"],
            inputs["attention_mask"],
            max_tokens,
            temperature,
            top_p,
        )
        generation_kwargs["streamer"] = streamer

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _generate_and_drain():
            gen_thread = threading.Thread(
                target=self._run_generation,
                args=(generation_kwargs,),
                daemon=True,
            )
            gen_thread.start()

            try:
                for text in streamer:
                    if self._shutdown_event.is_set():
                        break
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
                    except RuntimeError:
                        break
            except Exception:
                pass

            gen_thread.join(timeout=5)
            try:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except RuntimeError:
                pass

        thread = threading.Thread(target=_generate_and_drain, daemon=True)
        thread.start()

        while True:
            text = await queue.get()
            if text is None:
                break
            if text:
                yield text

        thread.join()

    def _run_generation(self, generation_kwargs: dict):
        with torch.no_grad():
            self.manager.model.generate(**generation_kwargs)
