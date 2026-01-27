import threading
from typing import AsyncGenerator, Union

import torch
from transformers import TextIteratorStreamer

from ..schemas.chat import ChatMessage
from ..config import settings
from .model_manager import ModelManager


class GenerationService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def _format_messages(self, messages: list[ChatMessage]) -> str:
        if self.manager.tokenizer is None:
            raise RuntimeError("No tokenizer loaded")

        message_dicts = [{"role": m.role, "content": m.content} for m in messages]

        if hasattr(self.manager.tokenizer, "apply_chat_template"):
            try:
                return self.manager.tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def _get_generation_kwargs(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        kwargs = {
            "input_ids": input_ids,
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
    ) -> Union[tuple[str, int, int], AsyncGenerator[str, None]]:
        if not self.manager.is_loaded():
            raise RuntimeError("No model loaded")

        max_tokens = max_tokens or settings.max_new_tokens
        prompt = self._format_messages(messages)

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
            max_tokens,
            temperature,
            top_p,
        )

        with torch.no_grad():
            outputs = self.manager.model.generate(**generation_kwargs)

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
            max_tokens,
            temperature,
            top_p,
        )
        generation_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=self._run_generation,
            args=(generation_kwargs,),
        )
        thread.start()

        for text in streamer:
            if text:
                yield text

        thread.join()

    def _run_generation(self, generation_kwargs: dict):
        with torch.no_grad():
            self.manager.model.generate(**generation_kwargs)
