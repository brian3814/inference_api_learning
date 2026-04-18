import logging

from .base import ModelStrategy
from .fallback import FallbackStrategy
from .gemma4 import Gemma4Strategy
from .llama import LlamaStrategy
from .mistral import MistralStrategy
from .qwen import QwenStrategy

logger = logging.getLogger(__name__)

_STRATEGY_MAP: list[tuple[list[str], type[ModelStrategy]]] = [
    (["qwen"], QwenStrategy),
    (["mistral", "mixtral"], MistralStrategy),
    (["llama", "meta-llama"], LlamaStrategy),
    (["gemma-4", "gemma4"], Gemma4Strategy),
]


def detect_strategy(model_id: str, native_tools: bool) -> ModelStrategy:
    """Select the right parsing strategy based on model_id and native tool support.

    If native_tools is False or the model family is unknown, FallbackStrategy is used.
    """
    if native_tools:
        model_id_lower = model_id.lower()
        for patterns, strategy_cls in _STRATEGY_MAP:
            if any(p in model_id_lower for p in patterns):
                strategy = strategy_cls()
                logger.info(f"Selected tool call strategy: {strategy.name} (model: {model_id})")
                return strategy

    strategy = FallbackStrategy()
    logger.info(f"Selected tool call strategy: {strategy.name} (model: {model_id}, native_tools={native_tools})")
    return strategy
