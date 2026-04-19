import gc
import os
from pathlib import Path
from typing import Optional

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger(__name__)

from ..config import settings
from .strategies import ModelStrategy, detect_strategy


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_id: Optional[str] = None
        self.device = self._detect_device()
        self._native_tool_support: bool = False
        self._strategy: ModelStrategy | None = None
        self._is_multimodal: bool = False

    def _detect_device(self) -> str:
        if settings.device != "auto":
            return settings.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _is_local_model(self, model_id: str) -> bool:
        if os.path.exists(model_id):
            return True
        local_path = Path(settings.models_dir) / model_id
        return local_path.exists()

    def _resolve_dtype(self) -> torch.dtype:
        dt = settings.torch_dtype
        if dt == "float16":
            return torch.float16
        if dt == "bfloat16":
            return torch.bfloat16
        if dt == "float32":
            return torch.float32
        # auto
        if self.device == "cpu":
            return torch.float32
        if self._is_multimodal:
            return torch.bfloat16
        return torch.float16

    @property
    def _project_cache_dir(self) -> str:
        """Project-local cache directory for model downloads."""
        path = Path(settings.models_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _resolve_cache_dir(self, model_id: str) -> str:
        """Find which cache dir holds the model, or pick one for download.

        Search order:
          1. HuggingFace default cache (~/.cache/huggingface/hub)
          2. Project cache (models_dir)
          3. Fall back to project cache for new downloads
        """
        cache_subdir = "models--" + model_id.replace("/", "--")

        # 1. Check HF default cache
        try:
            from huggingface_hub import constants
            hf_cache = Path(constants.HF_HUB_CACHE)
            if (hf_cache / cache_subdir).is_dir():
                logger.info(f"Found {model_id} in HF cache: {hf_cache}")
                return str(hf_cache)
        except Exception:
            pass

        # 2. Check project cache
        project_cache = Path(self._project_cache_dir)
        if (project_cache / cache_subdir).is_dir():
            logger.info(f"Found {model_id} in project cache: {project_cache}")
            return str(project_cache)

        # 3. Not cached anywhere — download to project cache
        logger.info(f"Model {model_id} not cached, will download to: {project_cache}")
        return str(project_cache)

    def _is_pre_quantized(self, model_path: str, cache_dir: str) -> bool:
        """Check if a model already has a native quantization config."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
            return getattr(config, "quantization_config", None) is not None
        except Exception:
            return False

    def _get_model_path(self, model_id: str) -> str:
        if os.path.exists(model_id):
            return model_id
        local_path = Path(settings.models_dir) / model_id
        if local_path.exists():
            return str(local_path)
        return model_id

    async def load_model(self, model_id: str) -> dict:
        if self.current_model_id == model_id and self.model is not None:
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "device": self.device,
            }

        await self.unload_model()

        model_path = self._get_model_path(model_id)
        cache_dir = self._resolve_cache_dir(model_id)

        try:
            # Detect multimodal: try loading a processor first
            try:
                self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
                self._is_multimodal = (
                    hasattr(self.processor, "image_processor")
                    and self.processor.image_processor is not None
                )
                logger.info(
                    f"Processor loaded for {model_id}: "
                    f"multimodal={self._is_multimodal}, "
                    f"type={type(self.processor).__name__}, "
                    f"has image_processor={hasattr(self.processor, 'image_processor')}"
                )
            except Exception as e:
                logger.warning(f"Failed to load processor for {model_id}: {e}")
                self.processor = None
                self._is_multimodal = False

            if self.processor and hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "torch_dtype": self._resolve_dtype(),
            }

            # Skip BitsAndBytes if the model is already quantized natively
            pre_quantized = self._is_pre_quantized(model_path, cache_dir)
            if pre_quantized:
                logger.info(f"Model {model_id} is already quantized, skipping BitsAndBytes")

            if not pre_quantized and settings.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                model_kwargs["device_map"] = "auto"
            elif not pre_quantized and settings.load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                model_kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                model_kwargs["device_map"] = "auto"

            if model_kwargs.get("device_map") == "auto":
                offload_dir = Path(settings.models_dir) / ".offload"
                offload_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs["offload_folder"] = str(offload_dir)

            if self._is_multimodal:
                from transformers import AutoModelForImageTextToText
                model_cls = AutoModelForImageTextToText
            else:
                model_cls = AutoModelForCausalLM

            try:
                self.model = model_cls.from_pretrained(
                    model_path, cache_dir=cache_dir, **model_kwargs,
                )
            except TypeError as e:
                if "_is_hf_initialized" in str(e):
                    logger.warning("Patching bitsandbytes Params4bit compatibility issue")
                    import bitsandbytes as bnb
                    _orig_new = bnb.nn.Params4bit.__new__
                    def _patched_new(cls, *args, **kwargs):
                        kwargs.pop("_is_hf_initialized", None)
                        return _orig_new(cls, *args, **kwargs)
                    bnb.nn.Params4bit.__new__ = _patched_new
                    self.model = model_cls.from_pretrained(
                        model_path, cache_dir=cache_dir, **model_kwargs,
                    )
                else:
                    raise

            if self._is_multimodal:
                logger.info(f"Loaded multimodal model: {model_id}")

            if not (settings.load_in_4bit or settings.load_in_8bit or self.device == "cuda"):
                self.model = self.model.to(self.device)

            self.model.eval()
            self.current_model_id = model_id
            self._native_tool_support = self._probe_native_tools()
            self._strategy = detect_strategy(model_id, self._native_tool_support)

            return {
                "status": "loaded",
                "model_id": model_id,
                "device": self.device,
            }

        except Exception as e:
            await self.unload_model()
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    async def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.current_model_id = None
        self._native_tool_support = False
        self._strategy = None
        self._is_multimodal = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _probe_native_tools(self) -> bool:
        """Check if the loaded model's chat template supports tool definitions."""
        if self.tokenizer is None:
            return False
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return False

        dummy_tools = [
            {
                "type": "function",
                "function": {
                    "name": "_probe",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        dummy_messages = [{"role": "user", "content": "hi"}]

        try:
            self.tokenizer.apply_chat_template(
                dummy_messages,
                tools=dummy_tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def supports_native_tools(self) -> bool:
        return self._native_tool_support

    @property
    def strategy(self) -> ModelStrategy:
        if self._strategy is None:
            raise RuntimeError("No model loaded")
        return self._strategy

    @property
    def is_multimodal(self) -> bool:
        return self._is_multimodal

    def list_cached_models(self) -> list[str]:
        """List model IDs cached in HF default cache or project cache."""
        seen = set()
        cached = []

        dirs_to_scan = [Path(self._project_cache_dir)]
        try:
            from huggingface_hub import constants
            hf_cache = Path(constants.HF_HUB_CACHE)
            if hf_cache.exists():
                dirs_to_scan.insert(0, hf_cache)
        except Exception:
            pass

        for cache_path in dirs_to_scan:
            if not cache_path.exists():
                continue
            for item in cache_path.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    model_id = item.name.removeprefix("models--").replace("--", "/", 1)
                    if model_id not in seen:
                        seen.add(model_id)
                        cached.append(model_id)
        return sorted(cached)

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded(),
            "model_id": self.current_model_id,
            "device": self.device,
            "multimodal": self._is_multimodal,
        }


model_manager = ModelManager()
