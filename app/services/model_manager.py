import gc
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import settings


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_id: Optional[str] = None
        self.device = self._detect_device()

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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            self.current_model_id = model_id

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

        self.current_model_id = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded(),
            "model_id": self.current_model_id,
            "device": self.device,
        }


model_manager = ModelManager()
