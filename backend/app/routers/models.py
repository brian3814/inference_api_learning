from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..services.model_manager import model_manager

router = APIRouter(tags=["Models"])


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class LoadModelRequest(BaseModel):
    model_id: str


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    models = []
    seen = set()

    if model_manager.current_model_id:
        models.append(
            ModelInfo(
                id=model_manager.current_model_id,
                owned_by="loaded",
            )
        )
        seen.add(model_manager.current_model_id)

    # Cached HuggingFace models in models_dir
    for model_id in model_manager.list_cached_models():
        if model_id not in seen:
            models.append(ModelInfo(id=model_id, owned_by="cached"))
            seen.add(model_id)

    # Plain local directories in models_dir
    models_dir = Path(settings.models_dir)
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith("models--") and item.name != ".gitkeep":
                if item.name not in seen:
                    models.append(ModelInfo(id=item.name))
                    seen.add(item.name)

    return ModelListResponse(data=models)


@router.post("/v1/models/load")
async def load_model(request: LoadModelRequest):
    try:
        result = await model_manager.load_model(request.model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/models/unload")
async def unload_model():
    if not model_manager.is_loaded():
        return {"status": "no_model_loaded"}

    previous_model = model_manager.current_model_id
    await model_manager.unload_model()

    return {
        "status": "unloaded",
        "previous_model": previous_model,
    }


@router.get("/v1/models/status")
async def model_status():
    return model_manager.get_status()
