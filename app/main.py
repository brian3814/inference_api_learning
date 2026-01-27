from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import chat_router, models_router
from .services.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Device: {model_manager.device}")
    print(f"Default model: {settings.default_model}")
    yield
    await model_manager.unload_model()
    print("Model unloaded, shutting down.")


app = FastAPI(
    title="Self-Hosted Inference API",
    description="OpenAI-compatible API for local model inference",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(models_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "current_model": model_manager.current_model_id,
        "device": model_manager.device,
    }


@app.get("/")
async def root():
    return {
        "message": "Self-Hosted Inference API",
        "docs": "/docs",
        "health": "/health",
    }
