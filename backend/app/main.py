from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from .config import settings
from .routers import chat_router, conversations_router, models_router, tools_router
from .routers.chat import generation_service
from .services.model_manager import model_manager

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Device: {model_manager.device}")
    print(f"Default model: {settings.default_model}")
    if STATIC_DIR.exists():
        print(f"Serving frontend from {STATIC_DIR}")
    yield
    print("Shutting down...")
    generation_service.shutdown()
    await model_manager.unload_model()
    print("Model unloaded, shutdown complete.")


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
app.include_router(conversations_router)
app.include_router(models_router)
app.include_router(tools_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "current_model": model_manager.current_model_id,
        "device": model_manager.device,
    }


if STATIC_DIR.exists():
    # Mount Vite's hashed assets
    if (STATIC_DIR / "assets").exists():
        app.mount("/chat", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
else:

    @app.get("/")
    async def root():
        return {
            "message": "Self-Hosted Inference API",
            "docs": "/docs",
            "health": "/health",
        }
