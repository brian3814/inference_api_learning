# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monorepo for a self-hosted OpenAI-compatible inference API with a React chat UI. Backend serves HuggingFace models locally with streaming support. Frontend is bundled into the backend as static files for production.

## Commands

### Backend (Python, managed with uv)
```bash
cd backend
uv sync                  # install dependencies
uv run python run.py     # start server at http://localhost:8000
```

### Frontend (React + Vite + TypeScript)
```bash
cd frontend
npm install
npm run dev              # dev server at http://localhost:5173, proxies API to backend
npm run build            # production build to dist/
npm run lint             # eslint
npx tsc --noEmit         # type-check without emitting
```

### Production Build
```bash
./build.sh               # or build.bat on Windows
# Builds frontend, copies dist/ to backend/static/
# Then: cd backend && uv run python run.py
```

### Quick Start (loads root .env, starts backend)
```bash
./start.sh               # or start.bat on Windows
```

## Architecture

**Backend** (`backend/app/`): FastAPI with a service layer pattern.
- `main.py` — App setup, lifespan hooks, static file serving (SPA catch-all when `backend/static/` exists)
- `config.py` — Pydantic-settings config, reads from env vars and `backend/.env`
- `routers/chat.py` — OpenAI-compatible `/v1/chat/completions` with SSE streaming
- `routers/models.py` — Model load/unload/status endpoints under `/v1/models/`
- `services/model_manager.py` — Singleton `model_manager` handling PyTorch model lifecycle, device detection, quantization
- `services/generation.py` — Text generation; uses `asyncio.to_thread()` for blocking inference, `threading.Thread` + `asyncio.Queue` for streaming

**Frontend** (`frontend/src/`): React 19 + TypeScript.
- `api.ts` — API client with `loadModel()`, `getModelStatus()`, `streamChat()` (manual SSE parsing via ReadableStream)
- `components/ModelPanel.tsx` — Model loading controls
- `components/ChatPanel.tsx` — Chat UI with streaming responses

**Static serving**: When `backend/static/` exists, `main.py` mounts `/assets` and adds a catch-all route for SPA fallback. API routes (`/v1/*`, `/health`, `/docs`) are registered first and take priority.

## Configuration

Environment variables configured via `.env` files (see `backend/.env.example`). Priority: OS env vars > `.env` file > Settings defaults. The root start scripts source a root `.env` before launching, so vars can live at either level.

## Key Dependencies

- **PyTorch**: Installed from `https://download.pytorch.org/whl/cu130` (CUDA 13.0). The uv index is configured as `explicit = true` so only `torch` resolves from it.
- **bitsandbytes**: Required for 4-bit/8-bit quantization (CUDA only).

## No Tests

There is currently no test framework or test suite configured.
