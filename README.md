# Self-Hosted Inference API

An OpenAI-compatible API for local model inference with a React chat UI.

## Structure

```
backend/    FastAPI server (uv-managed Python)
frontend/   React + Vite + TypeScript chat UI
build.bat   Windows build script
build.sh    Linux/Mac build script
```

## Development Setup

### Backend

```bash
cd backend
uv sync
uv run python run.py
```

The API runs at `http://localhost:8000`. Docs at `/docs`.

### Frontend (dev mode)

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173` with API calls proxied to the backend.

## Production Build

Build the frontend and bundle it into the backend as static files:

```bash
# Windows
build.bat

# Linux/Mac
./build.sh
```

Then run:

```bash
cd backend
uv run python run.py
```

Open `http://localhost:8000` to use the chat UI.

## Configuration

Create `backend/.env` (see `backend/.env.example`):

```
DEFAULT_MODEL=microsoft/DialoGPT-medium
MODELS_DIR=./models
MAX_NEW_TOKENS=256
DEVICE=auto
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
HOST=0.0.0.0
PORT=8000
```

## API Endpoints

- `GET /health` — Health check and model status
- `GET /v1/models` — List available models
- `POST /v1/models/load` — Load a model (`{"model_id": "..."}`)
- `POST /v1/models/unload` — Unload current model
- `GET /v1/models/status` — Current model status
- `POST /v1/chat/completions` — Chat completions (supports streaming)
- `GET /docs` — Interactive API docs

## Usage Examples

### Load a model

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "microsoft/DialoGPT-medium"}'
```

### Chat (non-streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Chat (streaming)

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": true
  }'
```
