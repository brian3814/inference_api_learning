# Self-Hosted Inference API

An OpenAI-compatible API for local model inference using HuggingFace Transformers.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

2. (Optional) Create a `.env` file to customize settings:

```bash
DEFAULT_MODEL=microsoft/DialoGPT-medium
MODELS_DIR=./models
MAX_NEW_TOKENS=256
DEVICE=auto  # Options: cuda, mps, cpu, auto
HOST=0.0.0.0
PORT=8000
```

## Starting the Server

```bash
python run.py
```

The server will start at `http://localhost:8000`. API docs are available at `/docs`.

## Models Directory

The API can load models from HuggingFace Hub or from local directories.

### Loading from HuggingFace

Simply use the model ID when making API calls (e.g., `microsoft/DialoGPT-medium`). The model will be downloaded automatically.

### Loading Local Models

Place models in the `./models` directory (or the path specified by `MODELS_DIR`). Each model should be in its own subdirectory containing the required HuggingFace model files.

Example structure:
```
models/
  dialogpt-small/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    generation_config.json
```

To use a local model, reference it by its folder name (e.g., `dialogpt-small`).

### Downloading Models

You can download models from HuggingFace using the `huggingface-cli`:

```bash
pip install huggingface_hub
huggingface-cli download microsoft/DialoGPT-small --local-dir ./models/dialogpt-small
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check and current model status
- `GET /docs` - Interactive API documentation
