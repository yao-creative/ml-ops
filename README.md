# Ray Serve Linear Scaffold

Minimal Ray Serve service that trains a linear model on toy data at startup and serves inference.

## Requirements

- Python 3.10+
- `uv` (https://docs.astral.sh/uv/)

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
uv sync
```

3. Start the service:

```bash
uv run serve-linear
```

The app starts on `http://0.0.0.0:8000` by default.

## Verify It Works

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction example (default model expects 3 features):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1.0,2.0,3.0]}'
```

Batch prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[[1.0,2.0,3.0],[4.0,5.0,6.0]]}'
```

## Configuration

Set these environment variables before starting the app if needed:

- `SERVE_HOST` (default: `0.0.0.0`)
- `SERVE_PORT` (default: `8000`)
- `ARTIFACT_DIR` (default: `artifacts`)
- `TRAIN_RANDOM_SEED` (default: `42`)
- `TRAIN_N_SAMPLES` (default: `300`)
- `TRAIN_N_FEATURES` (default: `3`)
- `TRAIN_NOISE` (default: `0.2`)
- `MODEL_VERSION` (default: `v1`)

Example:

```bash
SERVE_PORT=9000 MODEL_VERSION=v2 uv run serve-linear
```

## Docker Compose

Build and start:

```bash
docker compose up --build -d
```

Check logs:

```bash
docker compose logs -f app
```

Test endpoints:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1.0,2.0,3.0]}'
```

Stop:

```bash
docker compose down
```
