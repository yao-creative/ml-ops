FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY app ./app

RUN uv sync --frozen --no-dev

RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["uv", "run", "serve-linear"]
