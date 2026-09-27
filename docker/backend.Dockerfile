# Development image for the FastAPI backend. Build context: repository root.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# The venv lives outside /app so bind-mounting the source does not hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src ./src
RUN uv sync --frozen

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "blueprint3d.api:app", "--host", "0.0.0.0", "--port", "8000"]
