# Use a specialized uv image for faster dependency management
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies (ffmpeg for pydub/whisper, libmagic for python-magic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# We use --frozen to ensure we use the lockfile exactly
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.13-slim-bookworm

WORKDIR /app

# Install system dependencies in the final image as well
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
