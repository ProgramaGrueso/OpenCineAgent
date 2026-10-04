FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Copy application source
COPY opencine/ ./opencine/
COPY config/ ./config/
COPY examples/ ./examples/
COPY main.py .

ENTRYPOINT ["python", "main.py"]
CMD ["doctor"]
