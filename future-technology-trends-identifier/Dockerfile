# Python slim image, small and stable
FROM python:3.11-slim

# Basic env for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# (Optional) Where HF caches models inside the container
ENV HF_HOME=/root/.cache/huggingface

# System deps (very small), libgomp is needed by PyTorch CPU wheels on Debian/Ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install base Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch from the official index (no CUDA)
# Keep it separate so you don't fight with PyPI default wheels.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Copy your app
COPY . .

# Make sure runtime folders exist (your code also creates them on startup)
RUN mkdir -p storage datasets

# Expose FastAPI default port
EXPOSE 8000

# Start the API (adjust module path if different)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
