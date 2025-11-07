# Python slim image, small and stable
FROM python:3.11-slim

# Basic env for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Workdir
WORKDIR /app

# Add Portainer team label
LABEL io.portainer.accesscontrol.teams="skillab-all"

# Install base Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8878

# Start the API (adjust module path if different)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8878"]
