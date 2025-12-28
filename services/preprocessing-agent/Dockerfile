FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Environment
ENV GATEWAY_URL=http://gateway:8000
ENV REGISTRY_URL=http://object-registry:8010
ENV DATA_DIR=/data/preprocessing
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8008

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
