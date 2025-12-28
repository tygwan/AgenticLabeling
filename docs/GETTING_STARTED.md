# Getting Started - AgenticLabeling

This guide will help you set up and run the AgenticLabeling platform.

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA (8GB VRAM) | NVIDIA RTX 3080+ (12GB+ VRAM) |
| Storage | 50 GB SSD | 200 GB NVMe SSD |
| OS | Ubuntu 20.04+ / Windows 10+ / macOS 12+ | Ubuntu 22.04 |

### Software Requirements

- **Python**: 3.10 - 3.12
- **Docker**: 20.10+ with Docker Compose v2
- **NVIDIA Driver**: 525+ (for GPU support)
- **CUDA**: 11.8+ (for GPU support)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tygwan/AgenticLabeling.git
cd AgenticLabeling
```

### 2. Environment Setup

#### Option A: Docker (Recommended)

```bash
# Build and start all services
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs
docker-compose logs -f gateway
```

#### Option B: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install service-specific dependencies
pip install -r services/detection-agent/requirements.txt
pip install -r services/segmentation-agent/requirements.txt
pip install -r services/training-agent/requirements.txt
# ... etc
```

---

## Running the Platform

### Docker Compose (Full Platform)

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d gateway object-registry detection-agent

# Stop all services
docker-compose down
```

### Individual Services (Development)

```bash
# Terminal 1: Object Registry
cd services/object-registry
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload

# Terminal 2: Gateway
cd services/gateway
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Detection Agent
cd services/detection-agent
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| gateway | 8000 | API Gateway (main entry point) |
| detection-agent | 8001 | Florence-2 object detection |
| segmentation-agent | 8002 | SAM2 instance segmentation |
| classification-agent | 8003 | DINOv2 embedding & classification |
| labeling-agent | 8004 | Legacy label management |
| training-agent | 8005 | YOLO model training |
| data-manager | 8006 | Dataset export (YOLO/COCO) |
| evaluation-agent | 8007 | Model evaluation metrics |
| preprocessing-agent | 8008 | Video processing & tracking |
| object-registry | 8010 | Object database & search |
| label-studio-lite | 8501 | Validation UI (Streamlit) |
| mlflow | 5000 | Experiment tracking |
| redis | 6379 | Cache & job queue |

---

## Quick Start Guide

### 1. Health Check

```bash
# Check gateway health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "services": {...}}
```

### 2. Auto-Labeling Example

```bash
# Single image labeling
curl -X POST "http://localhost:8000/auto-label" \
  -F "file=@image.jpg" \
  -F "prompt=person, car, dog"

# Response includes detected objects with bounding boxes
```

### 3. Access UI

Open your browser and navigate to:
- **Validation UI**: http://localhost:8501
- **MLflow Dashboard**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs

---

## GPU Configuration

### NVIDIA Container Toolkit

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
GATEWAY_URL=http://localhost:8000
REGISTRY_URL=http://localhost:8010

# Model Configuration
DETECTION_MODEL=microsoft/Florence-2-large
SEGMENTATION_MODEL=facebook/sam2-hiera-large
CLASSIFICATION_MODEL=facebook/dinov2-giant

# Database
DATABASE_PATH=./data/registry/registry.db
CHROMA_PATH=./data/registry/chroma

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### Docker Compose Profiles

```bash
# CPU-only mode
docker-compose --profile cpu up -d

# GPU mode (default)
docker-compose up -d

# Development mode with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find and kill process using port
lsof -i :8000
kill -9 <PID>
```

#### 2. GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

#### 3. Memory Issues

```bash
# Reduce batch size in config
export BATCH_SIZE=1

# Or use CPU mode for memory-constrained systems
export CUDA_VISIBLE_DEVICES=""
```

#### 4. Model Download Fails

```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/Florence-2-large')"
```

---

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs for interactive API documentation
2. **Label Some Images**: Use the `/auto-label` endpoint with your images
3. **Train a Model**: Export dataset and train with `/train/start`
4. **Evaluate Performance**: Use `/evaluate/detection` to measure mAP

For detailed API documentation, see [API Reference](./API_REFERENCE.md).

For architecture details, see [Architecture Spec](./tech-specs/architecture-spec.md).
