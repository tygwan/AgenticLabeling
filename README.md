<p align="center">
  <h1 align="center">AgenticLabeling</h1>
  <p align="center">
    <strong>AI-Powered Automatic Labeling Platform</strong>
  </p>
  <p align="center">
    Microservices-based auto-labeling system using Florence-2, SAM2, and DINOv2
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/tests-120%20passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker">
</p>

---

## Overview

**AgenticLabeling** is a comprehensive AI-powered automatic labeling platform built on a microservices architecture. It combines state-of-the-art vision models to provide end-to-end object detection, segmentation, classification, and tracking capabilities.

### Key Capabilities

- **Auto-Labeling Pipeline**: Image → Detection → Segmentation → Classification → Registry
- **Video Processing**: Frame extraction, Re-ID tracking, trajectory visualization
- **Model Training**: YOLO training with MLflow experiment tracking
- **Quality Assurance**: Streamlit-based validation UI with track visualization
- **Dataset Export**: YOLO and COCO format support

---

## Features

### AI Models

| Model | Task | Description |
|-------|------|-------------|
| **Florence-2** | Detection | Open-vocabulary object detection with grounding |
| **SAM2** | Segmentation | Instance segmentation with fine masks |
| **DINOv2** | Classification | Visual embeddings for similarity search |
| **YOLO** | Training | Custom model training and inference |

### Core Features

- **Object Registry**: SQLite + ChromaDB for structured data and vector search
- **Evaluation Agent**: mAP, mAP50-95, Confusion Matrix metrics
- **Re-ID Tracker**: Appearance-based object tracking across frames
- **Embedding Search**: LRU cached similarity search with batch support
- **Track Visualization**: Trajectory and timeline views

<details>
<summary><b>View Feature Diagram</b></summary>

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (8000)                        │
├─────────────────────────────────────────────────────────────────┤
│  /auto-label  │  /detect  │  /segment  │  /train  │  /evaluate  │
└───────┬───────┴─────┬─────┴─────┬──────┴────┬─────┴──────┬──────┘
        │             │           │           │            │
        ▼             ▼           ▼           ▼            ▼
   ┌─────────┐  ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
   │Detection│  │Segment-  │ │Classif- │ │Training │ │Evaluation│
   │ Agent   │  │ation     │ │ication  │ │ Agent   │ │  Agent   │
   │Florence2│  │SAM2      │ │DINOv2   │ │YOLO     │ │mAP/CM    │
   └────┬────┘  └────┬─────┘ └────┬────┘ └────┬────┘ └────┬─────┘
        │            │            │           │           │
        └────────────┴────────────┴───────────┴───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    Object Registry      │
                    │   SQLite + ChromaDB     │
                    └─────────────────────────┘
```

</details>

---

## Architecture

### Services

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| gateway | 8000 | ❌ | API routing and orchestration |
| detection-agent | 8001 | ✅ | Florence-2 object detection |
| segmentation-agent | 8002 | ✅ | SAM2 instance segmentation |
| classification-agent | 8003 | ✅ | DINOv2 embeddings |
| training-agent | 8005 | ✅ | YOLO model training |
| evaluation-agent | 8007 | ✅ | Model evaluation metrics |
| preprocessing-agent | 8008 | ❌ | Video processing & Re-ID tracking |
| object-registry | 8010 | ❌ | SQLite + ChromaDB storage |
| data-manager | 8006 | ❌ | YOLO/COCO dataset export |
| label-studio-lite | 8501 | ❌ | Streamlit validation UI |
| mlflow | 5000 | ❌ | Experiment tracking |

### Data Flow

```
Image/Video Input
       │
       ▼
┌──────────────────┐
│  Preprocessing   │ ← Frame extraction, Video processing
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Detection     │ ← Florence-2 grounding
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Segmentation   │ ← SAM2 instance masks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Classification  │ ← DINOv2 embeddings
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Object Registry │ ← Store objects, tracks, embeddings
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Export │ │ Train │
│YOLO/  │ │ YOLO  │
│COCO   │ │ Model │
└───────┘ └───────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/tygwan/AgenticLabeling.git
cd AgenticLabeling

# Start with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Basic Usage

```bash
# Auto-label an image
curl -X POST "http://localhost:8000/auto-label" \
  -F "file=@image.jpg" \
  -F "prompt=person, car, dog"

# Export dataset
curl -X POST "http://localhost:8000/export" \
  -d "dataset_name=my_dataset" \
  -d "format=yolo"

# Train YOLO model
curl -X POST "http://localhost:8000/train/start" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "data/datasets/my_dataset", "epochs": 100}'
```

### Access UIs

- **API Documentation**: http://localhost:8000/docs
- **Validation UI**: http://localhost:8501
- **MLflow Dashboard**: http://localhost:5000

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and setup guide |
| [Architecture Spec](docs/tech-specs/architecture-spec.md) | Detailed system architecture |
| [Development Progress](docs/progress/development-progress.md) | Project status and roadmap |
| [PRD](docs/prd/agenticlabeling-prd.md) | Product requirements document |

---

## API Examples

### Auto-Labeling

```python
import httpx

# Label an image
with open("image.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/auto-label",
        files={"file": f},
        data={"prompt": "person, car, dog", "register": True}
    )
    result = response.json()
    print(f"Detected {len(result['data']['objects'])} objects")
```

### Object Search

```python
# Search similar objects by embedding
response = httpx.post(
    "http://localhost:8000/similar",
    json={"embedding": [...], "top_k": 10}
)
similar_objects = response.json()["data"]
```

### Model Evaluation

```python
# Evaluate detection performance
response = httpx.post(
    "http://localhost:8000/evaluate/detection",
    json={
        "predictions": [...],
        "ground_truth": [...],
        "iou_threshold": 0.5
    }
)
metrics = response.json()["data"]
print(f"mAP: {metrics['mAP']:.3f}")
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html

# Test results
# 120 tests passing (unit + integration)
```

---

## Project Status

```
Progress: ████████████████████░ 95%
```

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Microservices architecture |
| Phase 2 | ✅ Complete | Data management & export |
| Phase 3 | ✅ Complete | Video processing & tracking |
| Phase 4 | ✅ Complete | Training & evaluation |
| Phase 5 | ✅ Complete | Testing (120 tests) |
| Phase 6 | ⏳ Pending | Production deployment |

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) - Microsoft
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta AI
- [DINOv2](https://github.com/facebookresearch/dinov2) - Meta AI
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Ultralytics
- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) - Inspiration for project structure

---

<p align="center">
  Made with AI-assisted development
</p>
