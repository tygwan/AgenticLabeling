<p align="center">
  <h1 align="center">AgenticLabeling</h1>
  <p align="center">
    <strong>AI-Powered Automatic Labeling Platform</strong>
  </p>
  <p align="center">
    Single-service MVP for Florence-2 + SAM2 auto-labeling, review, and export
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
  <img src="https://img.shields.io/badge/tests-MVP%20baseline%20passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker">
</p>

---

## Overview

**AgenticLabeling** is currently packaged around a single MVP application that handles image upload, Florence-2 detection, SAM2 segmentation, review, and dataset export in one service. The previous microservices stack is preserved only as a legacy deployment path.

### Key Capabilities

- **Auto-Labeling Pipeline**: Image → Florence-2 Detection → SAM2 Segmentation → Registry
- **Review Workspace**: Browser-based approve/delete flow
- **Dataset Export**: YOLO and COCO zip export
- **Runtime Fallback**: If SAM2 is unavailable, the pipeline survives with box-mask fallback
- **Legacy Stack Preserved**: The old microservices deployment remains available separately

---

## Features

### AI Models In The MVP

| Model | Task | Description |
|-------|------|-------------|
| **Florence-2** | Detection | Open-vocabulary object detection with grounding |
| **SAM2** | Segmentation | Instance segmentation with fine masks |
### Legacy Components Still Available

- **DINOv2 classification**
- **YOLO training**
- **Video preprocessing / tracking**
- **Evaluation / MLflow**
- **Microservice gateway and Streamlit UI**

---

## Architecture

### Default Runtime

| Component | Port | GPU | Description |
|-----------|------|-----|-------------|
| agenticlabeling-mvp | 8090 | ✅ | Upload, detect, segment, review, export |

### Data Flow

```
Image Input
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
│  Local Registry   │ ← SQLite + filesystem assets
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Review │ │Export │
│Approve│ │YOLO/  │
│Delete │ │COCO   │
└───────┘ └───────┘
```

### Legacy Runtime

The previous multi-container stack is preserved in [`docker-compose.legacy.yml`](/home/coffin/dev/AgenticLabeling/docker-compose.legacy.yml). The default [`docker-compose.yml`](/home/coffin/dev/AgenticLabeling/docker-compose.yml) now starts only the MVP app.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Installation

```bash
# Start the default MVP container
docker compose up --build

# Or use the helper script
./scripts/run_mvp_docker.sh

# Check health
curl http://localhost:8090/health
```

### Basic Usage

```bash
# Auto-label an image
curl -X POST "http://localhost:8090/api/pipeline/auto-label" \
  -F "image=@image.jpg" \
  -F "project_id=default-project" \
  -F "classes=person,car,dog"

# Export dataset
curl -X POST "http://localhost:8090/api/export" \
  -F "dataset_name=my_dataset" \
  -F "export_format=yolo" \
  -F "only_validated=true"
```

### Access UIs

- **MVP Home**: http://localhost:8090/
- **Review UI**: http://localhost:8090/review
- **API Health**: http://localhost:8090/health
- **Legacy Stack**: `./scripts/run_legacy_docker.sh`

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and setup guide |
| [Architecture Spec](docs/tech-specs/architecture-spec.md) | Detailed system architecture |
| [Development Progress](docs/progress/development-progress.md) | Project status and roadmap |
| [PRD](docs/prd/agenticlabeling-prd.md) | Product requirements document |
| [Navigability Docs](docs/navigability/system-map.md) | System map, edit map, interface catalog, failure memory |
| [Standards Application](docs/standards/adoption-model.md) | How dev-standards are applied inside this repository |
| [Implementation Memory](docs/standards/implementation-memory-summary.md) | Why major implementation choices were made and what they changed |
| [Project Memory](project_memory/README.md) | Local engineering memory store and usage |

---

## API Examples

### Auto-Labeling

```python
import httpx

# Label an image
with open("image.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8090/api/pipeline/auto-label",
        files={"image": f},
        data={"project_id": "default-project", "classes": "person,car,dog"}
    )
    result = response.json()
    print(f"Detected {result['detections']} objects")
```

### Review Objects

```python
# List objects for a source
response = httpx.get(
    "http://localhost:8090/api/review/objects",
    params={"source_id": "src_1234567890ab"},
)
objects = response.json()["data"]
print(objects[0]["category_name"])
```

### Export Dataset

```python
response = httpx.post(
    "http://localhost:8090/api/export",
    data={
        "dataset_name": "my_dataset",
        "export_format": "yolo",
        "only_validated": "true",
    },
)
export_info = response.json()
print(export_info["download_url"])
```

---

## Testing

```bash
# Run the MVP baseline tests
pytest tests/test_mvp_app.py tests/test_mvp_detector.py tests/test_mvp_segmenter.py tests/test_mvp_e2e.py -q
```

```bash
# Run the project-memory regression test
pytest tests/test_project_memory.py -q
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
