# AgenticLabeling Interface Catalog

This document records the stable interfaces that other modules, tools, or humans should rely on before they rely on internal implementation details.

## HTTP interfaces

| Interface | Kind | Purpose |
|---|---|---|
| `GET /` | HTML route | Upload entrypoint |
| `POST /upload` | HTML form route | Upload image and run the pipeline |
| `GET /review` | HTML route | Review workspace |
| `POST /review/objects/{object_id}/approve` | HTML form route | Mark an object as approved |
| `POST /review/objects/{object_id}/delete` | HTML form route | Delete an object |
| `POST /review/export` | HTML form route | Export validated dataset |
| `GET /health` | JSON route | Runtime health, counts, segmentation backend |
| `GET /api/assets/{source_id}` | Image route | Raw uploaded source image |
| `GET /api/assets/{source_id}/bbox-overlay` | Image route | BBox overlay rendering |
| `GET /api/assets/{source_id}/overlay` | Image route | Segmentation overlay rendering |
| `GET /api/masks/{object_id}` | Image route | Individual mask rendering |
| `GET /api/export/download/{filename}` | File route | Export download |

## Configuration interfaces

The main MVP configuration contract lives in `mvp_app/config.py`.

Important environment variables:

- `DATA_DIR`
- `SQLITE_PATH`
- `ASSETS_DIR`
- `MASKS_DIR`
- `EXPORTS_DIR`
- `FAKE_MODELS`
- `FLORENCE_MODEL_ID`
- `SAM3_CHECKPOINT`
- `SAM3_VERSION`

Project memory configuration:

- `PROJECT_MEMORY_DB_PATH`

## File and data interfaces

| Interface | Purpose |
|---|---|
| `data/mvp/sqlite/mvp.db` | MVP runtime sqlite database |
| `data/mvp/assets/` | Uploaded source images |
| `data/mvp/masks/` | Segmentation masks |
| `data/mvp/exports/` | Exported datasets |
| `data/project-memory/project-memory.db` | Engineering memory store |

## CLI interfaces

| Command | Purpose |
|---|---|
| `./scripts/run_mvp.sh` | Start local MVP app |
| `./scripts/run_mvp_docker.sh` | Start Dockerized MVP app |
| `./scripts/run_legacy_docker.sh` | Start legacy microservice stack |
| `python3 -m project_memory.memory_store ...` | Project memory init/add/query/link |

