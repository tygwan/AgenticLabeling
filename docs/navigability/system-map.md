# AgenticLabeling System Map

## Current runtime split

AgenticLabeling currently has two structural layers:

1. `mvp_app/`
   - the active runtime path
   - handles upload, Florence-2 detection, SAM3 segmentation, review, approval, and export
2. `services/`
   - legacy microservice path
   - preserved for reference and fallback, but not the default runtime

## Top-level module map

| Scope | Role | Notes |
|---|---|---|
| `mvp_app/` | Active MVP application | Main code path for product behavior |
| `project_memory/` | Local engineering memory store | SQLite-backed project memory and query CLI |
| `scripts/` | Runtime entry helpers | MVP and legacy startup helpers |
| `tests/` | Verification baseline | MVP and project-memory regression checks |
| `docs/` | Product, architecture, planning, and navigability docs | Human and AI discovery index |
| `services/` | Legacy microservices | Kept to preserve prior architecture and selective reuse |
| `data/` | Runtime assets and local state | Images, sqlite, masks, exports, registry, models |
| `vendor/` | Vendored third-party code | Includes SAM3 (Segment Anything Model 3) |

## Active request path

The current primary path is:

`upload -> detect -> segment -> registry -> review -> approve/delete -> export`

The main code entrypoints for that path are:

- `mvp_app/main.py`
- `mvp_app/detector.py`
- `mvp_app/segmenter.py`
- `mvp_app/registry.py`
- `mvp_app/storage.py`

## Key architectural boundary

The MVP monolith is the default runtime boundary. New product-facing behavior should usually land in `mvp_app/` first unless there is a concrete reason to revive or reuse a legacy service.

