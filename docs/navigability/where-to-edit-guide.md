# AgenticLabeling Where-To-Edit Guide

Use this guide when the task is known but the edit location is not.

## If the task is product-facing

- upload or landing page behavior: edit `mvp_app/main.py`
- review page layout or overlay rendering: edit `mvp_app/main.py`
- detection output normalization or class handling: edit `mvp_app/detector.py`
- segmentation behavior or SAM3 fallback logic: edit `mvp_app/segmenter.py`
- source/object write paths and review state: edit `mvp_app/registry.py`
- export file handling or asset storage behavior: edit `mvp_app/storage.py`

## If the task is operational

- default Docker runtime: edit `docker-compose.yml` and `Dockerfile.mvp`
- legacy stack runtime: edit `docker-compose.legacy.yml`
- local startup behavior: edit `scripts/run_mvp.sh`

## If the task is about repository understanding

- architecture overview: edit `docs/tech-specs/architecture-spec.md`
- active refactor plan: edit `docs/plans/2026-04-20-mvp-refactor-plan.md`
- UI/UX guidance: edit `docs/plans/2026-04-02-mvp-ui-ux-guideline.md`
- module discovery and edit scoping: edit files under `docs/navigability/`
- engineering memory capture and lookup: edit files under `project_memory/`

## If the task is about historical behavior

- inspect `services/` only if the MVP path does not already own the behavior
- treat `services/label-studio-lite` and `services/object-registry` as reference implementations unless the task explicitly targets legacy behavior

