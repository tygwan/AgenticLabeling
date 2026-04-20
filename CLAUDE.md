# CLAUDE.md

This repository uses a tool-agnostic bootstrap model.

Claude Code should not treat this file as the only source of truth.  
Use it as an entry trigger into the shared repository guidance below.

## Read First

Before making broad changes, read these in order:

1. [docs/progress/WORKLOG.md](/home/coffin/dev/AgenticLabeling/docs/progress/WORKLOG.md) — 가장 최근 개발·실험·결정 맥락 (최신순)
2. [docs/agents/START-HERE.md](/home/coffin/dev/AgenticLabeling/docs/agents/START-HERE.md)
3. [docs/navigability/README.md](/home/coffin/dev/AgenticLabeling/docs/navigability/README.md)
4. [docs/standards/adoption-model.md](/home/coffin/dev/AgenticLabeling/docs/standards/adoption-model.md)
5. [docs/standards/implementation-memory-summary.md](/home/coffin/dev/AgenticLabeling/docs/standards/implementation-memory-summary.md)

## Current Product Boundary

The active product boundary is the MVP monolith:

- `mvp_app/`
- `docker-compose.yml`
- `Dockerfile.mvp`
- `scripts/run_mvp.sh`

Legacy microservices remain available for reference and fallback:

- `services/`
- `docker-compose.legacy.yml`
- `scripts/run_legacy_docker.sh`

New product-facing work should target the MVP path first unless the task explicitly concerns legacy behavior.

## Project Memory

Repository memory lives in:

- [project_memory/README.md](/home/coffin/dev/AgenticLabeling/project_memory/README.md)

Useful query examples:

```bash
python3 -m project_memory.memory_store query --module-scope mvp_app --limit 20
python3 -m project_memory.memory_store query --category decision --limit 20
python3 -m project_memory.memory_store query --text docker --limit 10
```

## UI Work

The next major workstream is UI application.

Before UI work, read:

- [docs/standards/ui-change-gate.md](/home/coffin/dev/AgenticLabeling/docs/standards/ui-change-gate.md)
- [docs/plans/2026-04-20-ui-reference-mapping.md](/home/coffin/dev/AgenticLabeling/docs/plans/2026-04-20-ui-reference-mapping.md)

UI work should follow the workstation direction, not a generic dashboard direction.

## Research / Model Observation

AgenticLabeling originated from a research paper. Observing model architectures and data flow through Florence-2 / SAM3 is a first-class project activity.

Before any model architecture / data flow observation work, read:

- [docs/standards/research-observation-protocol.md](/home/coffin/dev/AgenticLabeling/docs/standards/research-observation-protocol.md)
- [docs/standards/model-inspection-conventions.md](/home/coffin/dev/AgenticLabeling/docs/standards/model-inspection-conventions.md)
- [research/README.md](/home/coffin/dev/AgenticLabeling/research/README.md)

Observation code lives in `research/` and `scripts/inspect_*.py`. Production code under `mvp_app/` must not be polluted with observation-only logging, hooks, or prints — use PyTorch's external hook APIs from scripts instead.

