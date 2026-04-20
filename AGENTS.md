# AGENTS.md

This repository uses a tool-agnostic bootstrap model.

Any coding agent should treat this file as a trigger to the shared repository guidance, not as a complete standalone spec.

## Read First

Before major work, read these in order:

1. [docs/progress/WORKLOG.md](/home/coffin/dev/AgenticLabeling/docs/progress/WORKLOG.md) — single time-ordered feed of development, research and decisions. Start here for current context.
2. [docs/progress/LEARNINGS.md](/home/coffin/dev/AgenticLabeling/docs/progress/LEARNINGS.md) — curated candidate general principles (draft/validated/promoted)
3. [docs/agents/START-HERE.md](/home/coffin/dev/AgenticLabeling/docs/agents/START-HERE.md)
4. [docs/navigability/README.md](/home/coffin/dev/AgenticLabeling/docs/navigability/README.md)
5. [docs/standards/adoption-model.md](/home/coffin/dev/AgenticLabeling/docs/standards/adoption-model.md)
6. [docs/standards/learn-from-friction.md](/home/coffin/dev/AgenticLabeling/docs/standards/learn-from-friction.md) — the meta-loop that turns friction into reusable rules
7. [docs/standards/implementation-memory-summary.md](/home/coffin/dev/AgenticLabeling/docs/standards/implementation-memory-summary.md)

When writing a WORKLOG entry (any tag), apply learn-from-friction Rule 1.1: check whether the entry exposes a general, non-obvious, reusable rule. If it passes the four-question gate, append a draft to `docs/progress/LEARNINGS.md` and link it from the WORKLOG entry's Details.

## Active Product Boundary

Default runtime path:

- `mvp_app/`
- `docker-compose.yml`
- `Dockerfile.mvp`
- `scripts/run_mvp.sh`

Legacy path:

- `services/`
- `docker-compose.legacy.yml`
- `scripts/run_legacy_docker.sh`

Product-facing work should land in the MVP path first unless explicitly targeting legacy behavior.

## Repository Memory

Use the local project-memory store when the task depends on prior decisions, failures, or techniques.

Examples:

```bash
python3 -m project_memory.memory_store query --module-scope mvp_app --limit 20
python3 -m project_memory.memory_store query --category decision --limit 20
python3 -m project_memory.memory_store query --text sam2 --limit 10
```

## UI Work Trigger

Before UI work, read:

- [docs/standards/ui-change-gate.md](/home/coffin/dev/AgenticLabeling/docs/standards/ui-change-gate.md)
- [docs/plans/2026-04-20-ui-reference-mapping.md](/home/coffin/dev/AgenticLabeling/docs/plans/2026-04-20-ui-reference-mapping.md)

## Research / Model Observation Trigger

This project originated from a research paper. Observing model architectures and data flow (Florence-2, SAM3, etc.) is a first-class activity, not debugging.

Before any model architecture / data flow observation work, read:

- [docs/standards/research-observation-protocol.md](/home/coffin/dev/AgenticLabeling/docs/standards/research-observation-protocol.md)
- [docs/standards/model-inspection-conventions.md](/home/coffin/dev/AgenticLabeling/docs/standards/model-inspection-conventions.md)
- [research/README.md](/home/coffin/dev/AgenticLabeling/research/README.md)

Observation work lives in `research/` and `scripts/inspect_*.py`, not in `mvp_app/`.

