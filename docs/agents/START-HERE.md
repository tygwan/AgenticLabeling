# AgenticLabeling Agent Bootstrap

This is the canonical, tool-agnostic bootstrap document for coding agents and humans entering the repository.

Use this document to avoid relying on one CLI-specific convention.

## Purpose

Different coding CLIs use different entry files:

- Codex typically notices `AGENTS.md`
- Claude Code typically notices `CLAUDE.md`

Those files should both point here so the repository has one shared bootstrap path.

## Read order

### Always read first

1. [../progress/WORKLOG.md](/home/coffin/dev/AgenticLabeling/docs/progress/WORKLOG.md) — most recent development, research and decision context (newest first)
2. [../progress/LEARNINGS.md](/home/coffin/dev/AgenticLabeling/docs/progress/LEARNINGS.md) — curated candidate general principles
3. [../navigability/README.md](/home/coffin/dev/AgenticLabeling/docs/navigability/README.md)
4. [../standards/adoption-model.md](/home/coffin/dev/AgenticLabeling/docs/standards/adoption-model.md)
5. [../standards/learn-from-friction.md](/home/coffin/dev/AgenticLabeling/docs/standards/learn-from-friction.md)
6. [../standards/implementation-memory-summary.md](/home/coffin/dev/AgenticLabeling/docs/standards/implementation-memory-summary.md)

### Read by task type

If the task is:

- product/UI work:
  - [../standards/ui-change-gate.md](/home/coffin/dev/AgenticLabeling/docs/standards/ui-change-gate.md)
  - [../plans/2026-04-02-mvp-ui-ux-guideline.md](/home/coffin/dev/AgenticLabeling/docs/plans/2026-04-02-mvp-ui-ux-guideline.md)
  - [../plans/2026-04-20-ui-reference-mapping.md](/home/coffin/dev/AgenticLabeling/docs/plans/2026-04-20-ui-reference-mapping.md)
- architecture/runtime work:
  - [../tech-specs/architecture-spec.md](/home/coffin/dev/AgenticLabeling/docs/tech-specs/architecture-spec.md)
  - [../plans/2026-04-20-mvp-refactor-plan.md](/home/coffin/dev/AgenticLabeling/docs/plans/2026-04-20-mvp-refactor-plan.md)
- model architecture / data flow observation work (research):
  - [../standards/research-observation-protocol.md](/home/coffin/dev/AgenticLabeling/docs/standards/research-observation-protocol.md)
  - [../standards/model-inspection-conventions.md](/home/coffin/dev/AgenticLabeling/docs/standards/model-inspection-conventions.md)
  - [../../research/README.md](/home/coffin/dev/AgenticLabeling/research/README.md)
- historical or recurring issue analysis:
  - [../navigability/failure-memory-index.md](/home/coffin/dev/AgenticLabeling/docs/navigability/failure-memory-index.md)
  - [../../project_memory/README.md](/home/coffin/dev/AgenticLabeling/project_memory/README.md)

## Current product boundary

The active product boundary is the MVP monolith.

Default runtime path:

- `mvp_app/`
- `docker-compose.yml`
- `Dockerfile.mvp`
- `scripts/run_mvp.sh`

Legacy path:

- `services/`
- `docker-compose.legacy.yml`
- `scripts/run_legacy_docker.sh`

### Rule

New product-facing work should target the MVP path first unless the task is explicitly about legacy behavior.

## Repository memory trigger

When prior context matters, query the local memory store instead of scanning broad parts of the codebase first.

Example queries:

```bash
python3 -m project_memory.memory_store query --module-scope mvp_app --limit 20
python3 -m project_memory.memory_store query --category decision --limit 20
python3 -m project_memory.memory_store query --text docker --limit 10
python3 -m project_memory.memory_store query --text ui --limit 10
```

## Working rule

Before broad edits:

1. identify the active product boundary
2. identify the owning modules
3. identify likely interface changes
4. identify the verification surface
5. check whether a prior decision or failure record already exists

## Why this file exists

The repository should not depend on one vendor-specific trigger mechanism.

The correct model is:

- `AGENTS.md` -> points here
- `CLAUDE.md` -> points here
- this file -> points to the actual shared repository context

