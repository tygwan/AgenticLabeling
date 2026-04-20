# AGENTS.md

This repository uses a tool-agnostic bootstrap model.

Any coding agent should treat this file as a trigger to the shared repository guidance, not as a complete standalone spec.

## Read First

Before major work, read these in order:

1. [docs/agents/START-HERE.md](/home/coffin/dev/AgenticLabeling/docs/agents/START-HERE.md)
2. [docs/navigability/README.md](/home/coffin/dev/AgenticLabeling/docs/navigability/README.md)
3. [docs/standards/adoption-model.md](/home/coffin/dev/AgenticLabeling/docs/standards/adoption-model.md)
4. [docs/standards/implementation-memory-summary.md](/home/coffin/dev/AgenticLabeling/docs/standards/implementation-memory-summary.md)

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

