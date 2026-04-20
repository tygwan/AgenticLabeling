# CLAUDE.md

This repository uses a tool-agnostic bootstrap model.

Claude Code should not treat this file as the only source of truth.  
Use it as an entry trigger into the shared repository guidance below.

## Read First

Before making broad changes, read these in order:

1. [docs/progress/WORKLOG.md](/home/coffin/dev/AgenticLabeling/docs/progress/WORKLOG.md) — 가장 최근 개발·실험·결정 맥락 (최신순)
2. [docs/progress/LEARNINGS.md](/home/coffin/dev/AgenticLabeling/docs/progress/LEARNINGS.md) — curated 일반 원칙 후보집 (draft/validated/promoted)
3. [docs/concepts/README.md](/home/coffin/dev/AgenticLabeling/docs/concepts/README.md) — 개념 허브 (설명의 single source of truth)
4. [docs/agents/START-HERE.md](/home/coffin/dev/AgenticLabeling/docs/agents/START-HERE.md)
5. [docs/navigability/README.md](/home/coffin/dev/AgenticLabeling/docs/navigability/README.md)
6. [docs/standards/adoption-model.md](/home/coffin/dev/AgenticLabeling/docs/standards/adoption-model.md)
7. [docs/standards/learn-from-friction.md](/home/coffin/dev/AgenticLabeling/docs/standards/learn-from-friction.md) — 마찰 이벤트에서 일반 원칙을 뽑는 루프
8. [docs/standards/concepts-protocol.md](/home/coffin/dev/AgenticLabeling/docs/standards/concepts-protocol.md) — 개념 허브 규칙 (설명 중복 금지)
9. [docs/standards/implementation-memory-summary.md](/home/coffin/dev/AgenticLabeling/docs/standards/implementation-memory-summary.md)

**WORKLOG entry 작성 시 준수사항**:
1. learn-from-friction Rule 1.1 — "여기서 뽑을 LEARNING 후보 있나?" 자기 점검. 게이트 통과 시 `docs/progress/LEARNINGS.md` 상단에 draft로 append 후 entry Details에 링크.
2. concepts-protocol Rule 4 — "이 entry가 기대는 개념이 concept 파일에 있나? 없으면 stub 생성 가치 있나?" 자기 점검. 해당 concept 파일로 **위임** (entry 본문에 개념 설명 반복하지 말 것).

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

