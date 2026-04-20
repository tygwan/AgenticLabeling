# AgenticLabeling MVP Refactor Plan

## Objective

Stabilize the current monolith MVP as the new product baseline, then evolve it into a review-focused workstation without re-expanding into the previous microservice sprawl.

## Current Baseline

The active baseline is the monolith app in [mvp_app/main.py](/home/coffin/dev/AgenticLabeling/mvp_app/main.py).

Supported flow:

1. Image upload
2. Florence-2 detection
3. SAM2 segmentation
4. Source/object registry write
5. Review and approve/delete
6. YOLO/COCO export

Current runtime and packaging:

- Default runtime: [docker-compose.yml](/home/coffin/dev/AgenticLabeling/docker-compose.yml)
- MVP image: [Dockerfile.mvp](/home/coffin/dev/AgenticLabeling/Dockerfile.mvp)
- Legacy stack preserved in [docker-compose.legacy.yml](/home/coffin/dev/AgenticLabeling/docker-compose.legacy.yml)

## Phase 1. Lock The MVP Baseline

Goal: prevent regressions while UI and workflow evolve.

Deliverables:

- Document the MVP API and review workflow contract
- Keep smoke/e2e tests aligned with the current review/export path
- Keep `/health` as the runtime truth source for app health and segmentation backend
- Remove stale documentation that still describes the old stack as the default product

Success criteria:

- `upload -> detect -> segment -> review -> approve -> export` remains green
- `/review` continues to expose Original, BBox Overlay, and Segmentation Overlay
- `/health` always returns `status`, `app`, `stats`, and `segmentation_backend`

## Phase 2. Review Workspace UX

Goal: convert the current demo-style review page into a production review workstation.

Priority order:

1. Object inspector
2. Overlay controls
3. Source/object synchronized selection
4. Next pending object navigation
5. Keyboard shortcuts

Target layout:

- Left: source list and filters
- Center: image viewer
- Right: object inspector and actions
- Top: workflow controls and status

## Phase 3. Batch Curation Features

Goal: reduce operator time per image and per object.

Required capabilities:

- Batch approve
- Batch delete
- Category filter
- Pending only / validated only filter
- Confidence threshold filter
- Source-level approve all

Dependency note:

This phase requires both UI work and new API operations. It should not be treated as frontend-only work.

## Phase 4. Deployment And Ops Stabilization

Goal: make the MVP the default supported runtime.

Deliverables:

- Finalize environment variable contract
- Keep Docker and local run instructions in sync
- Document GPU/SAM2 expectations clearly
- Keep the legacy stack isolated from default workflows

Known constraint:

The monolith removes duplicated service images, but the GPU runtime image is still large. This is now an optimization problem, not an architecture problem.

## Phase 5. Legacy Isolation

Goal: stop the previous partially refactored microservice code from confusing the product path.

Current legacy hotspots:

- [services/label-studio-lite/app/main.py](/home/coffin/dev/AgenticLabeling/services/label-studio-lite/app/main.py)
- [services/object-registry/app/main.py](/home/coffin/dev/AgenticLabeling/services/object-registry/app/main.py)
- [services/object-registry/app/registry.py](/home/coffin/dev/AgenticLabeling/services/object-registry/app/registry.py)

Policy:

- Do not delete legacy code until the MVP replacement is clearly stronger
- Keep legacy execution paths explicit and separate
- Remove only after remaining references and operational need are confirmed

## Execution Order

1. Lock the MVP baseline
2. Upgrade the review workspace
3. Add batch curation features
4. Stabilize deployment and ops
5. Isolate or remove legacy code
