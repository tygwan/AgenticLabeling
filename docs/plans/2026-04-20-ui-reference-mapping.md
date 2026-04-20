# AgenticLabeling UI Reference Mapping

## Purpose

This document maps the UI reference package in `reference/ui_refer/` to the current AgenticLabeling MVP runtime.

It exists so the next UI implementation phase starts from:

- a known screen model
- a known component breakdown
- a known contract gap list

instead of from open-ended frontend exploration.

## Reference package contents

Primary files:

- `reference/ui_refer/AgenticLabeling.html`
- `reference/ui_refer/styles.css`
- `reference/ui_refer/components/home.jsx`
- `reference/ui_refer/components/review.jsx`
- `reference/ui_refer/components/viewer.jsx`
- `reference/ui_refer/components/screens.jsx`

The reference is a workstation-oriented frontend prototype with these routes:

- `home`
- `review`
- `export`
- `settings`

## Current MVP runtime

The current product runtime is still the monolith in:

- `mvp_app/main.py`
- `mvp_app/detector.py`
- `mvp_app/segmenter.py`
- `mvp_app/registry.py`
- `mvp_app/storage.py`

Current user-facing routes are centered around:

- `GET /`
- `POST /upload`
- `GET /review`
- review actions
- overlay image endpoints
- export download
- `GET /health`

## Direct fit with current MVP

### 1. Review workspace

The reference strongly matches the current product direction.

Already aligned:

- workstation model instead of dashboard model
- three-pane review layout
- separate overlay controls
- object selection and object actions
- source navigation

This is the best candidate for first implementation.

### 2. Home / ingest screen

The reference home screen also fits the project direction, but only partially maps to current backend behavior.

Already aligned:

- project id input
- class prompt input
- pipeline run mental model
- health/runtime awareness

Not yet fully backed:

- recent runs table
- per-stage live progress
- project cards and run history

This means the home screen can be implemented in phases:

- phase A: static or derived-from-current-state MVP home
- phase B: real run history and progress once backend support exists

### 3. Export screen

The reference export screen is directionally correct, but current backend support is thinner than the mock.

Already aligned:

- export format choice
- validated-only intent
- dataset naming

Gaps:

- split configuration is not yet a stable UI contract
- export history is not yet a stable route/record surface
- live class distribution summary is not yet surfaced by the current export path

### 4. Settings screen

The reference settings screen is mostly a future-facing admin surface.

Some values exist today:

- Florence-2 model id
- SAM2 config and checkpoint
- path settings

But the current app does not expose them as a dedicated settings workflow.

This screen should not be the first implementation target unless it is read-only.

## Recommended implementation order

### Phase 1. Review layout adoption

Target:

- apply the three-pane review workstation layout
- preserve the current review contract
- move existing review controls into the new structure

Expected edit scopes:

- `mvp_app/main.py`
- possibly new template/static assets if the UI is split out of inline HTML
- `tests/test_mvp_app.py`
- `tests/test_mvp_e2e.py`
- `docs/navigability/interface-catalog.md`
- `docs/navigability/where-to-edit-guide.md`

### Phase 2. Home / ingest adoption

Target:

- replace the minimal current landing page with the reference ingest-oriented home
- keep upload and class prompt behavior wired to the existing backend

Expected backend gaps:

- run history can begin as placeholder or derived data
- stage progress will need either fake progress or explicit pipeline event support

### Phase 3. Export surface adoption

Target:

- add a dedicated export screen structure
- wire the current export path into it

Expected backend gaps:

- split config
- export history
- richer export summary data

### Phase 4. Settings surface

Target:

- read-only settings summary first
- editable settings only after config contract is intentionally stabilized

## Contract impact assessment

### Review

Likely to be mostly a workflow and presentation change, with some possible contract additions.

Potential additions:

- richer object detail endpoint
- source list summary endpoint
- batch action endpoints

### Home

Likely to require new backend support if the reference behavior is implemented fully.

Potential additions:

- run history endpoint
- pipeline stage/progress state
- project summary endpoint

### Export

Likely to require new backend support.

Potential additions:

- export summary endpoint
- export history endpoint
- split configuration contract

### Settings

Likely to require explicit configuration read endpoints if it becomes more than a static summary.

## UI-change-gate implications

Per `docs/standards/ui-change-gate.md`, the next UI phase should be treated as:

- review layout work: `workflow change`
- home adoption: `workflow change` plus likely `backend-supporting UI change`
- export adoption: `workflow change` plus likely `contract change`
- settings adoption: `presentation-only` if read-only, otherwise `contract change`

## Practical next step

The first implementation pass should focus on:

1. review workspace layout
2. object inspector structure
3. overlay control placement
4. preserving current MVP review routes and tests

That gives the highest product value with the lowest backend expansion cost.

