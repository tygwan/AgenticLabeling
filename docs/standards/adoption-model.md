# AgenticLabeling Standards Adoption Model

## Purpose

`dev-standards` only matter if they change how work is done in this repository.

For AgenticLabeling, adoption means:

1. changes follow a stable product boundary
2. change scope is discoverable before edits start
3. non-obvious decisions and failures are recorded
4. verification is explicit
5. UI work does not bypass product and architecture rules

This document defines the project-specific operating model for that adoption.

## Current project boundary

The active product boundary is the MVP monolith.

Default runtime:

- `mvp_app/`
- `docker-compose.yml`
- `Dockerfile.mvp`
- `scripts/run_mvp.sh`
- `scripts/run_mvp_docker.sh`

Reference or legacy runtime:

- `services/`
- `docker-compose.legacy.yml`
- `scripts/run_legacy_docker.sh`

### Rule

New product-facing work should land in the MVP path first unless there is a specific reason to target legacy behavior.

## How the standards map into this repository

### C2 Repository Contract

Applied through:

- `docs/navigability/system-map.md`
- `docs/navigability/module-ownership-map.md`
- `docs/navigability/where-to-edit-guide.md`

Practical meaning:

- contributors should not start broad edits before locating the owning module
- UI changes should not begin in legacy services by default

### C3 Configuration And Environment

Applied through:

- `mvp_app/config.py`
- runtime environment variables
- path resolution that is relative to the repository or explicitly configured

Practical meaning:

- no new absolute machine-specific paths
- no UI asset or runtime path assumptions tied to a single workstation

### C4 Change Management

Applied through:

- scoped implementation plans in `docs/plans/`
- baseline tests for MVP behavior
- explicit separation between MVP and legacy runtime

Practical meaning:

- each major change should declare whether it touches MVP, legacy, docs, or standards
- cross-cutting changes should explain impact before implementation expands

### C5 Decision Records

Applied through:

- `docs/plans/`
- `project_memory/`

Practical meaning:

- non-obvious architecture or workflow decisions should be recoverable later
- do not rely only on commit history or chat transcript

### C6 Problem Tracking

Applied through:

- `docs/navigability/failure-memory-index.md`
- `project_memory/`

Practical meaning:

- recurring regressions should be recorded once and linked back to the owning scope

### C8 Verification

Applied through:

- `tests/test_mvp_app.py`
- `tests/test_mvp_detector.py`
- `tests/test_mvp_segmenter.py`
- `tests/test_mvp_e2e.py`
- `tests/test_project_memory.py`

Practical meaning:

- UI changes still require backend and workflow verification
- new behavior should extend tests or justify why test changes are not needed

### C9 Interfaces And Contracts

Applied through:

- `docs/navigability/interface-catalog.md`
- `mvp_app` route contracts

Practical meaning:

- endpoint changes and configuration changes are contract changes
- UI work that changes workflow behavior should update the interface catalog if the contract moves

### C12 Documentation And Discoverability

Applied through:

- `docs/navigability/`
- `project_memory/README.md`
- `docs/standards/`

Practical meaning:

- project understanding should not require reading the whole codebase
- navigability docs are part of the codebase, not optional side notes

## Required artifacts for this repository

These artifacts are part of the applied standard, not optional extras:

- `docs/navigability/`
- `project_memory/`
- MVP baseline tests
- project plans for major transitions
- standards application docs in `docs/standards/`

## Working model before each major change

Before a major change starts:

1. identify the active product boundary
2. identify the owning modules
3. identify the contracts likely to move
4. identify the verification surface
5. identify whether a new decision or failure record will be needed

After the change:

1. run or update the relevant verification
2. update navigability docs if edit locations or ownership changed
3. record the non-obvious decision or failure if the change introduced one

## What “applied” means for AgenticLabeling

The standards should be considered applied when:

- contributors can locate the right edit area without broad codebase scanning
- major behavior changes have an explicit verification path
- UI work is constrained by product boundaries and contracts
- repeated regressions are captured in project memory
- MVP remains the default runtime unless intentionally changed

## Immediate next linkage

The next major workstream is UI application.  
That means the standards must constrain the UI phase in the following way:

- UI work targets the MVP runtime first
- UI work respects current review/export contracts unless intentionally changed
- UI work updates navigability and interface docs if the edit surface changes
- UI work carries explicit verification instead of relying on visual inspection alone

