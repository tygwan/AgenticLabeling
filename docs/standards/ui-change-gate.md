# AgenticLabeling UI Change Gate

## Purpose

The next major workstream is UI application.  
This gate defines the minimum standard that UI work must satisfy before it expands.

## Scope

This applies to:

- review workspace changes
- overlay control changes
- object inspector changes
- source navigation changes
- export UX changes
- page structure or frontend asset changes tied to the MVP

## Rule 1. UI targets the active runtime

UI work must target the MVP runtime first:

- `mvp_app/`
- current review/export flow

Do not start by extending legacy UI surfaces unless the task is explicitly about legacy behavior.

## Rule 2. UI work must declare its change type

Every substantial UI change should classify itself as one or more of:

- presentation-only
- workflow change
- contract change
- backend-supporting UI change

This matters because verification depth changes with the class of change.

## Rule 3. UI work must name its owning modules

Before implementation starts, the change should identify:

- primary edit files
- related API or render endpoints
- affected tests
- affected docs

For the current MVP, this usually means some subset of:

- `mvp_app/main.py`
- `mvp_app/registry.py`
- `mvp_app/storage.py`
- `tests/test_mvp_app.py`
- `tests/test_mvp_e2e.py`
- `docs/navigability/interface-catalog.md`
- `docs/navigability/where-to-edit-guide.md`

## Rule 4. UI work must keep workflow verification explicit

Manual browser inspection is not enough by itself.

At least one of the following must remain true:

- existing tests still cover the affected workflow
- tests are updated to reflect the new workflow
- the change is documented as presentation-only and leaves interfaces untouched

## Rule 5. UI work must update discovery artifacts when structure changes

If the UI phase changes:

- route ownership
- page composition
- edit entrypoints
- interface boundaries

then update:

- `docs/navigability/interface-catalog.md`
- `docs/navigability/where-to-edit-guide.md`
- `docs/navigability/system-map.md` if the runtime structure moves

## Rule 6. UI work should prefer the workstation model

The current product direction is:

- MVP monolith first
- review workstation second
- legacy microservices last

UI work should reinforce that direction, not dilute it into a generic dashboard.

## Pre-implementation gate

Before UI implementation starts, confirm:

- target runtime is MVP
- target screens are identified
- target files are identified
- likely contract changes are identified
- likely verification surface is identified

## Post-implementation gate

Before UI work is considered done, confirm:

- relevant tests still pass
- changed contracts are documented
- changed edit surfaces are reflected in navigability docs
- new workflow assumptions are written down if they are non-obvious

