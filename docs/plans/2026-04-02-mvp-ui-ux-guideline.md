# AgenticLabeling MVP UI/UX Guideline

## Purpose

This document defines how UI/UX work should be handled in this project.

The MVP is not a generic dashboard. It is a data production workstation for:

- image upload
- Florence-2 detection
- SAM2 segmentation
- object review
- dataset export

Because of that, UI/UX is part of the core product behavior, not a cosmetic phase that starts only after all backend work is complete.

## Working Rule

Do not postpone UI/UX work until every backend feature is finished.

Use the following loop instead:

1. Build the minimum backend path.
2. Expose it in a usable review workflow.
3. Observe where the workflow is slow, unclear, or error-prone.
4. Add or change backend endpoints based on that workflow.
5. Refine the UI again.

In practice, this project should progress as:

1. core pipeline works
2. review workspace becomes usable
3. batch curation becomes efficient
4. export becomes predictable
5. visual polish comes last

## Product Principle

The UI should optimize for:

- review speed
- review accuracy
- low click count
- clear next action
- traceable validation state

The UI should not optimize first for:

- decorative layout
- marketing-style visuals
- large navigation structures
- feature completeness without operator flow

## Required Product Areas

### 1. Ingest

The ingest area must support:

- image upload
- class prompt input
- run action
- recent runs or recent sources
- backend status visibility

### 2. Review Workspace

This is the primary product surface.

It must contain:

- source list
- main image viewer
- object list
- object inspector
- validation actions
- overlay controls

The review workspace should answer these questions immediately:

- what image am I looking at
- how many objects exist
- which objects are still pending
- what did the model predict
- what should I do next

### 3. Curation

This area is for cleaning and preparing data at scale.

It should support:

- filter by status
- filter by category
- filter by confidence
- filter by object size
- batch approve
- batch delete
- relabel or merge categories later

### 4. Export

This area should support:

- export format selection
- validated-only option
- split configuration
- export history
- download access

## Review Workspace Standard Layout

The default review layout should follow this structure:

- left panel: source navigation
- center panel: image viewer
- right panel: selected object inspector
- lower or side list: object list with status

The image viewer must remain the visual center of the screen.

## Overlay Standard

The review UI should always support at least these visual modes:

- original
- bbox overlay
- segmentation overlay

The operator should be able to control:

- show bbox
- show mask
- show labels
- mask opacity
- validated only
- pending only

## Interaction Standard

The operator flow should prefer direct selection.

Required behavior:

- clicking an object in the list highlights it on the image
- approving an object should move focus to the next pending object
- deleting an object should preserve the current source context
- filtering should not destroy navigation context

## Batch Operation Standard

Single-object review is not enough for production use.

The product must grow toward:

- approve all objects in current source
- approve filtered objects
- delete filtered objects
- batch actions by category

## Keyboard Shortcut Standard

When the review flow stabilizes, keyboard-driven review should be added.

Recommended defaults:

- `A`: approve
- `D`: delete
- `N`: next object
- `P`: previous object
- `B`: toggle bbox
- `M`: toggle mask

## Engineering Rule

UI/UX changes are allowed to drive backend changes.

Examples:

- adding overlay controls may require new image render endpoints
- adding batch review may require bulk action endpoints
- adding inspector detail may require object detail endpoints

This is expected and should not be treated as rework.

## Priority Order

Use this order unless there is a strong reason to change it:

1. review workflow correctness
2. overlay controls
3. object inspector
4. batch actions
5. filtering and search
6. export usability
7. run history and progress visibility
8. visual polish

## Definition Of Good UI/UX For This Project

The UI is good when:

- a reviewer can understand the current image immediately
- a reviewer can distinguish bbox and mask output immediately
- a reviewer can process objects continuously without losing context
- a reviewer can export validated data without ambiguity

The UI is not good just because it looks modern.

## Current Direction

As of now, the project should be treated as:

- MVP monolith first
- review workstation second
- legacy microservices last

Any new UI work should be evaluated against that direction.
