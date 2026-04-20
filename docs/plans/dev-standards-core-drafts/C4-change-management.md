# C4 — Change Management

## Rule

Changes must be understandable, reviewable, and reversible.

Projects must group work coherently, describe what changed and why, and treat destructive changes as higher-risk operations.

## Why

Software quality depends on controlled change, not just correct code at a single point in time.

## Required minimum

- Coherent change units
- Human-readable change descriptions
- Verification evidence attached to meaningful changes
- Explicit handling of destructive or risky operations

## Application guidance

- Keep unrelated changes separate
- Explain intent, not just surface edits
- Prefer reversible moves over irreversible deletion where practical
- Escalate changes that alter structure, contracts, or critical data

## Examples

Good:

- One change for one feature or fix
- Description includes what changed, why, and how it was checked

Bad:

- Mixed refactor, bugfix, formatting, and config drift in one opaque change
- Risky deletion with no backup path or review context

## Out of scope

- One commit message format
- One branching model
- One pull request process

