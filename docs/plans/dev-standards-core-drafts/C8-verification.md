# C8 — Verification

## Rule

Work must not be considered complete without verification evidence appropriate to the kind of change.

Verification may be automated or manual, but it must be explicit.

## Why

Engineering outcomes are not credible if teams cannot show how they know something works or improved.

## Required minimum

- A defined verification path for meaningful changes
- Regression protection for important fixes when feasible
- Explicit manual verification steps when automation is not practical
- Clear distinction between unverified and verified work

## Application guidance

- Match verification depth to change risk
- Prefer automated checks for repeatable behavior
- Record manual verification when that is the only practical route
- Treat previous failures as candidates for regression coverage

## Examples

Good:

- Unit or integration tests for code-level behavior
- Manual checklist for UI or operational paths that cannot yet be automated

Bad:

- "Should work" with no evidence
- Important bug fixed without any repeatable check

## Out of scope

- One test framework
- One coverage target
- Automation for every possible workflow

