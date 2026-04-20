# C10 — Observability

## Rule

Projects must expose enough runtime information for developers or operators to understand health, failures, and key workflow transitions.

## Why

A system that cannot explain its own behavior is expensive to operate, debug, and trust.

## Required minimum

- Meaningful runtime logs or equivalent signals
- A discoverable health indicator where runtime service behavior matters
- Enough context to diagnose failures without guesswork
- Visibility into critical workflow outcomes

## Application guidance

- Log meaningful state transitions, not noise
- Prefer structured information over ambiguous free text where practical
- Expose health in a way appropriate to the system type
- Include identifiers that help correlate cause and effect

## Examples

Good:

- Startup health, backend selection, workflow counts, failure reasons
- Service logs that identify request, job, or artifact context

Bad:

- Silent failure
- Logs that say "error happened" with no identifying context

## Out of scope

- One logging framework
- One metrics backend
- Full observability stack for every small project

