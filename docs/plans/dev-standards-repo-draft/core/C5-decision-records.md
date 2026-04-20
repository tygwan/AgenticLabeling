# C5 — Decision Records

## Rule

Structural decisions must be recorded in a discoverable form with enough context to explain why the chosen direction was taken over plausible alternatives.

## Why

Architectural and process memory decays quickly. Unrecorded decisions force teams to re-litigate old trade-offs or repeat old mistakes.

## Required minimum

- What question or problem prompted the decision
- What decision was taken
- Why it was taken
- What alternatives were considered
- What impact or follow-up it creates

## Application guidance

- Record decisions that affect structure, conventions, dependencies, or long-lived behavior
- Do not record trivial local implementation choices
- Keep decision references stable over time
- When reversing a decision, record the reversal instead of erasing history

## Examples

Good:

- "Adopt one schema migration path because multiple paths caused drift."
- "Use runtime fallback because optional dependency absence must not break the service."

Bad:

- "We used X because it felt simpler."
- No alternatives or impact recorded

## Out of scope

- One exact ADR template
- One specific document location
- Mandatory formal review meetings

