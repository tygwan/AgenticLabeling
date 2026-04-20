# LEARNINGS

Curated candidate journal of general engineering rules discovered from friction events (failures, design decisions, repeated work, observations that contradict assumptions).

This file implements [C13 — Learn From Friction](../../../core/C13-learn-from-friction.md). Add a candidate only after passing the quality gate (at least two of four questions yes):

1. Is the rule applicable beyond this project?
2. Is it non-obvious compared to baseline documentation?
3. Did reaching it require meaningful diagnosis or deliberate design choice?
4. Can it be stated as an imperative rule usable by a future reader?

Do not use automated capture. Entries are created at three moments only:

- While writing a change-management or decision record
- Immediately after resolving a problem
- On explicit human request

## Lifecycle legend

| Status | Meaning |
|---|---|
| `draft` | Recorded once, observation only |
| `validated` | Recurrence confirmed in a materially different context |
| `promoted` | Absorbed into the formal standards corpus; link to target |
| `rejected` | Later evidence invalidates; retained with rejection reason |
| `stale` | No recurrence over the review window; cleanup candidate |

## Entry format

```markdown
### <YYYY-MM-DD> <one-line imperative title>
- **Trigger type**: error | design | repetition | observation | request
- **Triggered by**: concrete context (error message, design question, repetition scene)
- **Evidence**: log snippet / commit hash / file:line / measurement
- **Rule (draft)**: imperative statement, "When X, do Y"
- **Generality**: which languages / domains / frameworks this applies to
- **Recurrence**: 1 (increment on re-encounter)
- **Status**: draft | validated | promoted | rejected | stale
- **Related**: decision records, problem tickets, observation reports, commits
```

## Review cadence

- At session or milestone end: triage new `draft` entries, reject trivial ones.
- At project-phase boundaries (or every two weeks, whichever is sooner): consider `validated` → `promoted` transitions; sweep `stale` candidates.
- On new project bring-up: copy promoted standards. Leave `draft` behind — drafts are context-local to the project that discovered them.

## Entries

_No entries yet. Add the most recent entry at the top of this section._
