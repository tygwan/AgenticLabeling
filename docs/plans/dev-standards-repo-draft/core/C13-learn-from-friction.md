# C13 — Learn From Friction

## Rule

Engineering rules are discovered bottom-up from friction events (failures, design decisions, repeated work, observations that contradict assumptions), not inherited top-down from prescriptive documents. Projects must maintain a curated candidate journal so discovered rules survive beyond the session they were learned in, and must promote them to formal standards only after validation through recurrence.

## Why

Prescriptive standards written in advance tend to under-specify the situations practitioners actually run into, while in-session realizations evaporate when the session ends. Without a capture loop, the same friction is re-encountered on future projects by the same or different engineers. Indiscriminate logging (every shell error, every type mismatch) creates noise that drowns out genuine insights and devalues the record as a competence signal. A deliberate, lightly curated capture with a promotion gate produces a portfolio-grade knowledge base and keeps the formal standard set small and trusted.

## Required minimum

Projects adopting this rule must provide:

1. A single append-only candidate buffer file (for example `LEARNINGS.md`).
2. A quality gate applied before each recording. At least two of the four questions must answer yes:
   - Is the rule applicable beyond the project (language, domain, framework agnostic)?
   - Is it non-obvious compared to baseline documentation or tutorials?
   - Did reaching it require meaningful diagnosis or deliberate design choice?
   - Can it be stated as an imperative rule usable by a future reader?
3. A defined lifecycle: `draft` → `validated` (recurrence confirmed in a different context) → `promoted` (absorbed into the standard corpus) → `rejected` or `stale` where appropriate.
4. Three allowed capture moments, and no others:
   - While writing a change-management or decision entry (C4, C5)
   - Immediately after resolving a problem (C6)
   - On explicit human request
5. A link from each candidate entry to its supporting evidence (commit, log excerpt, decision record, observation report).

## Application guidance

- Do not use automated capture (hooks, CI scrapers) for this buffer. Automation captures without judgment and degrades the signal/noise ratio.
- Treat trivial typos, one-off environment configuration slips, and directly documented behavior as out of scope. They do not pass the gate.
- When an existing candidate recurs in a new context, increment its recurrence counter rather than duplicating the entry.
- Reject entries explicitly when later evidence invalidates them, and keep the rejection in-place so the same false lead is not re-raised.
- Promote only after recurrence is observed in a materially different context (another module, another service, another project, another language). A single-context observation stays in `draft`.
- When promoting, write the formal rule into the appropriate location in the standard corpus and record the promotion link on the candidate.
- Review the buffer at least at project-phase boundaries and at session end. Without a review cadence the buffer degrades into a wish list.

## Examples

Good candidates:

- "When a library dynamically loads third-party code (for example via trust-remote-code or plugin systems), cap the library version at the third-party code's target major version until upstream confirms the new major is supported."
- "Compare raw outputs bit-exactly before claiming two pipelines are equivalent; aggregate statistics can hide divergences localized to boundary cases."
- "Route observability code through external hooks rather than by editing the observed component. Inline debug logging accumulates removal cost and blurs responsibility."

Rejected candidates:

- "Typed the wrong path once." — fails Q3 and Q4.
- "Tutorial says to use pathlib." — fails Q2.
- "Added a TODO comment." — fails Q1 and Q4.

Good lifecycle:

- First observation → `draft`.
- Same principle rediscovered in an unrelated module or project → `validated`.
- Two or more validations plus visible generality → `promoted` into the formal standard set, candidate retains a promotion link.

Bad lifecycle:

- Entries promoted after a single observation without recurrence.
- Entries silently deleted after rejection.

## Out of scope

- One exact file name, schema, or tool. Teams may choose their own buffer file, their own metadata fields, and their own review cadence, as long as the required minimum is satisfied.
- Mandating a particular standards directory for promotions. The promotion destination is a team choice.
- Automated classification or machine-assisted promotion. Human judgment is required at the gate.
