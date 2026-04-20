# C14 — Concept Hub

## Rule

Explanatory content (how a concept works, why a design behaves the way it does) must live in a single canonical location — a concept hub — and be referenced from other records by link. Observations, learning candidates, decision records, and time-ordered work logs must not duplicate conceptual explanations; they link into the hub instead.

## Why

Projects naturally produce four kinds of durable records:

- observations — measurements and empirical findings from a specific run
- candidate rules — general principles discovered from friction (see C13)
- decisions — choices made with their rationale (see C5)
- work logs — time-ordered activity feed

Explanatory content (for example "how does the chosen async server route requests, why does BF16 drift appear at decode boundaries, how does SQLite WAL locking work") is structurally different from all four. When it is embedded inside them it scatters, drifts across copies, and stales as understanding deepens. A dedicated concept hub gives explanation a single source of truth, so other records shrink to their native purpose and the hub stays the single place to update.

## Required minimum

Projects adopting this rule must provide:

1. A dedicated concept directory (for example `docs/concepts/`) with an index.
2. Each concept file uses a fixed three-section structure:
   - **Concept framework** — general knowledge unbound to this project
   - **Evidence in this project** — concrete evidence accrued from observations, code references, and external documents, with citations
   - **Related rules / decisions** — back-links to relevant candidate rules (C13), decision records (C5), and work-log entries
3. Each concept file declares provenance metadata: whether it is research-derived, engineering-based, or hybrid; what its primary evidence is.
4. A quality gate applied before creation. At least two of the four questions must answer yes:
   - Will this concept recur elsewhere in the project?
   - Is it non-obvious to a new contributor?
   - Will the author or a future contributor want to re-read it?
   - Is at least one concrete evidence citation already available, so the file is not an orphan?
5. Four allowed trigger moments, and no others:
   - Observation completion, when the interpretation section exposes generalizable framing
   - Work-log entry authoring, when the entry leans on a non-obvious concept
   - Candidate-rule authoring, when the rule's background would otherwise bloat the rule body
   - Explicit human question reaching conceptual depth
6. A direction of flow: observations produce evidence, concepts synthesize evidence, candidate rules and decisions reference concepts. Concepts do not copy raw measurements from observations; they cite them.

## Application guidance

- Keep concept files self-contained. A reader landing on a concept should find the explanation, the project-local evidence, and the backward links without jumping between many files.
- Do not write the same explanation in both an observation report and a concept file. Prefer the concept file and have the observation end with an "evidence contribution" pointer.
- Candidate rules (C13) hold the imperative statement only. The mechanism or background belongs in a concept; link to it.
- Update an existing concept's Evidence section whenever a new observation or code change contributes a fresh datapoint. Promote deeper framing to the Concept framework section only when it has stabilized across multiple contributions.
- Move retired concepts to an archive rather than deleting them, so back-links stay valid and the decision history is preserved.
- Keep the concept index ordered so recent additions are easy to find.

## Examples

Good concepts:

- "BF16 precision in inference" — derived from multiple dtype observations, cites them as evidence, links to the rule "compare raw samples bit-exactly before claiming equivalence" for the regime where it applies.
- "WSGI vs ASGI and sync/async dispatch in a web framework" — engineering concept, cites the project's handler signatures and deployment configuration as evidence, links to any decision that picked the runtime.
- "Schema evolution with idempotent additive changes" — hybrid concept, cites both the project's migration pattern and the external SQLite documentation for why it works.

Bad concepts:

- A concept file that only restates the one observation it came from — belongs in the observation, not the hub.
- A concept file explaining something the official framework documentation already covers at the same depth, with no project-local evidence.
- A concept file whose Evidence section is empty and points at nothing.

Good flow:

- New observation reveals an unexpected behavior → writer captures measurements in the observation report, delegates the interpretation to a new or existing concept file, and leaves an Evidence contribution pointer back to it.
- A later rule candidate (C13) arises from the same observation → the rule links to the concept for the "why," and the concept's Related section grows by one backlink.

Bad flow:

- An observation report grows a multi-page Interpretation section that would be useful to every future reader but lives permanently inside that one report.
- A candidate rule file includes a textbook-style explanation of the underlying mechanism, duplicating the concept's Concept framework section.

## Out of scope

- One specific directory name or file-path layout. Teams may use `docs/concepts/`, `knowledge/`, `reference/`, or a wiki, as long as the required minimum holds.
- Mandating a single metadata schema. Provenance metadata must exist but its format (front matter, headers, separate index) is a team choice.
- Auto-generation of concept files from observations. Human judgment is required for synthesis; tooling can assist with link maintenance but must not replace authorship.
- Ownership rules. Concepts are collaboratively maintained; multi-author editing conventions are delegated to the project's change-management rule (C4).
