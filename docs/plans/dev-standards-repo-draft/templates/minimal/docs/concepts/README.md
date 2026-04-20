# Concepts

Explanatory hub for this project's recurring technical and domain concepts. Single source of truth per concept; other records (observations, learning candidates, decisions, work logs) link in rather than duplicate the explanation.

This directory implements [C14 — Concept Hub](../../../core/C14-concept-hub.md). Read that rule before adding files here.

## Structure

Organize concepts by category (examples below — adjust to the project's domain):

```
concepts/
├── README.md             # this file, plus the concept index
├── <domain-a>/           # e.g. ml, backend, frontend, infra, testing
└── <domain-b>/
```

## Required file template

Each concept file follows this fixed structure. Front matter is mandatory.

```markdown
---
type: research-derived | engineering | hybrid
primary_evidence: <observation paths | code refs | external docs>
promoted_from_learnings: <optional — a promoted candidate rule>
---

# <Concept Name>

## Concept framework
General background to advanced trade-offs. Written independent of this project so it transfers.

## Evidence in this project
Where and how the concept actually shows up here. Each bullet cites observation reports, code locations, or external documents with a short quote or summary. Never copy raw measurements; cite the observation.

## Related rules / decisions
Back-links to candidate rules (C13), decision records (C5), and work-log entries that lean on this concept. Links only, no duplicate explanation.
```

## Gate — when to create a concept file

Add a concept file only when at least two of these four questions answer yes:

1. Will this concept recur in other parts of the project?
2. Is it non-obvious to a new contributor?
3. Is it worth re-reading by the author or a future contributor?
4. Is at least one concrete evidence link already available?

If fewer than two pass, keep the explanation inline in the record where it first appeared.

## Triggers — when to touch the hub

- **Observation completion** — delegate interpretation-level content to a concept file; leave an "Evidence contribution" pointer in the observation report.
- **Work-log entry authoring** — if the entry leans on a concept that is already documented, link it. If not, consider creating a stub that passes the gate.
- **Candidate-rule authoring (C13)** — move the rule's background mechanism into the concept; the candidate rule stays imperative.
- **Deep human question** — if answering takes conceptual depth, persist the answer as a concept.

## Index

Maintain a chronological, newest-first list of registered concepts here.

_(No entries yet — add the most recent concept on top of this section when the first one is created.)_
