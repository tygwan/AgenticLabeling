# Example: AI-Assisted Internal Tool

## Project Shape

This example represents a medium-sized internal engineering tool that:

- has a codebase large enough that full re-reading is expensive
- uses AI assistance for implementation and refactoring
- maintains project memory outside raw chat transcripts
- uses SQLite as a lightweight structured memory store

The example is intentionally generic. It is not tied to a specific business domain.

## Active Standard Composition

- core: all universal core rules
- profiles:
  - `ai-assisted-development`
- adapters:
  - `github`
  - `python`
  - `ci`
  - `project-memory`

## Why This Example Exists

This example demonstrates the exact case where AI-driven development starts to break down if the project is not navigable:

- too much code to read in full for every change
- repeated re-discovery of old failures
- repeated duplication of helpers or shapes
- weak memory of past decisions

The example shows how the standard is meant to reduce those failures.

## Project Layout

```text
project/
├── README.md
├── docs/
│   ├── system-map.md
│   ├── module-ownership-map.md
│   ├── interface-catalog.md
│   ├── failure-memory-index.md
│   └── decisions/
├── src/
├── tests/
├── scripts/
└── project-memory/
    ├── schema.sql
    └── retrieval-notes.md
```

## What The AI Uses

The AI should not start from full repository ingestion.

It should start from:

1. project README
2. system map
3. module ownership map
4. interface catalog
5. failure memory index
6. selected decision records

Only then should it read the local code needed for the task.

## Why SQLite Appears Here

This example uses SQLite not because SQLite is the standard, but because it is a low-friction way to store structured project memory.

The standard requirement is:

- memory should be structured
- memory should be queryable
- memory should link back to authoritative artifacts

SQLite is one valid implementation choice.

## Example Memory Categories

- `decision`
- `failure`
- `technique`
- `constraint`
- `open_question`

## Example Retrieval Strategy

Instead of replaying all transcripts:

- store summaries
- store detailed content
- store tags
- store metadata
- store source links

Retrieval should prefer:

- decision summaries
- previous failures
- implementation techniques
- recent high-value edits

over raw transcript replay.

## Example Failure Prevention

This setup is designed to prevent:

- recreating helpers that already exist
- duplicating shapes already in the codebase
- reintroducing previously rejected approaches
- touching the wrong module because ownership is unclear

## Example Files

- [project-memory/schema.sql](project-memory/schema.sql)
- [project-memory/retrieval-notes.md](project-memory/retrieval-notes.md)
- [docs/system-map.md](docs/system-map.md)
- [docs/module-ownership-map.md](docs/module-ownership-map.md)
- [docs/interface-catalog.md](docs/interface-catalog.md)
- [docs/failure-memory-index.md](docs/failure-memory-index.md)

