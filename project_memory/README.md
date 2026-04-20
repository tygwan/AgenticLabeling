# AgenticLabeling Project Memory

This directory contains a minimal SQLite-backed project memory store and the base schema used to record:

- decisions
- failures
- implementation techniques
- cross-cutting project notes

## Why this exists

AgenticLabeling already has a mixed codebase:

- current MVP monolith in `mvp_app/`
- legacy microservices in `services/`
- Docker and runtime scripts in `scripts/`
- historical design and planning docs in `docs/`

The project memory store is intended to reduce re-discovery cost. It is not a general-purpose vector database or a production incident platform. It is a scoped engineering memory for this repository.

## Default database path

By default the database lives at:

`data/project-memory/project-memory.db`

You can override it with `PROJECT_MEMORY_DB_PATH`.

## Commands

Initialize the store:

```bash
python3 -m project_memory.memory_store init
```

Add an entry:

```bash
python3 -m project_memory.memory_store add-entry \
  --category decision \
  --module-scope mvp_app \
  --title "Use the monolith as the default runtime" \
  --summary "The MVP app replaced the microservice compose as the default entrypoint." \
  --content "The MVP path is now the default because it reduces Docker complexity and makes review/export easier to maintain." \
  --tags mvp,architecture,docker \
  --source-kind doc \
  --source-ref docs/plans/2026-04-20-mvp-refactor-plan.md \
  --change-kind architecture
```

Query entries:

```bash
python3 -m project_memory.memory_store query \
  --module-scope mvp_app \
  --category decision \
  --limit 10
```

Free-text query:

```bash
python3 -m project_memory.memory_store query --text sam2
```

## Recommended categories

- `decision`
- `failure`
- `technique`
- `context`

## Recommended module scopes

- `mvp_app`
- `project_memory`
- `scripts`
- `tests`
- `services/object-registry`
- `services/label-studio-lite`
- `docker`
- `docs`

## When to record a new entry

- a design decision affects module boundaries
- a recurring bug was found and fixed
- a workaround should not be rediscovered later
- a feature requires non-obvious edit locations
- a migration changes the default runtime path

## What not to do

- do not dump raw transcripts
- do not store every read-only observation
- do not store secrets or credentials
- do not use this as a replacement for source control history

