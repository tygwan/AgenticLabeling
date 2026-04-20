# Cross-CLI Trigger Design

## Goal

When `dev-standards` is applied to a repository, both Codex and Claude Code should discover the same project records without relying on one vendor-specific file format alone.

For AgenticLabeling, this means:

- shared bootstrap path
- shared navigability docs
- shared local project memory
- shared change gates

## Trigger layers

### Layer 1. CLI-specific entry files

These are discovery hooks, not primary knowledge stores.

- `AGENTS.md`
- `CLAUDE.md`

Their job is to point to the same shared bootstrap document.

### Layer 2. Shared bootstrap document

This repository uses:

- `docs/agents/START-HERE.md`

This is the canonical cross-CLI entrypoint.

### Layer 3. Shared project records

These are the actual durable repository records:

- `docs/navigability/`
- `docs/standards/`
- `project_memory/`

### Layer 4. Queryable memory

Structured recall should not depend only on markdown search.

This repository uses:

- `python3 -m project_memory.memory_store query ...`

That gives both humans and coding agents a common retrieval path.

## Why this design is needed

Without this split:

- Codex may see `AGENTS.md` but miss Claude-specific notes
- Claude Code may see `CLAUDE.md` but miss Codex-specific workflow
- project history may be duplicated across tool-specific files
- repository knowledge will drift

With this split:

- CLI-specific files are thin wrappers
- shared records stay in repository-owned docs and memory
- behavior is reproducible across tools

## Repository rule

Do not put the only important context in `AGENTS.md` or `CLAUDE.md`.

Put durable knowledge in shared repository artifacts, and use the tool-specific files only to route agents into that shared context.

