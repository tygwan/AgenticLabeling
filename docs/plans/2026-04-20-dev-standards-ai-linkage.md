# Dev Standards And AI Driven Development Linkage

## Purpose

This document explains how the reconstructed `dev-standards` model should connect logically to the project-management strategy required for AI-driven development.

The question is not whether AI-driven development is important.

The question is:

`Which parts belong in the universal standard, which parts belong in a profile, and which parts belong in adapters or local implementation?`

## Principle

AI-driven development should not redefine the universal engineering core.

Instead:

- universal engineering truths remain in the core
- AI-heavy workflow constraints move into the AI-assisted profile
- tool-specific memory and retrieval implementations move into adapters or local systems

This keeps the standard stable while still making it useful for modern AI-assisted teams.

## Logical Mapping

## 1. "You should not need to read the whole codebase"

This idea maps primarily to the core.

Relevant core rules:

- `C2 Repository Contract`
- `C9 Interfaces And Contracts`
- `C12 Documentation And Discoverability`

Why:

This is not AI-specific. It is a general engineering quality property.

## 2. "Documentation should let you navigate without loading everything"

This also belongs in the core.

Relevant core rules:

- `C5 Decision Records`
- `C6 Problem Tracking`
- `C12 Documentation And Discoverability`

Why:

This is discoverability, not AI specialization.

## 3. "LSP, compiler, tests, and runtime feedback are essential"

This belongs in the core.

Relevant core rules:

- `C8 Verification`
- `C10 Observability`

Why:

External verification is a universal engineering need. AI just makes the need more obvious.

## 4. "Use scoped agents or scoped work contexts"

This belongs partly in the core and partly in the AI-assisted profile.

Core side:

- `C2 Repository Contract`
- `C9 Interfaces And Contracts`

Profile side:

- `P2 AI-Assisted Development`

Why:

The core supports bounded responsibility.
The profile defines how AI workers or agent-like workflows should respect those boundaries.

## 5. "Maintain searchable project memory"

This must be split carefully.

Core side:

- `C5 Decision Records`
- `C6 Problem Tracking`
- `C12 Documentation And Discoverability`

Profile side:

- `P2 AI-Assisted Development`

Adapter or local implementation side:

- retrieval databases
- transcript parsers
- vector indexes
- memory schemas

Why:

The need for durable, searchable project memory is universal.
The concrete implementation with SQLite, vector search, or transcript parsing is not.

## 6. "Store decisions, failures, techniques, constraints"

This maps across core and profile.

Core:

- decisions and problems are universal

Profile:

- explicit AI-facing memory curation belongs in `P2`

Possible future extension:

- a reusable `project-memory` adapter or profile extension

## 7. "Prefer summaries and high-value events over raw transcript replay"

This belongs mostly in the AI-assisted profile and adapters.

Relevant profile:

- `P2 AI-Assisted Development`

Relevant adapters:

- `A2 Claude Code`
- possible future retrieval adapters

Why:

This is highly useful, but it is not a universal requirement for every project.

## What Should Be Added To The Current Standard Set

The reconstructed standard is already structurally compatible with AI-driven development.

But to make that relationship explicit, the next improvements should be:

## A. Strengthen P2 AI-Assisted Development

It should explicitly cover:

- scoped work contexts
- memory categories
- high-value event extraction
- when human approval is required
- how AI should consume project memory artifacts

## B. Add a future Project Memory adapter family

Not yet universal, but increasingly useful.

Potential adapter family:

- `project-memory-sqlite`
- `project-memory-vector-search`
- `project-memory-transcript-parser`

These would describe implementation patterns without polluting the core.

## C. Add module-map and ownership-map templates

These belong in templates because they help both humans and AI navigate large codebases.

Candidate new templates:

- system-map
- module-ownership-map
- interface-catalog
- failure-memory-index

## D. Add "navigability" language to C12

The current discoverability rule is good, but it should more explicitly say:

- a contributor should be able to find where to edit
- a contributor should be able to find what to read first
- a contributor should be able to find what not to duplicate

That would make the core more obviously aligned with AI-driven work without making it AI-specific.

## Recommended Standard Position

The correct position is:

1. AI-driven development does not replace core engineering discipline
2. It increases the importance of discoverability, verification, contracts, and bounded responsibility
3. Its workflow-specific practices belong in an optional profile
4. Its tool-specific memory or retrieval systems belong in adapters or local implementation

That is the cleanest logical connection between the strategy and the reconstructed standards model.

## Immediate Next Step

The next useful step is to fold this linkage back into the publishable package by:

1. expanding `P2 AI-Assisted Development`
2. adding new starter templates for navigability and memory
3. deciding whether `project memory` should become a first-class adapter family

