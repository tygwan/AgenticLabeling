# AI Driven Development Project Strategy

## Purpose

This document extracts practical project-management strategy from the discussion about large codebases, AI assistance, retrieval, and engineering scale.

The goal is not to preserve the exact wording of that discussion.
The goal is to isolate the engineering ideas that remain valid for AI-driven development.

## Core Observation

A codebase should not require a human or an AI to re-read everything before making a safe change.

If every implementation task requires re-reading the whole project, the main problem is usually not line count alone. The real problem is weak boundaries, weak discoverability, weak contracts, and weak verification.

## Strategic Principles

## 1. Optimize for bounded understanding

Good systems let a contributor understand the relevant slice of the project without loading the whole codebase into working memory.

That requires:

- clear module boundaries
- local ownership
- discoverable entry points
- stable contracts between parts

In AI-driven development, bounded understanding is even more important because model context is expensive and noisy.

## 2. Treat documentation as navigation, not prose

The most valuable documentation for AI-driven work is not maximal explanation.

It is:

- system maps
- module responsibility maps
- key data flow descriptions
- decision history
- known failures and limitations
- where-to-edit guidance

This lets both humans and AI find the right code before attempting edits.

## 3. Use external verification as the truth source

AI can propose changes, but it should not be treated as the final verifier of correctness.

The real verification loop is external:

- compiler or typechecker
- linter
- tests
- schema validation
- runtime health checks

In practice, these tools form the objective feedback loop for AI-assisted work.

## 4. Separate contexts by responsibility

Large-codebase work becomes tractable when responsibility is partitioned.

Examples:

- UI changes
- API changes
- data model changes
- tests and verification
- deployment and operations

In AI-driven development, this same idea appears as scoped agents, scoped retrieval, or scoped work contexts.

The principle is not "more agents". The principle is "narrower responsibility per worker".

## 5. Build project memory outside the prompt

A transcript is too large and too noisy to be the main memory substrate.

A better memory model stores distilled project knowledge such as:

- decisions
- failures
- techniques
- constraints
- recurring operational patterns

This memory should be searchable and reusable without replaying entire conversations.

## 6. Preserve failure memory, not just success memory

Projects regress not only because teams forget what worked, but because they forget:

- what failed
- why it failed
- what anti-patterns repeated
- what trade-offs were already rejected

AI-driven development benefits heavily from explicit failure memory because models otherwise tend to re-propose superficially plausible but previously rejected solutions.

## 7. Retrieval must be selective

Do not inject everything.

Retrieval should prefer:

- decisions over raw discussion
- edits over reads
- summaries over entire transcripts
- tagged project memory over raw logs

The point of retrieval is not to maximize context volume. It is to maximize relevance density.

## 8. Code size is not the primary metric

A large codebase is not automatically a bad codebase.

More important metrics are:

- duplication
- search cost
- change radius
- contract clarity
- verification quality
- dependency visibility

AI-driven workflows become unstable when those metrics are poor, even in smaller repositories.

## Project Management Implications

From these principles, the following management strategies follow.

## A. Enforce modular ownership

Each meaningful subsystem should have:

- a bounded responsibility
- a discoverable entry point
- a known contract
- a known verification path

## B. Maintain project memory as structured records

Project memory should be stored as durable, queryable artifacts rather than only in:

- chat
- commit history
- oral context
- local notes

Useful memory categories:

- Decision
- Failure
- Technique
- Constraint
- Open question

## C. Maintain a retrieval layer

A useful retrieval layer should answer:

- have we already solved this?
- did this fail before?
- where is the closest existing implementation?
- what file or module owns this concern?

This can be built with:

- SQLite
- tags
- summaries
- metadata
- optional vector search

The storage engine is secondary. The categorization quality is primary.

## D. Prefer edit-aware memory over transcript replay

Work logs are useful, but high-value memory usually comes from:

- real edits
- decisions
- validation outcomes
- production failures

Reads alone are often weak signals. Write and verification events usually carry more engineering value.

## E. Keep AI aligned to system boundaries

AI should work against:

- module maps
- interface contracts
- dependency inventories
- decision records
- problem records

These artifacts prevent the model from improvising new structure every time.

## Practical Recommendations

For AI-driven development, a project should have at minimum:

1. system overview
2. module ownership map
3. interface or contract references
4. decision record set
5. problem record set
6. verification path per change type
7. searchable project memory

That is enough to make AI useful without requiring full-codebase re-ingestion for every task.

## What This Strategy Is Not

This strategy is not:

- "just use vector search"
- "just split everything into microservices"
- "just add more agents"
- "just document more"

Those can all fail if the underlying project shape is still incoherent.

The real point is:

`make the project navigable, bounded, and externally verifiable`

## Immediate Next Use

This strategy should be connected to the redesigned `dev-standards` model so the standard can explain:

- which parts of AI-driven development belong in the universal core
- which parts belong in the AI-assisted profile
- which parts belong in adapters or implementation-specific memory systems

