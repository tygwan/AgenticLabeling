# P2 — AI-Assisted Development Profile

## Use when

Activate this profile when AI systems are active collaborators in engineering work, not merely external end-user features.

## Purpose

This profile defines how human judgment and AI acceleration coexist without losing accountability, clarity, or reviewability.

## Adds on top of core

- explicit human checkpoints for structural or risky decisions
- expectation that alternatives and trade-offs are surfaced before major implementation choices
- guidance for AI contribution attribution where useful
- project-level session bootstrap guidance where AI tools depend on project context
- scoped work-context guidance so AI workers do not operate against the whole codebase by default
- project memory guidance for decisions, failures, techniques, and constraints
- retrieval guidance that prefers high-value structured artifacts over raw transcript replay
- navigability artifacts that help both humans and AI find where to read and where to edit

## Why this is not core

The universal standard must not assume one AI workflow, one AI tool, or AI use at all.

## Typical activation signals

- AI is used for code generation or design support
- AI proposes architectural or workflow changes
- teams need a repeatable human-in-the-loop standard
- teams rely on retrieval or memory systems to avoid replaying full project history
- large codebases need scoped editing contexts for safe AI-assisted work

## Recommended operating model

When this profile is active, teams should prefer:

1. bounded work contexts
- define what subsystem an AI worker is allowed to read and modify
- avoid full-repository reasoning unless the task truly requires it

2. structured project memory
- store decisions
- store failures
- store techniques
- store constraints

3. selective retrieval
- prefer summaries over transcripts
- prefer edits and outcomes over passive reads
- prefer tagged memory over raw logs

4. human checkpoints for structural work
- architecture
- repository structure
- destructive operations
- dependency commitments

## Suggested memory categories

- Decision
- Failure
- Technique
- Constraint
- Open Question

## Typical non-goals

- prescribing one AI tool
- prescribing one memory file convention
- replacing human ownership of decisions
- requiring vector search or transcript databases for every project

## Suggested supporting artifacts

This profile benefits from navigability artifacts such as:

- system map
- module ownership map
- interface catalog
- failure memory index

These artifacts should help answer:

- where should I read first?
- where should I edit?
- what already exists?
- what previously failed?
