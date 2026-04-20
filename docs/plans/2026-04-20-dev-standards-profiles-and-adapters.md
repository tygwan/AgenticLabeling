# Dev Standards Profiles And Adapters

## Objective

Define the boundary between:

- universal core standards
- optional practice profiles
- tool or ecosystem adapters

The purpose of this split is to keep the core small, strong, and topic-agnostic while still allowing stricter guidance where context justifies it.

## Core vs Profile vs Adapter

## Core

Core rules are universal.

They answer:

- what every serious project should do
- regardless of subject
- regardless of stack
- regardless of tool choice

Core rules govern engineering behavior at a high enough level to survive major context changes.

## Profile

Profiles are context-heavy practice packs.

They answer:

- what a certain kind of project should additionally do
- when that project shape creates recurring risks or obligations

Profiles are stricter than the core, but they are activated only when relevant.

Examples:

- research / data / ML
- AI-assisted development
- incident / operations
- public writing

## Adapter

Adapters are implementation bridges to specific tools, ecosystems, or platforms.

They answer:

- how the standards are applied in a specific operational environment

Examples:

- GitHub
- Claude Code
- Project Memory
- Python
- TypeScript
- CI systems

Adapters should never redefine the core. They only operationalize it.

## Decision Tests

Use these tests to classify a rule.

## Test 1. Universality

If removing the rule would make most projects weaker, it belongs in the core.

## Test 2. Context dependency

If the rule matters strongly only in a recurring project shape, it belongs in a profile.

## Test 3. Tool dependency

If the rule only makes sense because a team uses a specific tool, vendor, platform, or language ecosystem, it belongs in an adapter.

## Test 4. Replaceability

If the same intent could be expressed through multiple tools, the intent belongs in core or profile, while the concrete implementation belongs in an adapter.

## What Profiles Are For

Profiles exist to prevent two bad outcomes:

1. making the core too weak to be useful
2. making the core too specific to stay universal

Profiles should be activated when the project shape justifies stricter behavior.

## Proposed Profiles

## P1. Research / Data / ML

Use when the project depends heavily on:

- experiments
- benchmarks
- data lineage
- reproducibility-heavy claims
- model evaluation or dataset integrity

Likely additions:

- stricter experiment records
- audit scripts
- artifact lineage
- benchmark discipline
- evidence-heavy findings

## P2. AI-Assisted Development

Use when AI systems are active collaborators in engineering work.

Likely additions:

- structural decision checkpoints
- explicit human approval rules
- AI contribution attribution guidance
- session bootstrap or memory guidance
- scoped work contexts
- structured project memory
- selective retrieval guidance

The core should not assume AI tooling. This profile should.

## P3. Incident / Operations

Use when the system is actively run, deployed, or operationally supported.

Likely additions:

- incident record format
- postmortem expectations
- rollback and mitigation rules
- service-level health and escalation expectations

## P4. Public Writing

Use when the project needs outward-facing communication artifacts.

Likely additions:

- portfolio case study structure
- technical blog structure
- artifact presentation guidance

This profile stays outside the universal engineering core.

## What Adapters Are For

Adapters translate standards into concrete ecosystems.

## Proposed Adapter Families

### Tool adapters

- GitHub
- Claude Code
- Jira
- Notion

### Language or ecosystem adapters

- Python
- TypeScript
- Rust
- JVM

### Delivery adapters

- CI systems
- container build systems
- release automation environments

## Adapter Rules

Adapters must follow these constraints.

1. They may add concrete implementation guidance.
2. They may not contradict the core.
3. They may tighten operational practice for that tool.
4. They may not redefine the meaning of a core rule.
5. They should stay replaceable.

If an adapter contains guidance that would still make sense without the tool, that guidance likely belongs in the core or a profile instead.

## Activation Model

The repository should support explicit activation.

### Core

Always active.

### Profile

Active only when the project declares that profile.

### Adapter

Active only when the project uses that tool or ecosystem.

This lets a team compose:

- one universal core
- zero or more profiles
- zero or more adapters

without losing clarity.

## Example Composition

Example A:

- core only
- useful for a small library or internal tool

Example B:

- core + incident / operations + GitHub + CI
- useful for a deployed backend service

Example C:

- core + research / data / ML + AI-assisted development + Python
- useful for model or dataset work

The composition changes. The core remains stable.

## Immediate Next Step

Write first-pass drafts for:

- the four proposed profiles
- the adapter design contract
- a small starter set of adapters
