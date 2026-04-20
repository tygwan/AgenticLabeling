# STANDARDS

## Purpose

`dev-standards` defines a universal engineering standard that can apply to almost any development project without assuming a specific topic, stack, architecture, or toolchain.

The standard is designed to remain useful across changing contexts by standardizing engineering behavior rather than domain content.

## Repository Model

The standard is organized into three layers:

- `core`: universal rules
- `profiles`: optional context-heavy practice packs
- `adapters`: tool or ecosystem specific application guides

This structure keeps the core small, strict, and broadly reusable.

## Core Rules

The universal core consists of twelve rules.

1. **Scope And Applicability**
   Defines where the standard applies and what it intentionally does not prescribe.
2. **Repository Contract**
   Defines the minimum discoverable structure of a project repository.
3. **Configuration And Environment**
   Prevents hardcoded runtime assumptions and hidden machine-specific behavior.
4. **Change Management**
   Requires coherent, reviewable, and reversible change.
5. **Decision Records**
   Preserves structural engineering decisions and their rationale.
6. **Problem Tracking**
   Ensures defects, risks, limitations, and unresolved questions remain visible.
7. **Dependency Management**
   Makes critical external dependencies explicit and maintainable.
8. **Verification**
   Requires evidence that meaningful work has been checked.
9. **Interfaces And Contracts**
   Treats important boundaries as first-class engineering assets.
10. **Observability**
   Requires enough runtime visibility to diagnose health and failure.
11. **Security And Risk**
   Requires explicit handling of secrets, unsafe inputs, and risky actions.
12. **Documentation And Discoverability**
   Keeps essential project knowledge accessible over time.

## Universality Test

A rule belongs in the core only if it remains useful:

- across project subjects
- across technology stacks
- across team sizes
- across delivery models
- across tool choices

If a rule is only strong in a recurring project shape, it belongs in a profile.
If a rule only makes sense because of a specific tool, it belongs in an adapter.

## Profiles

Profiles add stricter guidance for recurring project shapes.

The initial profile set is:

- `research-data-ml`
- `ai-assisted-development`
- `incident-operations`
- `public-writing`

Profiles are optional. They may be strict when active.

## Adapters

Adapters operationalize the standard in specific tools or ecosystems.

The initial adapter set is:

- `github`
- `claude-code`
- `project-memory`
- `python`
- `typescript`
- `ci`

Adapters may tighten implementation details but must not redefine the core.

## What The Standard Does Not Mandate

The universal standard does not mandate:

- one programming language
- one framework
- one architecture style
- one branching strategy
- one testing framework
- one deployment model
- one documentation tree
- one AI tool

Those are implementation choices or context-specific practices.

## Adoption Model

A project adopts the standard by composing:

- the universal core
- zero or more optional profiles
- zero or more adapters

Examples:

- `core`
- `core + incident-operations + github + ci`
- `core + research-data-ml + ai-assisted-development + python`

## Starter Templates

The minimal starter pack includes:

- repository README template
- docs index template
- decision record template
- problem record template
- dependency inventory template
- verification checklist template
- exception record template
- system map template
- module ownership map template
- interface catalog template
- failure memory index template

## Migration Model

Projects should migrate by convergence, not by full rewrite.

Recommended path:

1. adopt the universal core principles
2. add minimum discoverability artifacts
3. activate relevant profiles
4. activate relevant adapters
5. replace local conventions only where necessary

## Package Index

- [core/README.md](core/README.md)
- [profiles/README.md](profiles/README.md)
- [adapters/README.md](adapters/README.md)
- [templates/README.md](templates/README.md)
- [migration/MIGRATION-GUIDE.md](migration/MIGRATION-GUIDE.md)
