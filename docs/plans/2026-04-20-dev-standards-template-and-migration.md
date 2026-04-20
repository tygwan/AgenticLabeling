# Dev Standards Template And Migration Design

## Objective

Translate the new `dev-standards` architecture into something adoptable:

- repository skeleton
- baseline templates
- activation model
- migration path from older standards or ad hoc project conventions

## Design Principle

Templates should make adoption easier, not force one exact workflow onto every project.

That means:

- lightweight defaults
- explicit optionality
- profile and adapter composition
- no hidden dependence on one specific tool or personal environment

## Repository Skeleton

Recommended repository structure for the redesigned `dev-standards` project:

```text
dev-standards/
├── README.md
├── STANDARDS.md
├── CHANGELOG.md
├── core/
│   ├── C1-scope-and-applicability.md
│   ├── C2-repository-contract.md
│   ├── C3-configuration-and-environment.md
│   ├── C4-change-management.md
│   ├── C5-decision-records.md
│   ├── C6-problem-tracking.md
│   ├── C7-dependency-management.md
│   ├── C8-verification.md
│   ├── C9-interfaces-and-contracts.md
│   ├── C10-observability.md
│   ├── C11-security-and-risk.md
│   └── C12-documentation-and-discoverability.md
├── profiles/
├── adapters/
├── templates/
├── examples/
└── migration/
```

## Template Philosophy

Templates should exist at three levels.

## 1. Repository-level templates

These help new projects start with minimum discoverability and structure.

Examples:

- minimal README
- minimal docs index
- contribution notes
- environment example file

## 2. Rule-level templates

These help teams adopt specific core rules.

Examples:

- decision record template
- problem record template
- dependency inventory template
- verification checklist template

## 3. Composition templates

These show what a project gets when it activates specific profiles and adapters.

Examples:

- core only
- core + incident / operations
- core + research / data / ML + Python
- core + AI-assisted development + GitHub + CI

## Minimum Template Set

The first release of the redesigned standard should include:

1. minimal project README template
2. project docs index template
3. decision record template
4. problem tracking template
5. dependency inventory template
6. verification checklist template
7. exception record template

These are enough to make the standard operational without overcommitting to one exact project shape.

## Example Strategy

Examples should prove adaptability, not just completeness.

Recommended example set:

1. small library or CLI
2. service or API project
3. research or data-heavy project
4. internal tool with GitHub + CI

Each example should show:

- which profiles are active
- which adapters are active
- how the core appears in practice

## Migration Philosophy

Migration should not require a project to fully reorganize itself immediately.

The goal is adoption by convergence, not adoption by disruption.

## Migration Levels

### Level 1. Minimal adoption

Project adopts:

- the core principles
- minimum discoverability
- a small set of key templates

No major directory rewrite required.

### Level 2. Structured adoption

Project adopts:

- the core rules explicitly
- selected profiles
- selected adapters
- a stable project knowledge layout

Some repository cleanup likely required.

### Level 3. Full adoption

Project adopts:

- the full core
- declared profiles
- declared adapters
- standard templates and project metadata

Useful for greenfield or heavily maintained long-lived projects.

## Migration Guide Contents

The future migration guide should answer:

1. what a project must add first
2. what existing documents can remain unchanged
3. what can be wrapped instead of rewritten
4. how to classify old practices as core, profile, adapter, or local convention
5. how to phase adoption over time

## Suggested Migration Process

1. Adopt the universal core principles first
2. Add a minimal discoverability layer
3. Declare active profiles
4. Declare active adapters
5. Replace old templates gradually
6. Rewrite only the artifacts that truly block alignment

## Anti-patterns

Avoid these migration mistakes.

### 1. Full rewrite before agreement

Do not rewrite a repository around the standard before the taxonomy and core are accepted.

### 2. Template absolutism

Do not force every project into one exact documentation tree.

### 3. Tool lock-in

Do not make template adoption depend on one assistant, one issue tracker, or one CI provider.

### 4. Core inflation

Do not move profile-heavy or adapter-heavy practices back into the core just because they are useful somewhere.

## Immediate Next Step

The next useful step is to turn the draft rule set into:

- a normalized `STANDARDS.md`
- a first minimal `templates/` starter pack
- a migration guide outline
