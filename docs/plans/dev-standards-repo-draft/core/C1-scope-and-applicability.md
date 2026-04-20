# C1 — Scope And Applicability

## Rule

The standard must explicitly define:

- what kinds of development work it applies to
- what it intentionally does not prescribe
- how exceptions are handled

The default stance is broad applicability. If a rule is intended only for a subset of projects, it does not belong in the universal core.

## Why

Standards fail when they silently assume one type of team, one type of product, or one type of stack. A universal standard must say where it applies and where it stops.

## Required minimum

- A statement of applicability
- A statement of non-goals
- A way to classify non-core rules as profiles or adapters
- A rule for documenting exceptions

## Application guidance

- Put universal rules in the core
- Put context-heavy practices in profiles
- Put tool-specific instructions in adapters
- Record deviations when a project intentionally does not follow a core rule

## Examples

Good:

- "This rule applies to any git-tracked software project."
- "This rule does not mandate a programming language, architecture, or deployment model."

Bad:

- "All projects must use one fixed branch strategy."
- "All projects must keep one exact docs tree."

## Out of scope

- Choosing a technology stack
- Choosing an architecture style
- Choosing a team workflow tool

