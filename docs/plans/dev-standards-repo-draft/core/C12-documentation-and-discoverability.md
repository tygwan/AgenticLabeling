# C12 — Documentation And Discoverability

## Rule

Projects must keep essential knowledge discoverable so a contributor can recover context, continue work, and diagnose current state without relying on private memory.

## Why

Code alone does not preserve intent, context, risk, or operational meaning. Discoverability is a durability requirement.

## Required minimum

- A clear entry point for project understanding
- Discoverable location for key decisions and important operational knowledge
- Discoverable explanation of how to run, verify, or operate the project
- Enough context for a new contributor to orient without asking hidden experts first

## Application guidance

- Optimize for fast orientation, not documentation volume
- Keep critical knowledge in versioned project artifacts, not only in chat
- Separate durable project knowledge from temporary notes
- Update discoverability artifacts when project shape changes

## Examples

Good:

- Readme plus discoverable design and operational notes
- Contributor can answer "what is this, how do I run it, what is risky, what is unresolved?"

Bad:

- Project state recoverable only from oral tradition or message history
- Important operational assumptions hidden in local setup steps

## Out of scope

- One exact documentation tree
- Mandatory long-form journaling for every task
- Public-facing writing style requirements

