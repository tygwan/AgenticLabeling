# C7 — Dependency Management

## Rule

Projects must identify and track critical dependencies that meaningfully affect build, runtime, data correctness, operational behavior, or delivery.

## Why

Most systems are shaped by external components. Hidden dependency assumptions make maintenance, upgrades, and incident response harder.

## Required minimum

- Critical dependency list or equivalent discoverable record
- Ownership or source for each critical dependency
- Version or compatibility expectations where relevant
- A known path for upgrades, issues, or replacement

## Application guidance

- Focus on dependencies with real operational impact
- Record why a dependency matters, not just its name
- Treat model weights, external APIs, SaaS, and platform services as dependencies too
- Make upgrade decisions explicit rather than accidental

## Examples

Good:

- "Depends on external auth provider, versioned schema, and deployment runtime."
- "Model checkpoint source and compatibility expectation recorded."

Bad:

- Team cannot answer which third-party systems are critical
- Upgrade happens implicitly through drift with no review

## Out of scope

- One package manager
- One SBOM format
- Tracking every transitive dependency manually

