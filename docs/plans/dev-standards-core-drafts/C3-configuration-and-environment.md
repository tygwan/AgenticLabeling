# C3 — Configuration And Environment

## Rule

Runtime behavior must be configurable without editing source code for routine environment differences.

Projects must avoid hardcoded machine-specific paths, secrets in code, and undocumented environment assumptions.

## Why

Portable systems fail when they depend on one developer's machine layout or hidden environment state.

## Required minimum

- Configuration values separated from business logic
- Secret values excluded from source control
- Documented runtime defaults
- Documented environment-specific overrides where they exist

## Application guidance

- Use environment variables, versioned config, or equivalent mechanisms
- Keep defaults safe and explicit
- Prefer one configuration entry path over scattered constants
- Make local, test, and production differences understandable

## Examples

Good:

- "Storage path comes from env var with a documented default."
- "API token is read from environment and not committed."

Bad:

- Hardcoded `/home/user/...` path in application code
- Production endpoint embedded directly in multiple source files

## Out of scope

- One specific config library
- One secret manager
- One environment naming scheme

