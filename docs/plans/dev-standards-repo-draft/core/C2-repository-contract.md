# C2 — Repository Contract

## Rule

Every project must maintain a minimum repository contract so contributors can reliably find code, configuration, documentation, scripts, and generated outputs.

The core standard does not mandate one exact directory tree. It requires discoverability and clear ownership.

## Why

A repository is the primary operational surface of a project. If layout and ownership are unclear, onboarding, maintenance, and recovery all get slower.

## Required minimum

- A root README or equivalent entry point
- A clear location for source code
- A clear location for documentation or project knowledge
- A clear location for automation or helper scripts
- Clear handling of generated artifacts and ignored files

## Application guidance

- Keep committed artifacts intentional
- Keep generated files either reproducible or explicitly tracked
- Avoid dumping unrelated material at the root
- Document non-obvious layout decisions

## Examples

Good:

- "Source lives under `src/`, docs under `docs/`, scripts under `scripts/`."
- "Generated outputs are untracked and recreated by documented commands."

Bad:

- Root full of unlabeled scripts, outputs, notebooks, and temporary files
- Critical project knowledge living only in chat or local notes

## Out of scope

- One mandatory repository shape
- One mandatory documentation filename
- Monorepo vs polyrepo decisions

