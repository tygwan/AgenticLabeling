# Migration Guide

## Goal

Adopt the new `dev-standards` model by convergence, not by forcing a full repository rewrite.

## Migration Levels

### Level 1. Minimal adoption

Do this first:

- adopt the universal core principles
- add a README or equivalent entry point
- add a docs index or equivalent discoverability layer
- create a decision record template and a problem record template

### Level 2. Structured adoption

Then:

- identify active profiles
- identify active adapters
- establish a dependency inventory
- establish a verification checklist pattern

### Level 3. Full adoption

Finally:

- normalize project documentation against the new core
- retire conflicting local conventions
- document exception cases explicitly

## Suggested Migration Sequence

1. Read the core catalog in `STANDARDS.md`
2. Decide which profiles actually apply
3. Decide which adapters actually apply
4. Install only the minimum starter templates
5. Wrap existing project artifacts before rewriting them
6. Rewrite only the parts that block alignment

## What Not To Do

- do not rewrite the whole repo before agreeing on active rules
- do not copy every template blindly
- do not move tool-specific guidance into the universal core
- do not make adoption depend on one assistant or one vendor

## Migration Questions

For each existing project, answer:

- what stable project knowledge already exists?
- what is missing for discoverability?
- what current practices are local conventions rather than standards?
- which profiles are actually justified?
- which adapters are actually needed?

## Minimal Success Criteria

A project is meaningfully migrated when:

- contributors can find the project entry point
- important decisions are discoverable
- meaningful problems are tracked
- critical dependencies are visible
- verification expectations are explicit
- active profiles and adapters are declared

