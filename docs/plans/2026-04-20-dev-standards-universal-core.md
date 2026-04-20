# Dev Standards Universal Core

## Objective

Define a development standard that can apply to almost any project topic without assuming:

- a specific domain
- a specific technology family
- a specific architecture
- a specific delivery model
- a specific team shape
- a specific toolchain

The standard should remain useful whether the project is:

- a product
- a platform
- an internal tool
- a library
- an integration
- an automation system
- a research prototype
- an operational service

## First Principle

A universal standard does not prescribe what subject a team works on.

It prescribes how engineering work should remain:

- understandable
- safe to change
- verifiable
- diagnosable
- maintainable over time

That is the right abstraction boundary for a truly domain-independent standard.

## Universality Test

A rule belongs in the universal core only if it survives these questions.

### Q1. Subject independence

Would the rule still make sense even if the project subject changed completely?

### Q2. Stack independence

Would the rule still make sense if the project had no web UI, no API, no ML, and no cloud dependency?

### Q3. Scale independence

Would the rule still help both a small tool and a larger system?

### Q4. Team-shape independence

Would the rule still help a solo developer, a small team, and a larger organization?

### Q5. Tool independence

Would the rule still make sense if the team did not use the current tools, vendors, or AI assistants?

If a rule fails most of these questions, it should not be a universal core rule.

## Universal Core Rule Areas

The core should govern engineering behavior in these areas.

## 1. Scope And Applicability

The standard must define:

- what kinds of projects it applies to
- what it intentionally does not prescribe
- how exceptions are handled

## 2. Repository Contract

The standard must define:

- the minimum repository contract
- where stable project knowledge lives
- where code, docs, scripts, and generated outputs belong

## 3. Configuration And Environment

The standard must define:

- how configuration is expressed
- how environment differences are handled
- what must never be hardcoded

## 4. Change Management

The standard must define:

- how changes are grouped
- how risky changes are reviewed
- how reversibility is preserved
- how verification is attached to change

## 5. Decision Records

The standard must define:

- when a decision must be recorded
- what minimum information a decision record must contain
- how decisions remain discoverable later

## 6. Problem Tracking

The standard must define:

- what counts as a tracked problem
- what minimum status and impact information exists
- how resolution links back to code or artifacts

## 7. Dependency Management

The standard must define:

- how critical dependencies are identified
- how version or ownership information is tracked
- how dependency change is reviewed

## 8. Verification

The standard must define:

- what evidence is required before work is considered complete
- what minimum verification should exist
- how regressions are prevented after a fix

## 9. Interfaces And Contracts

The standard must define:

- how interfaces are documented or discoverable
- how validation and errors are represented
- how compatibility is handled

## 10. Observability

The standard must define:

- what runtime signals should exist
- what logs or health information are required
- how operators or developers diagnose failure

## 11. Security And Risk

The standard must define:

- how secrets are handled
- how unsafe inputs are treated
- how risky operations are controlled
- how trust in external systems is bounded

## 12. Documentation And Discoverability

The standard must define:

- what information must be easy to find
- how key project knowledge remains accessible over time
- how onboarding and recovery are supported

## What The Universal Core Should Not Govern

These do not belong in the universal core.

- portfolio writing
- blog writing
- one exact docs tree for every team
- one exact issue archive format for every project
- one AI tool's memory folder conventions
- one branch strategy
- one testing framework
- one architecture style
- one deployment model

These may belong in profiles or adapters, but not in the universal core.

## Profiles

Profiles are for practices that are strong but not universal.

Examples:

- research / data / ML
- AI-assisted development
- incident / operations
- public writing

Profiles may be strict. They just are not universal.

## Adapters

Adapters are where tool-specific guidance belongs.

Examples:

- GitHub
- Claude Code
- Python
- TypeScript
- CI

Adapters help adoption without polluting the core standard.

## Authoring Rule

When writing a new core rule, do not ask:

`Did the old repository already have something like this?`

Ask instead:

`Would a competent engineering team want this rule even if they had never seen the old repository?`

That is the inclusion test.

## Immediate Next Step

The next step is to write the first draft of the 12 universal core rules from scratch.
