# Dev Standards Hi-Fi Redesign

## Goal

Redesign `dev-standards` as a high-fidelity, topic-agnostic engineering standard set that can be applied across products, platforms, teams, and project subjects without assuming a specific domain, language, stack, runtime, delivery model, toolchain, or working style.

The standard should remain useful even when the project subject is completely different from the examples we happen to know today.

That means the standard must constrain engineering behavior, not domain content.

Reference reviewed:

- External source repository: `https://github.com/tygwan/dev-standards`

## Design Constraint

The old repository is reference material, not the template that the new standard must preserve.

The new standard may:

- keep ideas
- merge ideas
- discard ideas
- replace them completely

The redesign should start from first principles:

`What standards remain useful across almost any development project, regardless of topic?`

## What The Existing Repository Gets Right

The current repository has real strengths that should influence the redesign.

### 1. Engineering memory matters

Projects become hard to understand over time unless decisions, problems, and important context remain reconstructable.

This is a legitimate engineering concern, not a documentation preference.

### 2. Reproducibility matters

Claims about system behavior, fixes, and outcomes should be verifiable.

### 3. Reversible change matters

Destructive work should be explicit, cautious, and recoverable.

### 4. Dependency awareness matters

Modern projects are shaped by systems outside themselves. Standards should make those dependencies visible.

## What Must Change

The old repository mixes too many different layers into one standard set:

1. universal engineering rules
2. strong but context-specific workflow practices
3. tool-specific conventions
4. public writing guidance

That makes it harder to adopt as a true universal standard.

## The New Structure

The redesigned repository should be split into three layers.

## Layer 1. Core Standards

These are universal rules.

A rule belongs in the core only if it remains useful across most kinds of development work, including cases we have not anticipated.

The test is strict:

- Would this still help if the project had no web UI?
- Would this still help if the project had no API?
- Would this still help if the project had no ML or data pipeline?
- Would this still help in a library, service, CLI, integration, automation, research prototype, or internal platform?
- Would this still help in a project whose subject we did not predict?

If the answer is not "yes" in most of those cases, the rule does not belong in the core.

## Layer 2. Optional Practice Profiles

These are strong practices, but not universal.

Profiles may be strict. They just should not be mandatory for every project.

Examples:

- research / data / ML
- AI-assisted development
- incident / operations
- public writing

## Layer 3. Tool Adapters

Tool-specific conventions live here, not in the core.

Examples:

- GitHub
- Claude Code
- Python
- TypeScript
- CI providers

Adapters help adoption without polluting the universal standard.

## Proposed Universal Core

Recommended core catalog:

1. Scope And Applicability
2. Repository Contract
3. Configuration And Environment
4. Change Management
5. Decision Records
6. Problem Tracking
7. Dependency Management
8. Verification
9. Interfaces And Contracts
10. Observability
11. Security And Risk
12. Documentation And Discoverability

## What The Core Should Govern

The core should govern engineering behavior in these areas.

### Scope And Applicability

The standard must define:

- what it applies to
- what it does not prescribe
- how exceptions are handled

### Repository Contract

The standard must define:

- the minimum repository contract
- where stable project knowledge lives
- where code, docs, scripts, and generated outputs belong

### Configuration And Environment

The standard must define:

- how configuration is expressed
- how environment differences are handled
- what must never be hardcoded

### Change Management

The standard must define:

- how changes are grouped
- how risky changes are reviewed
- how reversibility is preserved
- how verification attaches to change

### Decision Records

The standard must define:

- when a decision must be recorded
- what a decision record must contain
- how decisions remain discoverable

### Problem Tracking

The standard must define:

- what counts as a tracked problem
- what minimum impact and state information exists
- how resolution links back to code or artifacts

### Dependency Management

The standard must define:

- how critical dependencies are identified
- how version or ownership information is tracked
- how dependency change is reviewed

### Verification

The standard must define:

- what evidence is required before work is considered complete
- what minimum verification should exist
- how regressions are prevented after a fix

### Interfaces And Contracts

The standard must define:

- how interfaces are documented or discoverable
- how validation and errors are represented
- how compatibility is handled

### Observability

The standard must define:

- what runtime signals should exist
- what logs or health information are required
- how developers or operators diagnose failure

### Security And Risk

The standard must define:

- how secrets are handled
- how unsafe inputs are treated
- how risky operations are controlled
- how trust in external systems is bounded

### Documentation And Discoverability

The standard must define:

- what information must be easy to find
- how key project knowledge remains accessible over time
- how onboarding and recovery are supported

## What The Core Should Not Govern

These should not be universal core requirements:

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

## Design Principles

### Principle 1. Core standards must be low-regret

If a rule is not valuable to most teams, it should not be a core MUST.

### Principle 2. Standards should constrain behavior, not over-script form

The standard should define what must be true, not always one exact file name, one exact process shape, or one exact tool.

### Principle 3. Tool-specific guidance must be separable

Anything tied to one AI assistant, one hosting provider, or one workflow tool should be detachable.

### Principle 4. Runtime quality matters as much as documentation quality

A high-fidelity engineering standard must include verification, observability, security, and contract stability, not just journaling.

### Principle 5. Topic-agnostic means more than technology-agnostic

A universal standard must still make sense across very different kinds of work:

- products
- platforms
- internal tools
- automations
- libraries
- research systems
- operational systems

So the core should standardize how engineering is done, not what kind of system is being built.

## Concrete Recommendation

The redesign should not begin by migrating old rules.

It should begin by:

1. defining the universal core from scratch
2. separating optional profiles from the core
3. separating tool adapters from the core
4. using older rules only as reference afterward

The new repository should be:

`core engineering standards + optional practice profiles + tool adapters`

The core should be authored from first principles, not from migration pressure.

## Suggested Repository Layout

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
│   ├── research-data-ml.md
│   ├── ai-assisted-development.md
│   ├── incident-operations.md
│   └── public-writing.md
├── adapters/
│   ├── github/
│   ├── claude-code/
│   ├── python/
│   ├── typescript/
│   └── ci/
└── templates/
```

## Next Step

The next useful step is to write the first draft of the 12 universal core rules from scratch.
