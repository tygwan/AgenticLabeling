# COMPETENCY

This document is a companion to [STANDARDS.md](/home/coffin/dev/AgenticLabeling/docs/plans/dev-standards-publishable-package/STANDARDS.md).

`STANDARDS.md` defines what engineering behavior must be true.  
`COMPETENCY.md` defines how deeply a contributor or team can apply those standards.

It is intentionally universal. It does not assume a specific programming language, architecture, subject area, or product category.

## Levels

### Foundation

The contributor can apply a standard safely within a local scope.

- follows existing structure instead of improvising
- uses configuration and existing contracts correctly
- runs and interprets the relevant verification steps
- avoids obvious duplication, hardcoding, and undocumented behavior changes

### Intermediate

The contributor can apply a standard across modules and make sound structural tradeoffs.

- reasons about change impact across boundaries
- evolves interfaces carefully
- records decisions and problem history with useful scope
- chooses verification depth based on risk rather than habit

### Advanced

The contributor can shape systems so the standard becomes easier for others to follow.

- redesigns boundaries, workflows, and operating models
- introduces reusable observability, verification, and migration patterns
- balances delivery, maintainability, and risk
- turns project-specific practice into reusable standards or templates

## Core Rule Mapping

### C1 Scope And Applicability

- Foundation: knows when the standard applies
- Intermediate: knows when a profile or adapter is needed
- Advanced: can define new standard boundaries cleanly

### C2 Repository Contract

- Foundation: finds the right module and edits with local discipline
- Intermediate: improves repository structure and reduces discovery cost
- Advanced: reshapes repository boundaries for long-term maintainability

### C3 Configuration And Environment

- Foundation: avoids hardcoded values and uses configuration correctly
- Intermediate: structures layered configuration for multiple environments
- Advanced: designs safe configuration strategy across local, CI, and production contexts

### C4 Change Management

- Foundation: makes scoped changes and verifies them
- Intermediate: manages multi-file changes with explicit impact reasoning
- Advanced: defines rollout and migration strategies for risky changes

### C5 Decision Records

- Foundation: records non-obvious decisions
- Intermediate: distinguishes temporary decisions from durable ones
- Advanced: designs decision-record systems that scale

### C6 Problem Tracking

- Foundation: records recurring failures and known issues
- Intermediate: links incidents, causes, and remediation work
- Advanced: designs failure-learning loops

### C7 Dependency Management

- Foundation: adds dependencies intentionally
- Intermediate: evaluates upgrade and compatibility risk
- Advanced: designs maintainable dependency strategy

### C8 Verification

- Foundation: runs relevant checks and understands the result
- Intermediate: chooses verification depth based on risk
- Advanced: designs regression strategy and verification architecture

### C9 Interfaces And Contracts

- Foundation: respects and updates contracts safely
- Intermediate: evolves contracts with compatibility in mind
- Advanced: designs contract strategy and migration paths

### C10 Observability

- Foundation: adds local diagnostic signals where needed
- Intermediate: defines useful signals across workflows
- Advanced: shapes an operating model where observability supports engineering decisions

### C11 Security And Risk

- Foundation: avoids obvious unsafe handling of secrets, inputs, and permissions
- Intermediate: evaluates operational risk introduced by a change
- Advanced: designs proportionate security and risk controls

### C12 Documentation And Discoverability

- Foundation: leaves enough information for the next contributor
- Intermediate: maintains navigability artifacts across modules
- Advanced: designs documentation systems that scale with the codebase

## Use Cases

Use this model for:

- onboarding
- review calibration
- engineering ladders
- AI-assisted development guardrails
- training and mentorship design

Do not use it as:

- a language-specific curriculum
- a framework tutorial
- a rigid scoring system without context

