# COMPETENCY

This document is a companion to [STANDARDS.md](/home/coffin/dev/AgenticLabeling/docs/plans/dev-standards-repo-draft/STANDARDS.md).

The standards define universal engineering behavior.  
The competency model defines how deeply those standards can be applied.

It exists to support onboarding, review depth, engineering ladders, and AI-assisted development without polluting the core rules with educational or stack-specific material.

## Levels

### Foundation

- applies standards safely within local scope
- follows existing contracts and structure
- runs relevant verification
- avoids obvious hardcoding, duplication, and undocumented drift

### Intermediate

- applies standards across module boundaries
- reasons about change impact and interface evolution
- chooses verification based on risk
- records decisions and failures with useful scope

### Advanced

- redesigns systems so standards become easier to follow
- introduces scalable verification, observability, and migration strategies
- balances delivery, maintainability, and operational risk
- turns project-specific practice into reusable patterns

## Mapping

### C1 Scope And Applicability

- Foundation: knows when the standard applies
- Intermediate: knows when a profile or adapter should be used
- Advanced: can define new standard boundaries cleanly

### C2 Repository Contract

- Foundation: edits the right module with local discipline
- Intermediate: improves repository structure and discovery cost
- Advanced: reshapes repository boundaries for long-term maintainability

### C3 Configuration And Environment

- Foundation: uses configuration correctly
- Intermediate: structures layered configuration
- Advanced: designs safe environment strategy across operating contexts

### C4 Change Management

- Foundation: makes scoped changes and verifies them
- Intermediate: manages multi-file impact explicitly
- Advanced: defines rollout and migration strategy

### C5 Decision Records

- Foundation: records non-obvious decisions
- Intermediate: distinguishes temporary from durable decisions
- Advanced: designs scalable decision-record systems

### C6 Problem Tracking

- Foundation: records recurring failures and known issues
- Intermediate: links incidents, causes, and remediation work
- Advanced: designs failure-learning loops

### C7 Dependency Management

- Foundation: adds dependencies intentionally
- Intermediate: evaluates upgrade risk and compatibility
- Advanced: designs maintainable dependency strategy

### C8 Verification

- Foundation: runs relevant checks and interprets results
- Intermediate: chooses verification depth based on risk
- Advanced: designs regression and verification architecture

### C9 Interfaces And Contracts

- Foundation: respects and updates contracts safely
- Intermediate: evolves contracts with compatibility in mind
- Advanced: designs contract strategy and migration paths

### C10 Observability

- Foundation: adds useful local diagnostics
- Intermediate: defines workflow-level signals
- Advanced: shapes an operating model where observability supports engineering decisions

### C11 Security And Risk

- Foundation: avoids obvious unsafe handling
- Intermediate: evaluates risk introduced by changes
- Advanced: designs proportionate security and risk controls

### C12 Documentation And Discoverability

- Foundation: leaves enough information for the next contributor
- Intermediate: maintains navigability artifacts across modules
- Advanced: designs documentation systems that scale with the codebase

