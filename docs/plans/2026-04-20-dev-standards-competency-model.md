# dev-standards Competency Model

## Purpose

`dev-standards` should define universal engineering standards. It should not mix those standards with role ladders, education tracks, or stack-specific training material.

At the same time, teams still need a way to answer:

- what "basic" application of a standard looks like
- what "intermediate" application looks like
- what "advanced" application looks like

This document defines a companion model for that purpose.

## Positioning

The competency model is not part of the core rules.

- `STANDARDS.md` answers: what must be true
- `COMPETENCY.md` answers: how deeply a person or team can apply those standards

This separation keeps the standard universal while still allowing concrete expectations for skill growth, review depth, and onboarding.

## Level definitions

### Foundation

Foundation means the contributor can follow the standard reliably with local scope.

Typical characteristics:

- can make safe changes inside one module or one clear boundary
- can follow existing contracts and configuration patterns
- can run and interpret basic verification steps
- can avoid obvious anti-patterns such as hardcoding, duplication, and undocumented behavior changes

### Intermediate

Intermediate means the contributor can apply the standard across boundaries and make sound structural decisions.

Typical characteristics:

- can reason about module boundaries and change impact
- can evolve interfaces without careless breakage
- can choose verification depth based on risk
- can record and justify architectural or operational decisions

### Advanced

Advanced means the contributor can design or improve systems so that the standard becomes easier for others to follow.

Typical characteristics:

- can redesign boundaries, workflows, and operating models
- can introduce scalable verification, observability, and migration strategies
- can balance delivery, maintainability, risk, and long-term system health
- can turn project-specific practice into reusable standards or templates

## Mapping model

Each core standard should be interpreted at three depths:

1. Foundation: correct local usage
2. Intermediate: correct cross-boundary usage
3. Advanced: system-shaping usage

This makes the standard universal without pretending every contributor should operate at the same level on every topic.

## Core-to-competency mapping

### C1 Scope And Applicability

- Foundation: understands when the standard applies and does not invent ad hoc exceptions
- Intermediate: can identify when a profile or adapter is needed instead of overloading the core
- Advanced: can define boundaries for new standards, profiles, or companion systems

### C2 Repository Contract

- Foundation: can find the right module and avoid editing unrelated areas
- Intermediate: can improve repository structure and reduce discovery cost
- Advanced: can redesign repository boundaries to support long-term maintainability

### C3 Configuration And Environment

- Foundation: uses configuration instead of hardcoded paths and values
- Intermediate: can structure layered configuration for multiple environments
- Advanced: can define safe environment strategy across local, CI, and production contexts

### C4 Change Management

- Foundation: makes scoped changes with clear intent and basic verification
- Intermediate: can manage multi-file changes with explicit impact reasoning
- Advanced: can define safe rollout and migration strategies for risky changes

### C5 Decision Records

- Foundation: records non-obvious decisions and links them to the change
- Intermediate: can distinguish temporary decisions from durable architectural ones
- Advanced: can design ADR and decision-memory processes that scale across teams

### C6 Problem Tracking

- Foundation: records recurring failures and known issues clearly
- Intermediate: can connect incidents, causes, and remediation work across modules
- Advanced: can design failure-memory and learning loops for the organization

### C7 Dependency Management

- Foundation: adds or changes dependencies intentionally and documents why
- Intermediate: can reason about upgrade risk, lockfiles, and dependency boundaries
- Advanced: can design dependency strategy for supply-chain safety and maintainability

### C8 Verification

- Foundation: runs the relevant checks and can explain what passed
- Intermediate: chooses verification based on risk and change surface
- Advanced: can design verification pyramids and regression strategy for the system

### C9 Interfaces And Contracts

- Foundation: respects existing contracts and updates dependent callers safely
- Intermediate: can evolve contracts with compatibility in mind
- Advanced: can design contract strategy, compatibility policy, and migration paths

### C10 Observability

- Foundation: adds logs, health signals, or metrics where local debugging requires them
- Intermediate: can define useful signals across workflow steps and failure boundaries
- Advanced: can shape an operating model where observability supports engineering decisions

### C11 Security And Risk

- Foundation: avoids obvious unsafe handling of inputs, secrets, and permissions
- Intermediate: can evaluate operational risk introduced by a change
- Advanced: can design security and risk controls proportionate to the system

### C12 Documentation And Discoverability

- Foundation: leaves the next contributor enough information to continue safely
- Intermediate: can maintain navigability artifacts and edit guides across modules
- Advanced: can design documentation systems that reduce search cost at scale

## What this model is for

- onboarding expectations
- review depth calibration
- engineering ladder support
- AI-assisted development guardrails
- training design

## What this model is not for

- stack-specific education content
- framework tutorials
- role titles or compensation policy
- rigid scoring of individuals without context

## Recommended repository shape

For the redesigned `dev-standards` repository, this should exist as a companion top-level document:

- `STANDARDS.md`
- `COMPETENCY.md`

The competency model should remain aligned to the core rules, not replace them.

