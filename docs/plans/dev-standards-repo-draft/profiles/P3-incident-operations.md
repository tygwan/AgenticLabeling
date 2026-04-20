# P3 — Incident / Operations Profile

## Use when

Activate this profile when the project is deployed, operated, or relied on in a way that makes runtime failures operationally significant.

## Purpose

This profile strengthens operational discipline beyond the core observability and risk baseline.

## Adds on top of core

- incident record expectations
- mitigation and rollback tracking
- postmortem structure for significant failures
- explicit prevention or follow-up actions after severe incidents

## Why this is not core

Not every project has operational incidents in the same sense. A small library or prototype may not need full incident discipline.

## Typical activation signals

- uptime or reliability matters
- end users or internal operators depend on the system
- failures need coordinated diagnosis and response

## Typical non-goals

- forcing full SRE process onto every project
- mandating one severity ladder or on-call model

