# C9 — Interfaces And Contracts

## Rule

Projects must make important interfaces discoverable and keep their behavioral contracts understandable.

An interface may be an API, CLI, event payload, file format, schema, library boundary, or any other stable interaction surface.

## Why

Most costly failures happen at boundaries. Stable teams treat contracts as first-class engineering assets.

## Required minimum

- Discoverable definition of important interfaces
- Validation rules or expectations
- Error behavior or failure mode expectations
- Compatibility policy where interface evolution matters

## Application guidance

- Prefer stable field names and explicit semantics
- Document what callers can rely on
- Distinguish internal experimental interfaces from supported ones
- Make breaking changes deliberate and visible

## Examples

Good:

- API request and response schema
- CLI arguments and exit behavior
- Dataset or file format contract

Bad:

- Consumers infer contract only by reading implementation details
- Errors vary arbitrarily with no consistent structure

## Out of scope

- One schema format
- One API versioning style
- One transport protocol

