# C11 — Security And Risk

## Rule

Projects must handle secrets, external trust, unsafe inputs, and risky operations deliberately.

Security and risk controls should scale with system risk, but they cannot be absent from the core standard.

## Why

Security failures are often engineering failures in configuration, validation, assumptions, and operational discipline.

## Required minimum

- No committed secrets
- Explicit handling of untrusted input
- Explicit review of risky or destructive operations
- Clear trust boundaries for external systems

## Application guidance

- Treat all external input as potentially unsafe until validated
- Keep secret rotation and storage outside normal source code paths
- Make destructive actions harder to trigger accidentally
- Record trust assumptions that affect design or operations

## Examples

Good:

- Uploaded files validated before use
- Secrets loaded from secure runtime config
- Dangerous operations gated by review or confirmation

Bad:

- API keys committed to source control
- External payloads trusted without validation
- Irreversible data deletion done casually

## Out of scope

- One security framework
- One compliance regime
- One threat model methodology

