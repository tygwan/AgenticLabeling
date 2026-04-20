# AgenticLabeling Project Baseline Checklist

Use this checklist before and after major changes.

## Product boundary

- MVP runtime is still the default path
- legacy code is not being edited accidentally
- the target modules are known before edits start

## Configuration and paths

- no machine-specific absolute paths were introduced
- environment variables remain the override path for runtime configuration
- repository-relative defaults still resolve correctly

## Contracts

- changed routes or config values were identified explicitly
- `docs/navigability/interface-catalog.md` was updated if needed
- workflow expectations were not changed silently

## Verification

- relevant tests were run or updated
- behavior-critical changes were not left as manual-only verification
- health and review flow expectations are still valid

## Discoverability

- navigability docs still point to the right edit locations
- a new non-obvious decision was documented if one was made
- a recurring failure was recorded if one was discovered

## UI-phase reminder

- UI work should not skip product boundary checks
- UI work should not move first into legacy services
- UI work should declare whether it changed workflow, contract, or only presentation

