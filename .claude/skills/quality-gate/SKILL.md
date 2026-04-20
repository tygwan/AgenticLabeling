---
name: quality-gate
description: Automated quality gates for development lifecycle. Pre-commit checks, pre-merge validation, release readiness verification, and post-release documentation.
---

# Quality Gate Skill

Automated quality validation at critical development checkpoints. Ensures code quality, documentation completeness, and release readiness.

## Usage

```bash
/quality-gate <checkpoint> [options]
```

### Checkpoints

| Checkpoint | Trigger | Purpose |
|------------|---------|---------|
| `pre-commit` | Before commit | Code quality, lint, format |
| `pre-merge` | Before PR merge | Tests, review, docs |
| `pre-release` | Before release | Full validation |
| `post-release` | After release | Documentation, changelog |
| `check` | On demand | Run all applicable checks |

## Quality Gate Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY GATE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Code Change → Pre-Commit → Pre-Merge → Pre-Release → Release  │
│       ↓            ↓            ↓            ↓           ↓     │
│   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐   │
│   │ Edit │ →  │ Lint │ →  │ Test │ →  │ Docs │ →  │ Tag  │   │
│   │      │    │Format│    │Review│    │ Sec  │    │Deploy│   │
│   └──────┘    └──────┘    └──────┘    └──────┘    └──────┘   │
│                                                                 │
│  Gate Status: ✅ Pass   ⚠️ Warning   ❌ Block                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Checkpoints Detail

### /quality-gate pre-commit

Run before committing code changes.

```bash
/quality-gate pre-commit [--fix] [--skip <check>]
```

**Checks:**
| Check | Description | Blocking |
|-------|-------------|:--------:|
| `lint` | Code linting | ✅ |
| `format` | Code formatting | ⚠️ |
| `types` | Type checking | ✅ |
| `secrets` | Secret detection | ✅ |
| `size` | File size limits | ⚠️ |

**Output:**
```
🔍 PRE-COMMIT QUALITY GATE

[1/5] Linting...
      ✅ No lint errors

[2/5] Formatting...
      ⚠️ 2 files need formatting
      → Run with --fix to auto-format

[3/5] Type Checking...
      ✅ No type errors

[4/5] Secret Detection...
      ✅ No secrets found

[5/5] File Size Check...
      ✅ All files under limit

━━━━━━━━━━━━━━━━━━━━━━━━━
Result: ⚠️ PASS WITH WARNINGS

Warnings: 1
- Format: 2 files need formatting

Proceed with commit? (Y/n)
```

### /quality-gate pre-merge

Run before merging PR to main branch.

```bash
/quality-gate pre-merge [--pr <number>]
```

**Checks:**
| Check | Description | Blocking |
|-------|-------------|:--------:|
| `tests` | All tests pass | ✅ |
| `coverage` | Test coverage threshold | ⚠️ |
| `review` | Code review approved | ✅ |
| `conflicts` | No merge conflicts | ✅ |
| `docs` | Documentation updated | ⚠️ |
| `changelog` | CHANGELOG updated | ⚠️ |

**Output:**
```
🔍 PRE-MERGE QUALITY GATE

PR: #42 - Add user authentication

[1/6] Tests...
      ✅ 127 tests passed

[2/6] Coverage...
      ⚠️ Coverage: 72% (threshold: 80%)
      New code coverage: 85%

[3/6] Code Review...
      ✅ Approved by: @reviewer

[4/6] Merge Conflicts...
      ✅ No conflicts

[5/6] Documentation...
      ⚠️ README.md not updated
      Consider: /readme-sync

[6/6] Changelog...
      ✅ CHANGELOG.md updated

━━━━━━━━━━━━━━━━━━━━━━━━━
Result: ⚠️ PASS WITH WARNINGS

Blocking: 0
Warnings: 2
- Coverage below threshold
- Documentation may need update

Recommend: Run /agile-sync before merge
```

### /quality-gate pre-release

Comprehensive validation before release.

```bash
/quality-gate pre-release --version <semver>
```

**Checks:**
| Check | Description | Blocking |
|-------|-------------|:--------:|
| `tests` | Full test suite | ✅ |
| `coverage` | Coverage threshold | ✅ |
| `lint` | Zero lint errors | ✅ |
| `security` | Security scan | ✅ |
| `docs` | Documentation complete | ✅ |
| `changelog` | Version in changelog | ✅ |
| `version` | Version consistency | ✅ |
| `dependencies` | No vulnerable deps | ⚠️ |
| `build` | Build succeeds | ✅ |

**Output:**
```
🔍 PRE-RELEASE QUALITY GATE

Version: v1.2.0

[1/9] Full Test Suite...
      ✅ 342 tests passed (0 failed, 0 skipped)

[2/9] Coverage Analysis...
      ✅ Coverage: 84% (threshold: 80%)
      - Statements: 86%
      - Branches: 79%
      - Functions: 88%
      - Lines: 84%

[3/9] Lint Check...
      ✅ Zero lint errors

[4/9] Security Scan...
      ✅ No vulnerabilities found
      Scanned: dependencies, code patterns

[5/9] Documentation...
      ✅ All required docs present
      - README.md: ✅
      - CHANGELOG.md: ✅
      - API docs: ✅

[6/9] Changelog Version...
      ✅ v1.2.0 entry found

[7/9] Version Consistency...
      ✅ Version matches across:
      - package.json: 1.2.0
      - CHANGELOG.md: 1.2.0

[8/9] Dependency Audit...
      ⚠️ 1 low severity issue
      → lodash: prototype pollution (low)

[9/9] Build Verification...
      ✅ Build successful
      Size: 2.3 MB (within limit)

━━━━━━━━━━━━━━━━━━━━━━━━━
Result: ✅ RELEASE READY

All blocking checks passed!

Warnings: 1
- Low severity dependency issue

Next Steps:
1. Create release tag: git tag v1.2.0
2. Push tag: git push origin v1.2.0
3. Run: /quality-gate post-release
```

### /quality-gate post-release

Documentation and tracking after release.

```bash
/quality-gate post-release --version <semver>
```

**Actions:**
| Action | Description |
|--------|-------------|
| Archive sprint | Close active sprint if any |
| Update docs | Update version references |
| Velocity | Record release velocity |
| Notify | Generate release notes |
| Retro prompt | Suggest retrospective |

**Output:**
```
📦 POST-RELEASE QUALITY GATE

Version: v1.2.0 released!

[1/5] Sprint Archive...
      ✅ Sprint 3 closed
      Velocity: 34 points

[2/5] Documentation Update...
      ✅ Version references updated
      - README.md: badge updated
      - docs/index.md: version updated

[3/5] Velocity Recording...
      ✅ Release velocity recorded
      - Features: 8
      - Fixes: 5
      - Docs: 3

[4/5] Release Notes...
      ✅ Generated: RELEASE-v1.2.0.md

[5/5] Retrospective...
      💡 Consider running: /feedback retro --milestone v1.2.0

━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Post-release tasks complete!

Generated:
- docs/releases/RELEASE-v1.2.0.md
- Updated: docs/feedback/VELOCITY.md
```

### /quality-gate check

Run applicable checks based on current state.

```bash
/quality-gate check [--all] [--fix]
```

Automatically detects:
- Uncommitted changes → pre-commit checks
- Open PR → pre-merge checks
- Release branch → pre-release checks

## Configuration

### settings.json
```json
{
  "quality-gate": {
    "pre-commit": {
      "enabled": true,
      "checks": ["lint", "format", "types", "secrets"],
      "auto-fix": false
    },
    "pre-merge": {
      "enabled": true,
      "coverage-threshold": 80,
      "require-review": true,
      "require-changelog": true
    },
    "pre-release": {
      "enabled": true,
      "coverage-threshold": 80,
      "security-scan": true,
      "require-all-docs": true
    },
    "post-release": {
      "auto-archive-sprint": true,
      "generate-release-notes": true,
      "prompt-retro": true
    }
  }
}
```

### Custom Checks

Add project-specific checks:

```yaml
# .claude/quality-checks.yml
custom-checks:
  pre-commit:
    - name: "API Schema"
      command: "npm run validate-schema"
      blocking: true

  pre-release:
    - name: "License Check"
      command: "npm run license-check"
      blocking: true

    - name: "Bundle Size"
      command: "npm run analyze-bundle"
      threshold: "5MB"
      blocking: false
```

## Integration

### With Git Hooks
```bash
# .git/hooks/pre-commit
#!/bin/bash
/quality-gate pre-commit --fail-on-warning
```

### With CI/CD
```yaml
# GitHub Actions
- name: Quality Gate
  run: |
    /quality-gate pre-merge
    /quality-gate pre-release --version ${{ github.ref_name }}
```

### With Other Skills
```bash
# Before PR
/quality-gate pre-merge && /agile-sync

# Release workflow
/quality-gate pre-release --version v1.2.0
git tag v1.2.0
git push origin v1.2.0
/quality-gate post-release --version v1.2.0
```

## Check Reference

### Lint Commands by Language
| Language | Command |
|----------|---------|
| JavaScript/TypeScript | `eslint .` |
| Python | `ruff check .` or `flake8` |
| Go | `golint ./...` |
| Rust | `cargo clippy` |
| C# | `dotnet format --verify-no-changes` |

### Test Commands by Framework
| Framework | Command |
|-----------|---------|
| Jest | `npm test -- --coverage` |
| pytest | `pytest --cov` |
| Go | `go test -cover ./...` |
| .NET | `dotnet test --collect:"XPlat Code Coverage"` |

### Security Scanners
| Tool | Purpose |
|------|---------|
| `npm audit` | Node.js dependencies |
| `safety` | Python dependencies |
| `trivy` | Container images |
| `gitleaks` | Secret detection |

## Best Practices

### DO
- ✅ Run pre-commit before every commit
- ✅ Require pre-merge for all PRs
- ✅ Run full pre-release before tags
- ✅ Document skipped checks with reason
- ✅ Fix warnings before they become blockers

### DON'T
- ❌ Skip security scans
- ❌ Ignore coverage drops
- ❌ Release without pre-release check
- ❌ Disable blocking checks permanently

## Troubleshooting

### "Check failed but code is correct"
```bash
# Skip specific check with reason
/quality-gate pre-commit --skip lint --reason "false positive in generated code"
```

### "Coverage dropped below threshold"
```bash
# View uncovered lines
/quality-gate pre-merge --coverage-details
```

### "Security scan timeout"
```bash
# Run with extended timeout
/quality-gate pre-release --timeout 600
```

## Related Skills

| Skill | Purpose |
|-------|---------|
| `/agile-sync` | Sync all agile artifacts |
| `/sprint` | Sprint management |
| `/feedback` | Post-release retrospective |
| `/test` | Test execution |
