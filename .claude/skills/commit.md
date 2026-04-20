---
name: commit
description: Git commit workflow with Conventional Commits format. Analyzes changes, generates commit message, handles staging.
---

# Commit Skill

## Usage
```
/commit [--type <type>] [--scope <scope>]
```

### Parameters
- `--type`: feat | fix | docs | style | refactor | test | chore
- `--scope`: Optional scope (e.g., auth, api, ui)

### Examples
```bash
/commit
/commit --type feat --scope auth
/commit --type fix
```

## Conventional Commits Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | feat(auth): add OAuth2 login |
| `fix` | Bug fix | fix(api): handle null response |
| `docs` | Documentation | docs: update README |
| `style` | Formatting | style: fix indentation |
| `refactor` | Code restructure | refactor(utils): extract helper |
| `test` | Tests | test(auth): add login tests |
| `chore` | Maintenance | chore: update dependencies |
| `perf` | Performance | perf(db): optimize query |
| `ci` | CI/CD | ci: add GitHub Actions |
| `build` | Build system | build: update webpack config |

### Scope Examples
- `auth` - Authentication
- `api` - API layer
- `ui` - User interface
- `db` - Database
- `config` - Configuration

## Workflow

### Step 1: Analyze Changes
```bash
git status
git diff --staged
git diff
```

### Step 2: Determine Type
Based on changes:
- New functionality → `feat`
- Bug fix → `fix`
- Only .md files → `docs`
- Tests added → `test`
- Code cleanup → `refactor`
- Dependencies → `chore`

### Step 3: Identify Scope
From changed files:
- `src/auth/*` → scope: auth
- `src/api/*` → scope: api
- Multiple areas → no scope

### Step 4: Generate Message
```markdown
## Changes Detected

### Modified Files
- src/auth/login.ts (+45, -12)
- src/auth/logout.ts (+8, -2)

### Analysis
- Type: feat (new functionality)
- Scope: auth (auth directory)
- Breaking: No

### Suggested Commit
```
feat(auth): implement session management

- Add session token generation
- Implement token refresh logic
- Add logout with token invalidation

Closes #42
```
```

### Step 5: Execute Commit
```bash
git add -A
git commit -m "feat(auth): implement session management

- Add session token generation
- Implement token refresh logic
- Add logout with token invalidation

Closes #42"
```

## Breaking Changes

For breaking changes, add `!` and footer:

```
feat(api)!: change response format

BREAKING CHANGE: Response now returns { data, meta } instead of raw data.
Migration: Wrap existing handlers with responseAdapter().
```

## Best Practices

1. **Atomic Commits**: One logical change per commit
2. **Present Tense**: "add feature" not "added feature"
3. **Imperative**: "fix bug" not "fixes bug"
4. **No Period**: Don't end subject with period
5. **50/72 Rule**: Subject ≤50 chars, body ≤72 per line

## Special Cases

### Multiple Types
Split into multiple commits:
```bash
/commit  # First: fix the bug
/commit  # Second: add tests for the fix
```

### Work in Progress
```
chore(wip): save progress on feature X

NOT READY FOR REVIEW
- Basic structure done
- Need to add validation
```

### Revert
```
revert: feat(auth): add OAuth2 login

This reverts commit abc1234.
Reason: Breaks legacy clients.
```

## Output

```markdown
## Commit Created

**Hash**: abc1234
**Type**: feat
**Scope**: auth
**Message**: implement session management

### Files Committed
- src/auth/login.ts
- src/auth/logout.ts

### Stats
- 2 files changed
- 53 insertions
- 14 deletions
```
