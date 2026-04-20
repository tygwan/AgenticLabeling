---
name: review
description: Code review workflow. Reviews code quality, security, performance. Use for PR reviews or code audits.
---

# Code Review Skill

## Usage
```
/review [target] [--focus <area>]
```

### Parameters
- `target`: File, directory, or git diff
- `--focus`: security | performance | quality | all (default: all)

### Examples
```bash
/review src/auth.ts --focus security
/review ./api --focus performance
/review HEAD~3  # Review last 3 commits
/review  # Review staged changes
```

## Review Areas

### Security Review
```bash
# Check for vulnerabilities
Grep: "eval\(|innerHTML|exec\("
Grep: "password|secret|token|key" -i
Grep: "http://|ftp://"  # Non-HTTPS
Grep: "SELECT.*\+|INSERT.*\+"  # SQL injection
```

**Checklist**:
- [ ] Input validation
- [ ] Output encoding
- [ ] Authentication checks
- [ ] Authorization checks
- [ ] Sensitive data handling
- [ ] Dependency security

### Performance Review
```bash
# Check for issues
Grep: "for.*for|while.*while"  # Nested loops
Grep: "new.*new.*new"  # Object creation in loops
Grep: "setTimeout|setInterval"  # Timer usage
Grep: "sync|Sync"  # Synchronous operations
```

**Checklist**:
- [ ] Algorithm complexity
- [ ] Memory management
- [ ] Database queries (N+1)
- [ ] Caching opportunities
- [ ] Async operations
- [ ] Resource cleanup

### Quality Review
```bash
# Check for smells
Grep: "TODO|FIXME|HACK|XXX"
Grep: "console\.|print\(|System\.out"
wc -l {file}  # Check file length
```

**Checklist**:
- [ ] Naming conventions
- [ ] Function length (<20 lines)
- [ ] File length (<300 lines)
- [ ] Error handling
- [ ] Code duplication
- [ ] Test coverage

## Workflow

### Step 1: Scope Definition
```bash
# For git changes
git diff --name-only [base]

# For directory
Glob: {target}/**/*.{ts,js,py}
```

### Step 2: Critical Files First
Priority order:
1. Authentication/Authorization
2. Data handling
3. API endpoints
4. Business logic
5. Utilities

### Step 3: Generate Report
```markdown
## Code Review Report

**Scope**: src/api/
**Focus**: All areas
**Date**: YYYY-MM-DD

### Summary
| Area | Score | Issues |
|------|-------|--------|
| Security | 8/10 | 2 |
| Performance | 7/10 | 3 |
| Quality | 9/10 | 1 |

### Critical Issues
- [SECURITY] Hardcoded credentials in config.ts:23

### Warnings
- [PERF] N+1 query in users.ts:45

### Suggestions
- [QUALITY] Consider extracting helper function

### Positive Notes
- Good error handling throughout
- Comprehensive input validation
```

## Severity Levels

| Level | Symbol | Action Required |
|-------|--------|-----------------|
| Critical | :rotating_light: | Block merge |
| Warning | :warning: | Should fix |
| Suggestion | :bulb: | Consider |
| Note | :memo: | FYI |

## Output

Review generates:
1. Summary table with scores
2. Issue list by severity
3. Specific file:line references
4. Suggested fixes with code examples
