---
name: agile-sync
description: Synchronize all agile development artifacts in one command. Updates CHANGELOG, README stats, progress tracking, and validates documentation completeness.
---

# Agile Sync Skill

One-command synchronization of all agile development artifacts. Ensures documentation, progress tracking, and project state are always up-to-date.

## Usage

```bash
/agile-sync [--full|--quick|--validate]
```

### Options

| Option | Description |
|--------|-------------|
| `--full` | Complete sync: changelog + readme + progress + validation |
| `--quick` | Quick sync: readme stats + recent changes only |
| `--validate` | Validate only: check for inconsistencies without changes |
| (default) | Standard sync: changelog + readme + progress |

## Workflow

### Step 1: Analyze Git State
```bash
# Get recent commits since last sync
git log --oneline -10

# Check for uncommitted changes
git status --porcelain

# Get current branch
git branch --show-current
```

### Step 2: Update CHANGELOG.md
```bash
# Analyze commits not yet in changelog
# Group by type: feat, fix, docs, refactor, test, chore

# Generate entries:
## [Unreleased]
### Added
- feat commits

### Fixed
- fix commits

### Changed
- refactor commits

### Documentation
- docs commits
```

### Step 3: Sync README Stats
```bash
# Count components
agents_count=$(find .claude/agents -name "*.md" | wc -l)
skills_count=$(find .claude/skills -name "SKILL.md" -o -name "*.md" | wc -l)
hooks_count=$(find .claude/hooks -name "*.sh" -o -name "*.md" | wc -l)
commands_count=$(find .claude/commands -name "*.md" | wc -l)

# Update README.md Stats section
| Category | Count |
|----------|-------|
| Agents | $agents_count |
| Skills | $skills_count |
| Hooks | $hooks_count |
| Commands | $commands_count |
```

### Step 4: Update Progress Tracking (Phase-Integrated)
```bash
# Read from Phase system (standardized location)
# Primary: docs/PROGRESS.md
# Source: docs/phases/phase-*/TASKS.md

# Scan all phase TASKS.md files
for phase_dir in docs/phases/phase-*/; do
    tasks_file="$phase_dir/TASKS.md"
    if [[ -f "$tasks_file" ]]; then
        total=$(grep -c "^- \[" "$tasks_file" 2>/dev/null || echo 0)
        done=$(grep -c "^- \[x\]\|^- \[X\]\|✅" "$tasks_file" 2>/dev/null || echo 0)
    fi
done

# Calculate overall progress
# Update docs/PROGRESS.md with progress bar

# Progress format:
# [████████████░░░░░░░░] 60% (Phase 2 of 5)
```

> **Note**: Uses Phase system. Legacy `docs/progress/status.md` is deprecated.

### Step 5: Validate Documentation
```bash
# Check for:
# - Missing CLAUDE.md
# - Outdated README sections
# - Broken internal links
# - Missing required files

# Output validation report
```

### Step 6: Generate Sync Report
```markdown
## 📊 Agile Sync Report

**Synced at**: YYYY-MM-DD HH:MM
**Branch**: main

### Changes Made
- ✅ CHANGELOG.md: Added 3 entries
- ✅ README.md: Updated stats (Agents: 15→16)
- ✅ Progress: Updated to 65%
- ⚠️ docs/api.md: Missing (suggested)

### Recommendations
- [ ] Create docs/api.md
- [ ] Update CLAUDE.md with new patterns
- [ ] Run /sprint status for sprint progress
```

## Output Examples

### Standard Sync
```
📊 AGILE-SYNC: Starting synchronization...

[1/5] Analyzing git state...
      Branch: feature/new-feature
      Uncommitted: 2 files
      Recent commits: 5

[2/5] Updating CHANGELOG.md...
      ✅ Added 3 new entries
      - feat(auth): add OAuth support
      - fix(api): resolve timeout issue
      - docs: update README

[3/5] Syncing README stats...
      ✅ Updated component counts
      - Agents: 15 → 16 (+1)
      - Skills: 13 → 14 (+1)

[4/5] Updating progress tracking...
      ✅ Progress: 58% → 62%
      - Completed: 12 → 14 tasks

[5/5] Validating documentation...
      ✅ All checks passed

📊 AGILE-SYNC: Complete!
   Duration: 2.3s
   Changes: 4 files updated
```

### Validation Mode
```
📊 AGILE-SYNC: Validation Report

✅ CHANGELOG.md: Up to date
✅ README.md: Stats accurate
⚠️ Progress: 3 tasks marked complete but no commit found
❌ docs/architecture.md: Referenced but missing

Issues found: 2
Recommendations: 2
```

## Integration

### With Other Skills
```bash
# After implementing a feature
/agile-sync

# Before creating PR
/agile-sync --full

# Quick check before commit
/agile-sync --validate
```

### With Sprint Management
```bash
# Sync includes sprint progress when active
/sprint status  # Shows current sprint
/agile-sync     # Includes sprint metrics in sync
```

### With Hooks
```bash
# Auto-triggered after:
# - git commit (via auto-doc-sync hook)
# - File changes in .claude/

# Manual trigger for full sync
/agile-sync --full
```

## Configuration

### settings.json
```json
{
  "agile": {
    "auto_changelog": true,
    "auto_readme_sync": true,
    "sprint_tracking": true,
    "velocity_tracking": true,
    "sync_on_commit": true
  }
}
```

### Customization
```yaml
# .claude/agile-config.yml
sync:
  changelog:
    enabled: true
    group_by_type: true
    include_scope: true

  readme:
    update_stats: true
    update_badges: false
    sections:
      - Stats
      - Quick Start

  progress:
    source: docs/progress/status.md
    auto_calculate: true
    format: "progress_bar"

  validation:
    check_links: true
    check_required_files: true
    required_files:
      - README.md
      - CLAUDE.md
      - CHANGELOG.md
```

## Best Practices

### DO
- ✅ Run `/agile-sync` before PR creation
- ✅ Run `/agile-sync --validate` before major releases
- ✅ Keep CHANGELOG.md under version control
- ✅ Use conventional commits for accurate changelog

### DON'T
- ❌ Skip validation before releases
- ❌ Manually edit generated sections
- ❌ Ignore sync warnings

## Troubleshooting

### "CHANGELOG.md not found"
```bash
# Create initial CHANGELOG
/agile-sync  # Will create automatically
```

### "README stats section not found"
```bash
# Add Stats section to README.md:
## Stats

| Category | Count |
|----------|-------|
| Agents | 0 |
| Skills | 0 |
```

### "Progress tracking failed"
```bash
# Create progress file
mkdir -p docs/progress
touch docs/progress/status.md
/agile-sync
```

## Related Skills

| Skill | Purpose |
|-------|---------|
| `/sprint` | Sprint lifecycle management |
| `/readme-sync` | Detailed README synchronization |
| `/changelog` | Manual changelog management |
| `/doc-validate` | Comprehensive doc validation |
