---
name: sprint
description: Complete sprint lifecycle management. Start sprints, track velocity, generate burndown charts, and automate retrospectives.
---

# Sprint Management Skill

Complete agile sprint lifecycle management. Handles sprint planning, daily tracking, velocity measurement, and retrospective automation.

## Usage

```bash
/sprint <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `start` | Start a new sprint |
| `status` | Show current sprint status |
| `end` | End current sprint and generate retro |
| `add` | Add item to current sprint |
| `complete` | Mark item as complete |
| `velocity` | Show velocity history |
| `burndown` | Generate burndown chart |

## Sprint Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│                    SPRINT LIFECYCLE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  /sprint start     /sprint status      /sprint end           │
│       ↓                  ↓                   ↓               │
│  ┌─────────┐      ┌───────────┐       ┌──────────┐          │
│  │ PLANNING│  →   │ IN PROGRESS│  →    │ COMPLETE │          │
│  │         │      │           │       │          │          │
│  │ • Goals │      │ • Daily   │       │ • Retro  │          │
│  │ • Items │      │ • Burndown│       │ • Velocity│         │
│  │ • Team  │      │ • Blockers│       │ • Report │          │
│  └─────────┘      └───────────┘       └──────────┘          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Commands Detail

### /sprint start

Start a new sprint with planning session.

```bash
/sprint start --name "Sprint 1" --duration 2w --goal "Complete auth module"
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Sprint name | "Sprint N" |
| `--duration` | Sprint duration (1w, 2w, 3w) | 2w |
| `--goal` | Sprint goal | (prompted) |
| `--capacity` | Team capacity in points | (calculated) |

**Creates:**
- `docs/sprints/sprint-{N}/SPRINT.md` - Sprint plan document
- `docs/sprints/sprint-{N}/BACKLOG.md` - Sprint backlog
- `docs/sprints/sprint-{N}/DAILY.md` - Daily standup log

**Output:**
```
🏃 SPRINT: Starting Sprint 1

📋 Sprint Details:
   Name: Sprint 1
   Duration: 2 weeks (Jan 8 - Jan 22)
   Goal: Complete authentication module

📊 Capacity:
   Planned: 40 story points
   Items: 8 tasks

📁 Created:
   - docs/sprints/sprint-1/SPRINT.md
   - docs/sprints/sprint-1/BACKLOG.md
   - docs/sprints/sprint-1/DAILY.md

✅ Sprint started! Use `/sprint add` to add items.
```

### /sprint status

Show current sprint progress with burndown.

```bash
/sprint status [--detailed]
```

**Output:**
```
🏃 SPRINT STATUS: Sprint 1

📅 Timeline:
   Started: Jan 8, 2025
   Ends: Jan 22, 2025
   Day: 5 of 14 (36%)

🎯 Goal: Complete authentication module

📊 Progress:
   [████████░░░░░░░░░░░░] 40% (16/40 points)

   Completed: 4 items (16 pts)
   In Progress: 2 items (8 pts)
   Remaining: 4 items (16 pts)

📉 Burndown:
   Day 1: ████████████████████ 40 pts
   Day 2: ██████████████████░░ 36 pts
   Day 3: ████████████████░░░░ 32 pts
   Day 4: ██████████████░░░░░░ 28 pts
   Day 5: ████████████░░░░░░░░ 24 pts ← Today
   Ideal: ██████████░░░░░░░░░░ 20 pts

⚠️ Status: Slightly behind schedule (-4 pts)

🚧 Blockers:
   - API integration waiting for backend team
```

### /sprint end

End current sprint and generate retrospective.

```bash
/sprint end [--skip-retro]
```

**Actions:**
1. Calculate velocity
2. Move incomplete items to backlog
3. Generate retrospective template
4. Update velocity history
5. Archive sprint documents

**Output:**
```
🏁 SPRINT END: Sprint 1

📊 Results:
   Completed: 32/40 points (80%)
   Items Done: 6/8
   Velocity: 32 pts

📈 Velocity History:
   Sprint -2: 28 pts
   Sprint -1: 30 pts
   Sprint 1:  32 pts ← Current
   Average:   30 pts

📝 Incomplete Items (moved to backlog):
   - [ ] OAuth integration (8 pts)
   - [ ] Password reset UI (4 pts)

📋 Retrospective Generated:
   → docs/sprints/sprint-1/RETRO.md

🔄 Next Sprint:
   Recommended capacity: 30-32 pts (based on velocity)

Continue to retrospective? (Y/n)
```

### /sprint add

Add item to current sprint.

```bash
/sprint add "Implement login form" --points 5 --priority high
```

**Options:**
| Option | Description |
|--------|-------------|
| `--points` | Story points (1, 2, 3, 5, 8, 13) |
| `--priority` | high, medium, low |
| `--assignee` | Team member |

### /sprint complete

Mark sprint item as complete.

```bash
/sprint complete "Implement login form"
# or
/sprint complete --id TASK-001
```

### /sprint velocity

Show velocity trends and predictions.

```bash
/sprint velocity [--chart] [--last N]
```

**Output:**
```
📈 VELOCITY REPORT

Sprint History:
┌────────────┬────────┬────────────┬───────────┐
│ Sprint     │ Points │ Completed  │ Velocity  │
├────────────┼────────┼────────────┼───────────┤
│ Sprint -4  │ 35     │ 28         │ 28        │
│ Sprint -3  │ 40     │ 30         │ 30        │
│ Sprint -2  │ 38     │ 32         │ 32        │
│ Sprint -1  │ 42     │ 35         │ 35        │
│ Sprint 1   │ 40     │ 32         │ 32        │
└────────────┴────────┴────────────┴───────────┘

📊 Statistics:
   Average Velocity: 31.4 pts
   Std Deviation: 2.6 pts
   Trend: ↗️ Improving (+1.4 pts/sprint)

🎯 Recommendations:
   Next Sprint Capacity: 32-34 pts
   Confidence Range: 29-37 pts (95%)
```

### /sprint burndown

Generate ASCII burndown chart.

```bash
/sprint burndown
```

**Output:**
```
📉 BURNDOWN CHART: Sprint 1

Points │
   40  │●
   36  │  ●───────── Ideal
   32  │    ○
   28  │      ○──── Actual
   24  │        ○
   20  │          ●
   16  │            ●
   12  │              ●
    8  │                ●
    4  │                  ●
    0  │____________________●
       └──────────────────────
        1  2  3  4  5  6  7  8  9  10  Days

Legend: ● Ideal, ○ Actual

Status: 🟡 Slightly behind (-4 pts from ideal)
Projection: Complete by Day 11 (1 day delay)
```

## File Structure

```
docs/
└── sprints/
    ├── VELOCITY.md          # Velocity history
    ├── sprint-1/
    │   ├── SPRINT.md        # Sprint plan
    │   ├── BACKLOG.md       # Sprint backlog
    │   ├── DAILY.md         # Daily log
    │   └── RETRO.md         # Retrospective
    ├── sprint-2/
    │   └── ...
    └── current -> sprint-2/  # Symlink to current
```

## Templates

### SPRINT.md Template
```markdown
# Sprint {N}: {Name}

## Overview
- **Duration**: {start_date} - {end_date}
- **Goal**: {sprint_goal}
- **Capacity**: {capacity} points

## Team
| Member | Role | Availability |
|--------|------|--------------|
| {name} | {role} | {%} |

## Sprint Backlog
| ID | Task | Points | Priority | Status |
|----|------|--------|----------|--------|
| T-001 | {task} | {pts} | {pri} | ⏳ |

## Progress
`[░░░░░░░░░░░░░░░░░░░░]` 0% (0/{total} points)

## Daily Log
### Day 1 ({date})
- Started: {items}
- Completed: {items}
- Blockers: {blockers}
```

### RETRO.md Template
```markdown
# Sprint {N} Retrospective

## Summary
- **Completed**: {completed}/{planned} points ({percentage}%)
- **Velocity**: {velocity} points
- **Items**: {completed_items}/{total_items}

## What Went Well 🌟
- {positive_1}
- {positive_2}

## What Could Improve 🔧
- {improve_1}
- {improve_2}

## Action Items 📋
| Action | Owner | Due |
|--------|-------|-----|
| {action} | {owner} | {date} |

## Velocity Trend
{velocity_chart}

## Notes
{additional_notes}
```

## Integration

### With Phase System (Primary Integration)

Sprint는 Phase 시스템과 연동하여 작동합니다:

```
Phase (기능 단위)                Sprint (시간 단위)
    │                                │
    ├── docs/phases/phase-1/         ├── docs/sprints/sprint-1/
    │   └── TASKS.md ─────────────► │   └── BACKLOG.md
    │       (Task Source)            │       (Sprint Items)
    │                                │
    └── CHECKLIST.md ◄─────────────┘
        (자동 업데이트)                  (완료 시)
```

**Phase-Sprint 연동 옵션:**

```bash
# Phase의 Task를 Sprint에 추가
/sprint start --phase 2 --name "Sprint 3"

# Sprint 항목 완료 시 Phase TASKS.md도 자동 업데이트
/sprint complete T2-03
# → docs/phases/phase-2/TASKS.md의 T2-03도 ✅ 표시
```

> **상세 가이드**: `.claude/docs/SPRINT-PHASE-INTEGRATION.md` 참조

### With /agile-sync
```bash
# agile-sync includes sprint + phase progress
/agile-sync  # Updates progress from both sprint and phase data
```

### With Progress Tracking
```bash
# Sprint completion updates project progress
/sprint complete "task"
# → Automatically updates docs/PROGRESS.md (Phase + Sprint 통합)
```

### With Git Workflow
```bash
# Commit message includes sprint reference
git commit -m "feat(auth): login form [Sprint-1][Phase-2]"
```

## Configuration

```json
{
  "sprint": {
    "default_duration": "2w",
    "point_scale": [1, 2, 3, 5, 8, 13],
    "auto_velocity_track": true,
    "auto_retro_generate": true,
    "burndown_chart": "ascii",
    "daily_reminder": true
  }
}
```

## Best Practices

### DO
- ✅ Set clear sprint goals
- ✅ Keep items small (≤8 points)
- ✅ Update daily progress
- ✅ Complete retrospectives
- ✅ Track velocity trends

### DON'T
- ❌ Add items mid-sprint without discussion
- ❌ Skip retrospectives
- ❌ Ignore velocity trends
- ❌ Overcommit beyond velocity

## Related Skills

| Skill | Purpose |
|-------|---------|
| `/agile-sync` | Full agile artifact sync |
| `/progress` | Progress tracking |
| `/retro` | Standalone retrospective |
| `/backlog` | Backlog management |
