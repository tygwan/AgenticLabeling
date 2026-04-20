---
name: phase-progress
description: Phase 진행 상태 자동 업데이트 훅. Task 완료 시 PROGRESS.md와 CHECKLIST.md를 자동으로 업데이트합니다.
event: PostToolUse
---

# Phase Progress Hook

Task 상태 변경 시 자동으로 진행률을 업데이트하는 훅입니다.

## Trigger Conditions

다음 조건에서 활성화:
1. `docs/phases/phase-*/TASKS.md` 파일 수정 시
2. Task 상태가 `⬜` → `✅`로 변경 시
3. Phase 관련 작업 완료 시

## Actions

### 1. Progress Calculation
```
When: TASKS.md modified
Action:
  1. Count total tasks
  2. Count completed tasks (✅)
  3. Calculate percentage
  4. Update PROGRESS.md
```

### 2. Checklist Update
```
When: All tasks in category complete
Action:
  1. Find corresponding CHECKLIST.md item
  2. Update status to checked
```

### 3. Phase Transition Alert
```
When: All tasks complete + All checklist items checked
Action:
  1. Alert user about phase completion
  2. Suggest next phase activation
```

## Implementation

```yaml
trigger:
  file_pattern: "docs/phases/phase-*/TASKS.md"
  change_type: "modify"

actions:
  - calculate_progress:
      source: "${modified_file}"
      update: "docs/PROGRESS.md"

  - check_completion:
      tasks_file: "${modified_file}"
      checklist: "${dir}/CHECKLIST.md"

  - notify_if_complete:
      condition: "all_tasks_complete AND all_checks_done"
      message: "Phase ${phase_number} complete! Ready for next phase"
```

## Integration Points

### With phase-tracker agent
- Provides real-time progress data
- Triggers status calculations

### With PROGRESS.md
- Auto-updates progress bars
- Maintains milestone status

### With CHECKLIST.md
- Cross-references task completion
- Validates completion criteria

## Output

### Progress Update Log
```
[Hook] Phase Progress Updated
- Phase: N
- Completed: X/Y tasks
- Progress: Z%
- Updated: docs/PROGRESS.md
```

### Completion Alert
```
[Hook] Phase Completion Detected
- Phase: N
- All tasks: ✅
- All checks: ✅
- Action: Ready for next phase
```

## Configuration

```yaml
phase_progress:
  enabled: true
  auto_update_progress: true
  auto_check_completion: true
  notify_on_completion: true
  log_changes: true
```
