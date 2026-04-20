---
name: phase-development
description: Phase 기반 개발 워크플로우 스킬. 현재 Phase 문서 로드, 작업 가이드, 진행률 추적을 지원합니다.
---

# Phase-Based Development Skill

프로젝트의 Phase 기반 개발을 지원하는 스킬입니다.

## Activation

다음 상황에서 사용:
- 개발 세션 시작 시
- Phase 작업 수행 시
- 진행 상태 확인 시
- Task 완료 처리 시

## Phase Structure

```
docs/phases/
├── phase-1/
│   ├── SPEC.md
│   ├── TASKS.md
│   └── CHECKLIST.md
├── phase-2/
│   ├── SPEC.md
│   ├── TASKS.md
│   └── CHECKLIST.md
└── ...
```

## Development Workflow

### 1. Session Start

```markdown
## 개발 세션 시작

1. Load: docs/CONTEXT.md
2. Check: docs/PROGRESS.md → 현재 Phase 확인
3. Load: docs/phases/phase-N/SPEC.md → 작업 범위
4. Load: docs/phases/phase-N/TASKS.md → 작업 목록
5. Start: 첫 번째 ⬜ task
```

### 2. Task Execution

```markdown
## Task 작업 순서

1. TASKS.md에서 ⬜ task 선택
2. SPEC.md에서 상세 내용 확인
3. 코드 구현
4. TASKS.md 상태 → ✅ 업데이트
5. CHECKLIST.md 관련 항목 체크
```

### 3. Session End

```markdown
## 세션 종료 시

1. 완료 tasks → TASKS.md 업데이트
2. 진행 중 tasks → 상태 메모
3. PROGRESS.md 갱신
4. CONTEXT.md 현재 상태 업데이트
```

## Context Loading Strategy

### Current Phase Focus
```yaml
Essential (Always Load):
  - docs/CONTEXT.md
  - docs/PROGRESS.md
  - docs/phases/phase-N/SPEC.md
  - docs/phases/phase-N/TASKS.md

On-Demand:
  - docs/phases/phase-N/CHECKLIST.md
  - Previous phase CHECKLIST.md (for dependencies)

Exclude:
  - Completed phase SPEC.md (unless referenced)
  - Future phase documents
```

### Token Optimization
| Session Type | Load | Tokens |
|--------------|------|--------|
| Quick check | CONTEXT + PROGRESS | ~2K |
| Development | + Current phase | ~5K |
| Full context | + Related phases | ~10K |

## Task Management

### Status Flow
```
⬜ Not Started
    ↓ Start work
🔄 In Progress
    ↓ Complete work
✅ Complete
```

### Priority Order
1. **P0** (Critical): Must complete first
2. **P1** (High): Important for phase
3. **P2** (Medium): Nice to have

### Dependency Handling
```
Task B depends on Task A
→ Complete Task A first
→ Then start Task B
```

## Phase Transition

### Completion Criteria
1. All TASKS.md items ✅
2. All CHECKLIST.md items ✅
3. Build passes
4. Documentation updated

### Transition Steps
```
1. Verify phase-N CHECKLIST complete
2. Update PROGRESS.md → Phase N ✅
3. Set Phase N+1 → 🔄 In Progress
4. Load Phase N+1 documents
```

## Integration

### With phase-tracker agent
- Auto-calculates progress
- Manages phase transitions

### With context-optimizer
- Phase-aware context loading
- Token-efficient development

### With phase-progress hook
- Auto-updates on task completion
- Maintains document consistency

### With doc-splitter agent
- Creates phase folder structure
- Maintains cross-references

## Best Practices

1. **One Task at a Time**: Focus on single task
2. **Update Immediately**: Mark complete when done
3. **Check Dependencies**: Verify prerequisites before starting
4. **Document Blockers**: Log issues in TASKS.md
5. **Session Continuity**: Update CONTEXT.md for next session
