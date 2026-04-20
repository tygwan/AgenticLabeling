---
name: context-optimizer
description: Optimize context loading for efficient token usage. Use when working with large codebases, context limits, or when the user mentions "context", "token", "optimize", "summarize", or asks to reduce context size.
---

# Context Optimizer

Optimize AI context loading for efficient token usage and focused development sessions.

## When to Use This Skill

Use this skill when:
- Working with large codebases (>50 files)
- Context window is approaching limits
- User mentions "context", "token", "optimize"
- Need to focus on specific subsystem
- Starting a new development session

## Core Workflow

### Step 1: Analyze Current Context

**Identify loaded files:**
- Recently read files
- Referenced documentation
- Active working files

**Assess relevance:**
- Current task requirements
- File dependencies
- Historical usage patterns

### Step 2: Context Scoring

Score files by relevance (1-10):

| Factor | Weight | Description |
|--------|--------|-------------|
| Direct relevance | 40% | Directly related to current task |
| Dependency chain | 25% | Required by relevant files |
| Recent access | 20% | Recently read or modified |
| Reference frequency | 15% | Often referenced in codebase |

### Step 3: Optimization Strategies

#### Strategy A: Essential Only
```
Load only:
- Files directly being modified
- Critical type definitions
- Immediate dependencies
Token savings: 60-80%
```

#### Strategy B: Focused Context
```
Load:
- Working files + 1 level dependencies
- Relevant documentation
- Key configuration
Token savings: 40-60%
```

#### Strategy C: Summarized Context
```
Load:
- Full working files
- Summaries of related files
- Index of available resources
Token savings: 30-50%
```

## Context Summary Format

### File Summary Template
```markdown
## [filename] Summary
**Purpose:** [one-line description]
**Key exports:** [list of main functions/classes]
**Dependencies:** [key imports]
**Size:** [lines] lines

### Key Sections
- [Section 1]: Lines X-Y - [description]
- [Section 2]: Lines X-Y - [description]
```

### Project Context Template
```markdown
# Project Context Summary

## Architecture
- Pattern: [MVVM/MVC/etc]
- Language: [language + version]
- Framework: [framework details]

## Key Files
| File | Purpose | Priority |
|------|---------|----------|
| file1.cs | Main entry | High |
| file2.cs | Core logic | High |
| file3.cs | Utilities | Medium |

## Current Focus
Working on: [current task]
Relevant files: [list]
```

## Output Format

### Context Analysis Report
```markdown
## Context Optimization Analysis

### Current Context
- Files loaded: 25
- Estimated tokens: ~45,000
- Utilization: 75%

### Recommended Optimization

**Strategy:** Focused Context
**Expected savings:** 40%

#### Keep (High Priority)
- ViewModel.cs - direct modification
- Model.cs - type definitions
- Services/*.cs - active dependencies

#### Summarize (Medium Priority)
- Utils/*.cs - create summaries
- Helpers/*.cs - create summaries

#### Defer (Low Priority)
- Tests/*.cs - load on demand
- Docs/*.md - reference only

### Action
Apply optimization? [Yes/No]
```

## Integration with dev-docs-writer

This skill works with the `dev-docs-writer` agent for optimal context management:

### Document Priority Loading

```yaml
Priority 1 (Always Load):
  - docs/CONTEXT.md      # Quick reference, architecture snapshot
  - docs/PROGRESS.md     # Current phase, active tasks

Priority 2 (Phase-Specific):
  - docs/phases/phase-N/SPEC.md     # Current phase details
  - docs/phases/phase-N/TASKS.md    # Phase tasks
  - src/[active-module]/*           # Active development files

Priority 3 (On-Demand):
  - docs/PRD.md          # Requirements reference
  - docs/TECH-SPEC.md    # Technical details
  - src/**/*             # Specific files as needed
```

### Session Continuity

```markdown
## Starting a new session:

1. Load: docs/CONTEXT.md
2. Check: Current phase from PROGRESS.md
3. Load: Phase-specific files (docs/phases/phase-N/)
4. Resume: Work from last checkpoint
```

### Token Budget Guidelines

| Session Type | Token Budget | Loading Strategy |
|--------------|--------------|------------------|
| Quick check | ~2K | CONTEXT.md only |
| Standard dev | ~10K | CONTEXT + PROGRESS + active files |
| Deep dive | ~30K | All docs + relevant source |
| Full context | ~50K+ | Complete project load |

## Phase-Aware Context Loading

### 자동 Phase 감지

```yaml
# settings.json
context-optimizer:
  auto_load_phase_docs: true
  token_budget:
    quick: 2000
    standard: 10000
    deep: 30000
    full: 50000
```

### Phase 문서 로딩 전략

```
Phase 감지 흐름:
1. PROGRESS.md에서 현재 Phase 확인
2. 해당 Phase 디렉토리 로드
3. SPEC.md → 범위 및 요구사항
4. TASKS.md → 현재 작업 목록
5. CHECKLIST.md → 완료 체크
```

### Phase별 컨텍스트 템플릿

```markdown
# Phase {{N}} Context Summary

## 현재 상태
- Phase: {{PHASE_NAME}}
- 진행률: {{PROGRESS}}%
- 활성 Task: {{ACTIVE_TASKS}}

## 핵심 파일
{{PRIORITY_FILES}}

## 현재 작업
{{CURRENT_WORK}}

## 참조 문서
- [SPEC.md](docs/phases/phase-{{N}}/SPEC.md)
- [TASKS.md](docs/phases/phase-{{N}}/TASKS.md)
```

### 세션 복구 워크플로우

```
새 세션 시작 시:

┌─────────────────────┐
│  CONTEXT.md 로드    │◀─── 필수
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ PROGRESS.md 확인    │◀─── Phase N 감지
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Phase N 문서 로드   │◀─── SPEC + TASKS
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 작업 재개           │
└─────────────────────┘
```

### 토큰 예산별 Phase 로딩

| 예산 | 로드 범위 | 토큰 |
|------|----------|------|
| Quick | CONTEXT.md + PROGRESS.md | ~2K |
| Standard | + 현재 Phase (SPEC, TASKS) | ~10K |
| Deep | + 인접 Phase + 소스 코드 | ~30K |
| Full | 모든 Phase + 전체 문서 | ~50K+ |

## Sprint 통합

Phase와 Sprint 동시 사용 시:

```yaml
Context Loading Priority:
  1. CONTEXT.md
  2. PROGRESS.md
  3. 현재 Sprint (sprints/sprint-N/)
  4. 연결된 Phase (phases/phase-N/)
  5. 소스 코드
```

## Best Practices

1. **Start Lean**: Load minimum required context
2. **Expand as Needed**: Add files when referenced
3. **Summarize Utilities**: Keep only interfaces for helpers
4. **Cache Summaries**: Reuse context summaries across sessions
5. **Document Dependencies**: Track what requires what
6. **Use Phase Documents**: Leverage doc-splitter phase structure
7. **Update PROGRESS.md**: Record session outcomes for continuity
8. **Phase-First**: 현재 Phase 문서 우선 로드
9. **Token Budget**: 세션 유형에 맞는 토큰 예산 설정
