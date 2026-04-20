# 문서 위치 표준 (Document Structure Standard)

cc-initializer의 모든 문서는 이 표준을 따릅니다.

## 표준 문서 구조

```
project-root/
│
├── CLAUDE.md                    # 프로젝트 요약 (Claude Code용)
├── README.md                    # 사용자/개발자 가이드
├── CHANGELOG.md                 # 버전 변경 이력
├── CONTRIBUTING.md              # 기여 가이드 (선택)
│
└── docs/                        # 개발 문서 루트
    │
    ├── _INDEX.md                # 문서 네비게이션
    │
    ├── PRD.md                   # 제품 요구사항 문서
    ├── TECH-SPEC.md             # 기술 설계서
    ├── PROGRESS.md              # 통합 진행 현황
    ├── CONTEXT.md               # 컨텍스트 요약 (AI용)
    │
    ├── phases/                  # Phase 기반 개발
    │   ├── phase-1/
    │   │   ├── SPEC.md          # Phase 기술 상세
    │   │   ├── TASKS.md         # Phase 작업 목록
    │   │   └── CHECKLIST.md     # Phase 완료 체크리스트
    │   ├── phase-2/
    │   │   └── ...
    │   └── phase-N/
    │       └── ...
    │
    ├── sprints/                 # Sprint 기반 실행 (선택)
    │   ├── VELOCITY.md          # 속도 이력
    │   ├── sprint-1/
    │   │   ├── SPRINT.md        # Sprint 계획
    │   │   ├── BACKLOG.md       # Sprint 백로그
    │   │   ├── DAILY.md         # 일일 로그
    │   │   └── RETRO.md         # 회고
    │   └── current -> sprint-N/ # 현재 Sprint 심볼릭 링크
    │
    ├── adr/                     # Architecture Decision Records
    │   ├── 0001-use-phase-system.md
    │   └── 0002-integrate-sprint.md
    │
    ├── api/                     # API 문서 (선택)
    │   └── README.md
    │
    ├── architecture/            # 아키텍처 문서 (선택)
    │   └── README.md
    │
    ├── feedback/                # 피드백 및 학습
    │   ├── learnings/
    │   └── retros/
    │
    └── releases/                # 릴리스 문서
        ├── v1.0.0-notes.md
        └── v1.1.0-checklist.md
```

## 문서 카테고리

### 1. 루트 문서

| 파일 | 용도 | 담당 |
|------|------|------|
| `CLAUDE.md` | AI 컨텍스트 | /init |
| `README.md` | 사용자 가이드 | doc-generator |
| `CHANGELOG.md` | 버전 이력 | agile-sync |

### 2. 개발 프로세스 문서 (docs/)

| 파일 | 용도 | 담당 |
|------|------|------|
| `PRD.md` | 요구사항 정의 | dev-docs-writer |
| `TECH-SPEC.md` | 기술 설계 | dev-docs-writer |
| `PROGRESS.md` | 통합 진행 현황 | progress-tracker |
| `CONTEXT.md` | AI 컨텍스트 요약 | context-optimizer |

### 3. Phase 문서 (docs/phases/)

| 파일 | 용도 | 담당 |
|------|------|------|
| `SPEC.md` | Phase 기술 상세 | doc-splitter |
| `TASKS.md` | Phase 작업 목록 | phase-tracker |
| `CHECKLIST.md` | 완료 체크리스트 | phase-tracker |

### 4. Sprint 문서 (docs/sprints/)

| 파일 | 용도 | 담당 |
|------|------|------|
| `VELOCITY.md` | 속도 이력 | /sprint |
| `SPRINT.md` | Sprint 계획 | /sprint |
| `BACKLOG.md` | Sprint 백로그 | /sprint |
| `DAILY.md` | 일일 로그 | /sprint |
| `RETRO.md` | 회고 | /sprint |

## 문서 명명 규칙

### 파일명

```yaml
Standard Files:
  - 대문자: CLAUDE.md, README.md, PROGRESS.md, TASKS.md
  - 케밥-케이스: tech-spec.md (대문자 대안으로 허용)

Numbered Files:
  - ADR: 0001-decision-title.md
  - Sprint: sprint-1/, sprint-2/
  - Phase: phase-1/, phase-2/

Version Files:
  - v1.0.0-notes.md
  - v1.2.0-checklist.md
```

### 폴더명

```yaml
Standard Folders:
  - 소문자 복수형: phases/, sprints/, releases/
  - 케밥-케이스: release-notes/ (복합어일 경우)

Numbered Folders:
  - phase-N: phase-1/, phase-2/
  - sprint-N: sprint-1/, sprint-2/
```

## Deprecated 패턴

다음 패턴들은 더 이상 사용하지 않습니다:

```yaml
Deprecated:
  # 기능별 progress 파일 (→ Phase TASKS.md로 통합)
  - docs/progress/{feature}-progress.md

  # 별도 status 파일 (→ PROGRESS.md로 통합)
  - docs/progress/status.md

  # 루트의 진행 문서 (→ docs/ 아래로 이동)
  - PROGRESS.md (루트)

Migration:
  docs/progress/auth-progress.md → docs/phases/phase-N/TASKS.md
  docs/progress/status.md → docs/PROGRESS.md
```

## settings.json 연동

```json
{
  "documents": {
    "standard_locations": {
      "progress": "docs/PROGRESS.md",
      "context": "docs/CONTEXT.md",
      "prd": "docs/PRD.md",
      "tech_spec": "docs/TECH-SPEC.md",
      "phases": "docs/phases/",
      "sprints": "docs/sprints/",
      "adr": "docs/adr/",
      "feedback": "docs/feedback/",
      "releases": "docs/releases/"
    }
  }
}
```

## 사용 시나리오별 구조

### 시나리오 1: 소규모 프로젝트

```
docs/
├── PROGRESS.md      # 간단한 진행 추적
└── CONTEXT.md       # AI 컨텍스트
```

### 시나리오 2: Phase 기반 프로젝트

```
docs/
├── PRD.md
├── TECH-SPEC.md
├── PROGRESS.md
├── CONTEXT.md
└── phases/
    ├── phase-1/
    └── phase-2/
```

### 시나리오 3: Agile 팀 프로젝트

```
docs/
├── PRD.md
├── TECH-SPEC.md
├── PROGRESS.md
├── CONTEXT.md
├── phases/
│   └── ...
├── sprints/
│   └── ...
└── adr/
    └── ...
```

## 검증

`/validate --full` 실행 시 이 표준에 따라 문서 위치를 검증합니다.

```bash
# 검증 예시 출력
[Documents]
✅ docs/PROGRESS.md
✅ docs/CONTEXT.md
✅ docs/phases/ (3 phases)
⚠️ docs/sprints/ (empty - enable sprint if needed)
❌ docs/progress/status.md (deprecated, migrate to docs/PROGRESS.md)
```

## 마이그레이션 가이드

기존 프로젝트에서 이 표준으로 마이그레이션:

```bash
# 1. 새 구조 생성
mkdir -p docs/{phases,sprints,adr,feedback,releases}

# 2. 기존 파일 이동
mv docs/progress/status.md docs/PROGRESS.md
mv docs/progress/*-progress.md → docs/phases/phase-N/TASKS.md

# 3. Deprecated 폴더 제거
rm -rf docs/progress/

# 4. 검증
/validate --full
```
