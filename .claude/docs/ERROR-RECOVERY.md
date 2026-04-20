# 에러 복구 시스템

## 개요

cc-initializer의 에러 복구 시스템은 Hook 실패, 문서 손상, 설정 오류를 자동으로 감지하고 복구합니다.

## 아키텍처

```
┌────────────────────────────────────────────────────────────────────┐
│                    ERROR RECOVERY SYSTEM                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │   Hooks     │─────▶│  Recovery   │─────▶│   Logs      │         │
│  │   (5개)     │ fail │   Handler   │ log  │   System    │         │
│  └─────────────┘      └─────────────┘      └─────────────┘         │
│         │                    │                    │                 │
│         │                    ▼                    │                 │
│         │            ┌─────────────┐              │                 │
│         │            │  Graceful   │              │                 │
│         └───────────▶│ Degradation │◀─────────────┘                 │
│                      └─────────────┘                                │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    RECOVERY SKILLS                           │   │
│  ├──────────────────┬──────────────────┬───────────────────────┤   │
│  │    /repair       │   /sync-fix      │     /validate         │   │
│  │    시스템 복구   │   동기화 수정    │     설정 검증         │   │
│  └──────────────────┴──────────────────┴───────────────────────┘   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Hook 분류

### Critical Hooks (차단형)

실패 시 작업을 차단합니다:

| Hook | 역할 | 실패 시 동작 |
|------|------|------------|
| `pre-tool-use-safety.sh` | 위험 명령어 차단 | **작업 차단** |

### Non-Critical Hooks (계속형)

실패 시 로그만 남기고 계속 진행합니다:

| Hook | 역할 | 실패 시 동작 |
|------|------|------------|
| `phase-progress.sh` | 진행률 계산 | 로그 후 계속 |
| `auto-doc-sync.sh` | 문서 동기화 | 로그 후 계속 |
| `post-tool-use-tracker.sh` | 변경 추적 | 로그 후 계속 |
| `notification-handler.sh` | 알림 처리 | 무시 후 계속 |

## Graceful Degradation

```
Hook 실패 발생
       │
       ▼
┌──────────────────────┐
│  Hook 분류 확인      │
└──────────────────────┘
       │
       ├── Critical ──────────▶ 작업 차단 + 오류 반환
       │
       └── Non-Critical ──────▶ 로그 기록 + 계속 진행
                                      │
                                      ▼
                               ┌──────────────┐
                               │ 복구 제안    │
                               │ 출력         │
                               └──────────────┘
```

## 로그 시스템

### 로그 위치

```
.claude/logs/
├── error.log      # 에러 로그
└── recovery.log   # 복구 작업 로그
```

### 로그 형식

**error.log**:
```
[2025-01-09 15:30:00] ERROR: phase-progress.sh failed
  Context: Exit code: 1, Line: 45
```

**recovery.log**:
```
[2025-01-09 15:30:01] RECOVERY: phase-progress.sh failed
  Action: Skipping progress update, manual sync needed
```

### 로그 관리

- **자동 로테이션**: 1MB 초과 시 자동 로테이션
- **최대 파일 수**: 5개 (error.log.1 ~ error.log.5)
- **보관 기간**: 7일 (설정 가능)

## 복구 스킬

### /repair

시스템 전반의 문제를 진단하고 수정합니다.

```bash
# 진단만 수행
/repair --diagnose

# 자동 복구
/repair --auto

# 전체 복구
/repair --full
```

**자동 수정 가능 항목**:
- Hook 실행 권한
- 누락 디렉토리
- 로그 파일 정리

**수동 수정 필요 항목**:
- 핵심 문서 (PROGRESS.md, CONTEXT.md)
- JSON 설정 오류

### /sync-fix

문서 간 동기화 문제를 해결합니다.

```bash
# 문제 확인
/sync-fix --dry-run

# Phase 동기화
/sync-fix --phase

# 전체 동기화
/sync-fix --all
```

**감지 가능 문제**:
- Phase ↔ PROGRESS.md 불일치
- Sprint ↔ Phase Task 불일치
- 깨진 문서 링크

## 설정

```json
// settings.json
{
  "recovery": {
    "enabled": true,
    "auto_recover_on_hook_failure": true,
    "max_retry_count": 3,
    "graceful_degradation": true,
    "critical_hooks": ["pre-tool-use-safety.sh"],
    "non_critical_hooks": [
      "auto-doc-sync.sh",
      "phase-progress.sh",
      "post-tool-use-tracker.sh",
      "notification-handler.sh"
    ]
  }
}
```

## 문제 해결 흐름

```
문제 발생
    │
    ▼
/validate --full  ──────▶  설정 문제 확인
    │
    ▼
/repair --diagnose ─────▶  시스템 문제 확인
    │
    ▼
/repair --auto ─────────▶  자동 수정 시도
    │
    ▼
/sync-fix --all ────────▶  동기화 문제 해결
    │
    ▼
/validate --full ───────▶  최종 검증
```

## 관련 컴포넌트

| 컴포넌트 | 역할 |
|---------|------|
| `error-recovery.sh` | 복구 핸들러 Hook |
| `/repair` | 시스템 복구 스킬 |
| `/sync-fix` | 동기화 복구 스킬 |
| `/validate` | 설정 검증 스킬 |
| `config-validator` | 검증 에이전트 |
