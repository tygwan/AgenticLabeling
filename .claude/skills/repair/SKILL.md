---
name: repair
description: cc-initializer 자동 복구 및 문제 해결. Hook 실패, 문서 손상, 설정 오류를 진단하고 수정합니다.
---

# /repair - 시스템 복구 스킬

## Usage

```bash
/repair [mode] [options]
```

### Modes

| Mode | Description |
|------|-------------|
| `--diagnose` | 문제 진단만 수행 (기본) |
| `--auto` | 자동 복구 실행 |
| `--hooks` | Hook 관련 문제만 수정 |
| `--docs` | 문서 구조만 수정 |
| `--config` | 설정 파일만 수정 |
| `--full` | 전체 복구 (모든 항목) |
| `--report` | 복구 보고서 생성 |

## Repair Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│                      /repair WORKFLOW                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 시스템 진단                                                     │
│     ├── Hook 상태 확인                                              │
│     ├── 문서 구조 확인                                              │
│     ├── 설정 유효성 확인                                            │
│     └── 로그 분석                                                   │
│                                                                     │
│  2. 문제 분류                                                       │
│     ├── CRITICAL: 즉시 수정 필요                                    │
│     ├── WARNING: 수정 권장                                          │
│     └── INFO: 참고사항                                              │
│                                                                     │
│  3. 자동 복구                                                       │
│     ├── Hook 권한 복구                                              │
│     ├── 누락 디렉토리 생성                                          │
│     ├── 설정 기본값 복원                                            │
│     └── 캐시/로그 정리                                              │
│                                                                     │
│  4. 수동 복구 안내                                                  │
│     └── 자동 복구 불가 항목 안내                                    │
│                                                                     │
│  5. 복구 보고서                                                     │
│     └── 결과 요약 및 후속 조치                                      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Diagnosable Issues

### Hook Issues

```yaml
Hook Problems:
  - permission_denied: Hook 실행 권한 없음
  - file_missing: Hook 파일 누락
  - syntax_error: 스크립트 문법 오류
  - dependency_missing: 의존성 누락

Auto-Fix:
  - chmod +x: 실행 권한 부여
  - restore_template: 기본 템플릿 복원
```

### Document Issues

```yaml
Document Problems:
  - missing_directory: docs/, phases/, sprints/ 누락
  - missing_required: PROGRESS.md, CONTEXT.md 누락
  - broken_links: 내부 링크 손상
  - invalid_format: 마크다운 형식 오류

Auto-Fix:
  - mkdir -p: 디렉토리 생성
  - create_template: 기본 문서 생성
  - fix_links: 링크 수정
```

### Configuration Issues

```yaml
Config Problems:
  - invalid_json: settings.json 문법 오류
  - missing_section: 필수 섹션 누락
  - invalid_reference: 존재하지 않는 파일 참조
  - deprecated_setting: 지원 종료 설정

Auto-Fix:
  - restore_defaults: 기본값 복원
  - migrate_settings: 설정 마이그레이션
  - add_missing: 누락 섹션 추가
```

## Output Examples

### Diagnose Mode

```
🔍 REPAIR: System Diagnosis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Category        Status    Issues
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hooks           ⚠️ WARN   2
Documents       ❌ ERROR  1
Configuration   ✅ OK     0
Logs            ✅ OK     0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Detailed Issues

[WARN] Hook: auto-doc-sync.sh
       Not executable (chmod +x needed)
       Auto-fixable: Yes

[WARN] Hook: notification-handler.sh
       Not executable (chmod +x needed)
       Auto-fixable: Yes

[ERROR] Document: docs/PROGRESS.md
        File missing
        Auto-fixable: Yes (run /init --docs-only)

💡 Recommendation: Run `/repair --auto` to fix 3 issues
```

### Auto Mode

```
🔧 REPAIR: Auto Recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting automatic repair...

[1/4] Fixing hook permissions...
      ✅ chmod +x .claude/hooks/auto-doc-sync.sh
      ✅ chmod +x .claude/hooks/notification-handler.sh

[2/4] Creating missing directories...
      ✅ mkdir -p docs/phases
      ✅ mkdir -p docs/sprints

[3/4] Restoring missing files...
      ⚠️ docs/PROGRESS.md requires /init (skipped)

[4/4] Cleaning up...
      ✅ Cleared old log files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Repair Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixed automatically: 4
Requires manual action: 1
Total issues: 5

📝 Manual Action Required:
   1. Run `/init --docs-only` to create PROGRESS.md
```

### Report Mode

```
📊 REPAIR: Recovery Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated: 2025-01-09 15:30:00

## Error History (Last 7 Days)

| Date       | Hook              | Error           | Recovery      |
|------------|-------------------|-----------------|---------------|
| 2025-01-09 | auto-doc-sync.sh  | syntax error    | Fixed regex   |
| 2025-01-08 | phase-progress.sh | permission      | chmod +x      |

## Recovery Actions Taken

- Total recoveries: 5
- Auto-fixed: 4
- Manual: 1

## System Health Score: 95/100

Recommendations:
1. Review auto-doc-sync.sh for edge cases
2. Consider enabling auto_check_on_start

Full logs: .claude/logs/recovery.log
```

## Recovery Commands

### Quick Fixes

```bash
# Hook 권한 일괄 수정
/repair --hooks

# 문서 구조 복구
/repair --docs

# 설정 검증 및 수정
/repair --config
```

### Full Recovery

```bash
# 전체 진단 및 자동 복구
/repair --full

# 진단만 수행
/repair --diagnose

# 복구 후 검증
/repair --auto && /validate --full
```

## Integration

### With /validate

```bash
# validate 실패 시 repair 연계
/validate --full
# If fails:
/repair --auto
/validate --full
```

### With Error Recovery Hook

```bash
# Hook 실패 시 자동 복구 트리거
# settings.json에서 활성화:
{
  "recovery": {
    "auto_recover_on_hook_failure": true,
    "max_retry_count": 3
  }
}
```

### With quality-gate

```bash
# Pre-commit 실패 시 복구 시도
quality-gate pre-commit || /repair --auto
```

## Configuration

```json
// settings.json
{
  "recovery": {
    "enabled": true,
    "auto_recover_on_hook_failure": true,
    "max_retry_count": 3,
    "log_retention_days": 7,
    "critical_hooks": ["pre-tool-use-safety.sh"],
    "auto_fixable": {
      "hook_permissions": true,
      "missing_directories": true,
      "log_rotation": true
    },
    "manual_required": {
      "missing_docs": ["PROGRESS.md", "CONTEXT.md"],
      "config_validation": true
    }
  }
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 0 | All repairs successful |
| 1 | Some repairs needed manual action |
| 2 | Critical errors (repair failed) |
| 3 | System in unrecoverable state |

## Related

| Command | Purpose |
|---------|---------|
| `/validate` | System validation |
| `/init` | Initialize/restore configuration |
| `/agile-sync` | Sync documentation |
| `error-recovery.sh` | Recovery hook |
