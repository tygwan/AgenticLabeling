# cc-initializer Templates

## 개요

cc-initializer에서 사용하는 표준 템플릿 모음입니다.

## 템플릿 구조

```
templates/
├── phase/                 # Phase 기반 개발 템플릿
│   ├── SPEC.md           # Phase 명세서
│   ├── TASKS.md          # Task 목록
│   └── CHECKLIST.md      # 완료 체크리스트
│
└── README.md             # 이 파일
```

## Phase 템플릿

### SPEC.md - Phase 명세서

Phase의 목표, 범위, 요구사항을 정의합니다.

**포함 내용**:
- 개요 및 기간
- 목표 및 범위 (In/Out)
- 성공 기준
- 기술 요구사항
- 산출물
- 리스크 분석
- 의존 관계

### TASKS.md - Task 목록

Phase에서 수행할 작업 목록을 관리합니다.

**포함 내용**:
- 진행률 표시
- Task 테이블 (ID, 상태, 우선순위, 담당, 공수)
- 마크다운 체크리스트 (Hook 호환)
- 변경 이력

**Task ID 형식**: `T{Phase번호}-{순번}`
- 예: T1-01, T2-05, T3-20

**카테고리**:
- Core (01-09): 핵심 기능
- Infrastructure (10-19): 인프라/설정
- Quality (20-29): 테스트/품질
- Documentation (30-39): 문서화

### CHECKLIST.md - 완료 체크리스트

Phase 완료를 위한 포괄적인 체크리스트입니다.

**체크 항목**:
1. Pre-Phase: 요구사항, 환경, 팀 준비
2. Development: 코드 품질, 테스트, 보안
3. Review: 코드 리뷰, QA
4. Documentation: 기술/사용자 문서
5. Deployment: 준비, 실행
6. Completion: 마무리, 회고, 다음 Phase

## 사용 방법

### 새 Phase 생성 시

```bash
# docs/phases/phase-N/ 생성 후 템플릿 복사
mkdir -p docs/phases/phase-1
cp .claude/templates/phase/*.md docs/phases/phase-1/

# 플레이스홀더 치환
# {{PHASE_NUMBER}} → 1
# {{PHASE_NAME}} → "기초 인프라"
# {{START_DATE}} → "2025-01-15"
# 등...
```

### /init 스킬 연동

```bash
# /init --phase 명령 시 자동으로 템플릿 적용
/init --full

# 복잡도 HIGH인 경우:
# - docs/phases/phase-N/ 자동 생성
# - 템플릿 자동 복사 및 초기화
```

## 플레이스홀더

| 플레이스홀더 | 설명 | 예시 |
|-------------|------|------|
| `{{PHASE_NUMBER}}` | Phase 번호 | 1, 2, 3 |
| `{{PHASE_NAME}}` | Phase 이름 | "기초 인프라" |
| `{{PHASE_GOAL}}` | 핵심 목표 | "MVP 구현" |
| `{{START_DATE}}` | 시작일 | "2025-01-15" |
| `{{END_DATE}}` | 종료일 | "2025-01-31" |
| `{{CREATE_DATE}}` | 생성일 | "2025-01-09" |
| `{{AUTHOR}}` | 작성자 | "개발팀" |
| `{{TASK_N}}` | Task 설명 | "인증 모듈 구현" |
| `{{ESTIMATE_N}}` | 예상 공수 | "2d", "4h" |

## 관련 컴포넌트

| 컴포넌트 | 역할 |
|---------|------|
| `/init` | 프로젝트 초기화 시 템플릿 적용 |
| `dev-docs-writer` | 문서 생성 시 템플릿 참조 |
| `phase-tracker` | Phase 진행 상황 추적 |
| `phase-progress.sh` | 진행률 자동 계산 |

## 커스터마이징

템플릿을 프로젝트에 맞게 수정할 수 있습니다:

1. `.claude/templates/phase/` 파일 수정
2. 새 플레이스홀더 추가 가능
3. 섹션 추가/제거 가능

**주의**: `- [ ]` 체크리스트 형식은 `phase-progress.sh` Hook과 호환을 위해 유지하세요.
