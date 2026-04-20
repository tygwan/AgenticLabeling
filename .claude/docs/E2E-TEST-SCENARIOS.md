# E2E 테스트 시나리오

## 개요

cc-initializer의 전체 워크플로우를 검증하기 위한 통합 테스트 시나리오입니다.

## 시나리오 1: 프로젝트 초기화 → 기능 개발 → 릴리스

### 1.1 프로젝트 초기화

```bash
# 사전 조건
- 새 프로젝트 디렉토리
- Git 초기화 완료

# 실행
/init --full

# 예상 결과
✅ CLAUDE.md 생성
✅ docs/PRD.md 생성
✅ docs/TECH-SPEC.md 생성
✅ docs/PROGRESS.md 생성
✅ docs/CONTEXT.md 생성
✅ docs/phases/ 구조 생성 (HIGH complexity 시)

# 검증 방법
ls -la docs/
cat CLAUDE.md
```

### 1.2 설정 검증

```bash
# 실행
/validate --full

# 예상 결과
✅ settings.json: Valid
✅ Hooks: All executable
✅ Agents: All valid frontmatter
✅ Skills: All have SKILL.md

# 검증 방법
# 모든 구성요소 상태 확인
```

### 1.3 기능 개발 시작

```bash
# 실행
/feature start "사용자 인증" --phase 1

# 예상 결과
✅ Git branch 생성: feature/user-authentication
✅ Phase 1 TASKS.md에 Task 추가
✅ PROGRESS.md 업데이트

# 검증 방법
git branch
cat docs/phases/phase-1/TASKS.md
cat docs/PROGRESS.md
```

### 1.4 개발 중 Hook 자동화

```bash
# 파일 수정 시
Edit docs/phases/phase-1/TASKS.md  # Task 완료 표시

# 예상 결과 (Hook 자동 실행)
✅ phase-progress.sh: 진행률 자동 계산
✅ PROGRESS.md 자동 업데이트

# 검증 방법
cat docs/PROGRESS.md  # 진행률 변경 확인
```

### 1.5 기능 완료

```bash
# 실행
/feature complete

# 예상 결과
✅ Quality gate 통과 (lint, test)
✅ Phase Task 완료 표시
✅ PR 생성
✅ CHANGELOG.md 업데이트

# 검증 방법
git log --oneline -3
cat CHANGELOG.md
```

### 1.6 릴리스

```bash
# 실행
/release prepare v1.0.0
/release create v1.0.0
/release publish v1.0.0

# 예상 결과
✅ 모든 Phase 완료 확인
✅ CHANGELOG.md 정리
✅ Git tag 생성
✅ GitHub Release 생성

# 검증 방법
git tag -l
```

---

## 시나리오 2: 버그 수정 워크플로우

### 2.1 버그 수정 시작

```bash
# 실행
/bugfix start "로그인 실패" --issue 123 --priority high

# 예상 결과
✅ Git branch 생성: fix/login-failure-123
✅ Sprint에 hotfix 항목 추가

# 검증 방법
git branch
```

### 2.2 원인 분석

```bash
# 실행
/bugfix analyze

# 예상 결과
✅ 관련 파일 검색
✅ Git history 분석
✅ Root cause 도출

# 검증 방법
# 분석 결과 확인
```

### 2.3 버그 수정 완료

```bash
# 실행
/bugfix complete

# 예상 결과
✅ Quality gate 통과
✅ PR 생성 (Fixes #123)
✅ CHANGELOG.md에 버그 수정 기록

# 검증 방법
cat CHANGELOG.md | grep "Fixed"
```

---

## 시나리오 3: Phase + Sprint 통합

### 3.1 Phase 기반 개발 시작

```bash
# Phase 구조 확인
ls docs/phases/

# Phase 1 Task 확인
cat docs/phases/phase-1/TASKS.md
```

### 3.2 Sprint 시작

```bash
# 실행
/sprint start --phase 1 --name "Sprint 1"

# 예상 결과
✅ docs/sprints/sprint-1/ 생성
✅ Phase 1 Task → Sprint Backlog 연결

# 검증 방법
cat docs/sprints/sprint-1/BACKLOG.md
```

### 3.3 Sprint Task 완료

```bash
# 실행
/sprint complete T1-01

# 예상 결과
✅ Sprint BACKLOG.md 업데이트
✅ Phase TASKS.md 자동 업데이트 (T1-01 ✅)
✅ PROGRESS.md 진행률 반영

# 검증 방법
cat docs/phases/phase-1/TASKS.md | grep T1-01
cat docs/PROGRESS.md
```

### 3.4 Sprint 종료

```bash
# 실행
/sprint end

# 예상 결과
✅ Velocity 계산
✅ RETRO.md 생성
✅ 미완료 항목 다음 Sprint로 이월

# 검증 방법
cat docs/sprints/sprint-1/RETRO.md
cat docs/sprints/VELOCITY.md
```

---

## 시나리오 4: 에러 상황 처리

### 4.1 위험 명령어 실행 시도

```bash
# 실행
Bash: rm -rf /

# 예상 결과
❌ BLOCKED by pre-tool-use-safety.sh
[Safety] 🚫 BLOCKED: Dangerous command detected

# 검증 방법
# 명령어가 실행되지 않음
```

### 4.2 민감 파일 접근 시도

```bash
# 실행
Edit: .env.production

# 예상 결과
⚠️ WARNING by pre-tool-use-safety.sh
[Safety] ⚠️ WARNING: Accessing potentially sensitive file

# 검증 방법
# 경고 메시지 출력 후 진행
```

### 4.3 잘못된 설정 검증

```bash
# 실행
/validate --full

# 예상 결과 (설정 오류 시)
⚠️ settings.json: Invalid JSON
❌ Hook: missing required file
💡 Recommendations: 수정 방법 안내

# 검증 방법
# 오류 및 권장사항 확인
```

---

## 시나리오 5: 경계 케이스

### 5.1 Phase 없이 Sprint만 사용

```bash
# 사전 조건
- docs/phases/ 없음
- Sprint만 사용하는 유지보수 프로젝트

# 실행
/sprint start --name "Maintenance Sprint 1"

# 예상 결과
✅ Sprint 생성
✅ Phase 연동 없이 독립 운영

# 검증 방법
cat docs/sprints/sprint-1/SPRINT.md
```

### 5.2 Sprint 없이 Phase만 사용

```bash
# 사전 조건
- Sprint 비활성화
- Phase만으로 진행 관리

# 실행
/phase status

# 예상 결과
✅ Phase 진행률 표시
✅ Sprint 관련 내용 없음

# 검증 방법
cat docs/PROGRESS.md
```

### 5.3 중간에 Phase 전환

```bash
# 사전 조건
- Phase 1 진행 중
- 긴급하게 Phase 2 시작 필요

# 실행
/phase complete 1 --force
/phase start 2

# 예상 결과
⚠️ Phase 1 미완료 항목 경고
✅ Phase 2 활성화

# 검증 방법
cat docs/phases/phase-1/CHECKLIST.md
```

---

## 테스트 체크리스트

### 초기화

- [ ] /init --full 실행
- [ ] docs/ 구조 생성 확인
- [ ] CLAUDE.md 생성 확인

### 설정

- [ ] /validate --full 통과
- [ ] 모든 Hook 동작 확인

### 기능 개발

- [ ] /feature start 브랜치 생성
- [ ] /feature complete PR 생성

### 버그 수정

- [ ] /bugfix start Issue 연결
- [ ] /bugfix complete 수정 완료

### Phase + Sprint

- [ ] Phase → Sprint Task 연결
- [ ] Sprint 완료 → Phase 자동 업데이트
- [ ] PROGRESS.md 자동 갱신

### 릴리스

- [ ] /release prepare 검증
- [ ] /release create Tag 생성
- [ ] CHANGELOG.md 정리

### 안전성

- [ ] 위험 명령어 차단
- [ ] 민감 파일 경고
- [ ] 에러 발생 시 복구 가능

---

## 테스트 실행 방법

```bash
# 1. 테스트 프로젝트 생성
mkdir test-project && cd test-project
git init

# 2. cc-initializer 복사
cp -r ../cc-initializer/.claude .

# 3. 시나리오별 테스트 실행
# (각 시나리오 명령어 순차 실행)

# 4. 결과 검증
# (각 시나리오 검증 방법 참조)
```
