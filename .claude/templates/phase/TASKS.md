# Phase {{PHASE_NUMBER}} Tasks

## 개요

| 항목 | 내용 |
|------|------|
| **Phase** | Phase {{PHASE_NUMBER}} - {{PHASE_NAME}} |
| **총 Task** | 0 |
| **완료** | 0 |
| **진행률** | 0% |

## 진행률

```
[░░░░░░░░░░░░░░░░░░░░] 0%
```

## Task 목록

### 핵심 기능 (Core)

| ID | Task | 상태 | 우선순위 | 담당 | 예상 공수 |
|----|------|------|----------|------|----------|
| T{{PHASE_NUMBER}}-01 | {{TASK_1}} | ⏳ | P0 | - | {{ESTIMATE_1}} |
| T{{PHASE_NUMBER}}-02 | {{TASK_2}} | ⏳ | P0 | - | {{ESTIMATE_2}} |
| T{{PHASE_NUMBER}}-03 | {{TASK_3}} | ⏳ | P1 | - | {{ESTIMATE_3}} |

### 인프라/설정 (Infrastructure)

| ID | Task | 상태 | 우선순위 | 담당 | 예상 공수 |
|----|------|------|----------|------|----------|
| T{{PHASE_NUMBER}}-10 | {{INFRA_TASK_1}} | ⏳ | P1 | - | {{ESTIMATE}} |

### 테스트/품질 (Quality)

| ID | Task | 상태 | 우선순위 | 담당 | 예상 공수 |
|----|------|------|----------|------|----------|
| T{{PHASE_NUMBER}}-20 | 단위 테스트 작성 | ⏳ | P1 | - | - |
| T{{PHASE_NUMBER}}-21 | 통합 테스트 작성 | ⏳ | P2 | - | - |

### 문서화 (Documentation)

| ID | Task | 상태 | 우선순위 | 담당 | 예상 공수 |
|----|------|------|----------|------|----------|
| T{{PHASE_NUMBER}}-30 | API 문서 작성 | ⏳ | P2 | - | - |
| T{{PHASE_NUMBER}}-31 | 사용자 가이드 | ⏳ | P3 | - | - |

---

## 상태 범례

| 아이콘 | 상태 | 설명 |
|--------|------|------|
| ⏳ | 대기 | 시작 전 |
| 🔄 | 진행 | 작업 중 |
| 🔍 | 리뷰 | 리뷰 대기 |
| ✅ | 완료 | 완료됨 |
| ❌ | 취소 | 취소됨 |
| 🚧 | 차단 | 블로커 있음 |

## 우선순위 정의

| 우선순위 | 설명 |
|----------|------|
| P0 | 크리티컬 - 이번 Phase 필수 |
| P1 | 높음 - 핵심 기능 |
| P2 | 중간 - 중요하지만 지연 가능 |
| P3 | 낮음 - Nice to have |

---

## Task 체크리스트 (마크다운 형식)

> Hook 호환을 위한 체크리스트 형식

### Core Tasks
- [ ] T{{PHASE_NUMBER}}-01: {{TASK_1}}
- [ ] T{{PHASE_NUMBER}}-02: {{TASK_2}}
- [ ] T{{PHASE_NUMBER}}-03: {{TASK_3}}

### Infrastructure Tasks
- [ ] T{{PHASE_NUMBER}}-10: {{INFRA_TASK_1}}

### Quality Tasks
- [ ] T{{PHASE_NUMBER}}-20: 단위 테스트 작성
- [ ] T{{PHASE_NUMBER}}-21: 통합 테스트 작성

### Documentation Tasks
- [ ] T{{PHASE_NUMBER}}-30: API 문서 작성
- [ ] T{{PHASE_NUMBER}}-31: 사용자 가이드

---

## 변경 이력

| 날짜 | Task ID | 변경 내용 |
|------|---------|----------|
| {{CREATE_DATE}} | - | 최초 작성 |
