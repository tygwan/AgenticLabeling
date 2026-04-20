# Concepts Protocol

## Purpose

이 프로젝트는 observation(측정·관찰 사실), LEARNING(명령형 규칙), WORKLOG(시간순 결정), concept(개념 설명) 네 종류의 기록을 유지한다. **개념 설명이 여러 기록에 분산·중복되면 drift와 silo가 발생**하므로, 개념 설명의 single source of truth로 `docs/concepts/` 허브를 둔다. 이 표준은 어느 기록에 무엇이 살아야 하는지, concept 파일을 언제·어떻게 작성·갱신하는지를 정한다.

이 규칙은 이 프로젝트용이지만 다른 프로젝트에도 그대로 적용 가능한 universal loop를 지향한다.

## Scope

적용 대상:

- 프로젝트에서 반복 등장하는 기술·도메인 개념 (프레임워크 동작, 모델 구조, 프로토콜, 데이터 포맷, 아키텍처 패턴)
- 관찰(observation)이 드러낸 일반 개념 (dtype precision 동작, attention 경로 선택 등)
- 엔지니어링 선택의 배경 설명 (async vs sync 서버, WAL vs 기본 journaling)

적용하지 않음:

- 프로젝트 고유 파일 경로·구체 구성값 (WORKLOG에 남김)
- 단발성 디버깅 팁 (concept 가치 게이트 미통과)
- 공식 문서에 그대로 있는 수준의 상식 (링크만 남기거나 생략)

## Rule 1. 단일 진원지 원칙 (Single source of truth)

각 개념의 **설명**은 `docs/concepts/<category>/<slug>.md` 에 오직 한 번만 존재한다.

- observation report의 `Interpretation` 섹션, LEARNING 본문, WORKLOG entry 설명은 **개념 설명을 다시 쓰지 않는다**. 대신 해당 concept 파일로 링크한다.
- concept 파일은 observation의 **구체 수치를 복제하지 않는다**. 구체 수치는 observation report에 두고 concept는 `Evidence in this project` 섹션에서 링크·짧은 인용으로만 사용한다.

## Rule 2. 교육 가치 게이트 (concept 생성 기준)

concept 파일은 아래 4문항 중 2개 이상 YES일 때만 생성한다.

1. **반복 등장하는가** — 이 개념이 이 프로젝트의 다른 곳(다른 모듈·다른 실험·다른 feature)에서도 재등장할 가능성이 있는가?
2. **비자명한가** — 처음 보는 기여자에게 자명하지 않은가? 설명 없이는 맥락이 안 잡히는가?
3. **참조 가치** — 나 또는 미래 기여자가 **다시 찾아볼 가치가 있는가**? 일회성 트릭이 아닌가?
4. **근거 가용성** — 이미 observation·코드·외부 문서에 구체적 근거가 있는가? (isolation 방지)

2개 미만이면 WORKLOG 본문이나 observation report 안에 짧게 남기고 concept는 만들지 않는다.

## Rule 3. 3-섹션 고정 템플릿

```markdown
---
type: research-derived | engineering | hybrid
primary_evidence: <research/observations/... | code file refs | external docs>
promoted_from_learnings: <optional>
---

# <Concept Name>

## Concept framework
(일반 기초 → 심화 trade-offs. 이 프로젝트에 얽매이지 않는 일반론.)

## Evidence in this project
(프로젝트 내 구체 발현. 각 bullet은 observation·코드 링크와 짧은 인용 포함.)

## Related rules / decisions
(LEARNING·WORKLOG 결정 링크. 본문 반복 금지.)
```

- `type: research-derived` — 근거가 `research/observations/`에서 오는 경우 (대부분 ML)
- `type: engineering` — 근거가 코드·외부 문서인 경우 (대부분 backend·frontend·infra)
- `type: hybrid` — 둘 다

## Rule 4. 작성·업데이트 트리거 (네 분기점)

concept 파일은 아래 네 순간에 참조·업데이트·생성된다.

1. **observation report 완료 시** — Interpretation 성격의 내용을 concept의 Evidence section으로 이관. report.md의 Interpretation 섹션은 축소하고 `Evidence contribution` 링크만 남김 (research-observation-protocol.md Rule 9).
2. **WORKLOG entry 작성 시** — entry가 기대는 개념이 concept에 있으면 `Details`에 링크. 없으면 Rule 2 게이트 통과 시 stub 생성.
3. **LEARNING 작성 시** — 규칙의 배경·메커니즘 설명을 concept으로 위임. LEARNING 본문은 명령형 규칙만 유지.
4. **사용자 깊이 있는 질문 시** — "왜 X?" 답변이 개념적 깊이를 가지면 concept 파일로 저장.

## Rule 5. Integration — 다른 기록과의 관계

| 다른 기록 | 역할 분담 |
|---|---|
| **observation report** | 측정·관찰 사실만. 해석적 개념 설명은 concept로 위임. report는 `Evidence contribution` 목록을 둠 (어느 concept의 어느 subsection에 기여하는지 명시). |
| **LEARNINGS** | 명령형 규칙만. 배경·메커니즘·예시 설명은 concept로 링크. 한 LEARNING이 여러 concept을 참조 가능. 한 concept이 여러 LEARNING을 모을 수 있음. |
| **WORKLOG** | 시간순 활동·결정. 개념 설명은 링크만 가능. entry의 `Details`에 `Concept: [링크]` 형식. |
| **standards** | prescriptive rules. LEARNING의 promoted 버전 + concept 참조 |

## Rule 6. 흐름 방향 (중요)

```
observation.Observations (raw)
        ↓
observation.Evidence contribution (기여 선언)
        ↓                                     ↑
concept.Evidence in this project (축적)      concept.Concept framework (일반론)
        ↓                                     ↓
LEARNING (명령형 규칙)                        WORKLOG (결정·활동)
```

- 방향은 **observation → concept → LEARNING / WORKLOG** 가 primary. 반대 방향은 발생 시 concept의 Evidence에 추가.
- concept은 hub, 다른 기록은 spokes. concept은 duplication 허용하지 않지만 링크 aggregation은 허용.

## Rule 7. 인덱스

`docs/concepts/README.md`에 현재 등록된 concept 목록을 최신순으로 유지. 새 concept 생성 시 이 인덱스에 한 줄 추가.

## Pre-create gate

concept 파일 생성 전에:

- Rule 2 게이트 4문항 중 2개 이상 YES 확인
- 이미 존재하는 concept과 중복 안 되는지 검색
- 초기 Evidence section에 최소 1개의 구체 근거 링크 가능한지 확인 (isolation 방지)
- 카테고리(ml/backend/frontend/infra/testing) 선택 명확한지 확인

## Post-create cadence

- **매 observation 완료**: Evidence contribution 링크가 실제로 concept에 반영됐는지 확인
- **2주마다**: concept별 Evidence section 재훑어 누락된 observation·코드 변경 반영. 링크 유효성 검사.
- **분기별**: concept 중 stale(6개월+ 무수정)한 항목은 retire 또는 merge 고려.

## Why this exists

관찰과 설명은 서로 다른 수명과 재사용 범위를 갖는다. 관찰은 **특정 시점의 측정**이라 불변이고 언제든 근거로 인용될 수 있다. 개념 설명은 **시간에 따라 발전·정제**된다. 이 둘을 같은 문서 안에 섞으면 관찰 시점의 이해 수준에 개념 설명이 고착되고, 나중에 더 깊은 해석이 나와도 과거 observation에 반영되지 못한다.

concept hub를 두면:
- observation은 raw data로서 불변
- concept는 계속 업데이트 가능한 살아있는 synthesis
- LEARNING은 행동 지침으로 공격적으로 적용 가능
- WORKLOG는 순수 시간순 피드로 휘발 걱정 없음

이 분리가 오랜 기간에 걸쳐 **지식의 drift 없이 축적**을 가능하게 한다.
