# Learn-from-Friction Standard

## Purpose

프로그래밍 규칙(standards)은 top-down 문서에서 얻어지지 않는다. **실패·설계 결정·반복 작업 같은 "마찰(friction) 이벤트"에서 bottom-up으로 발견**된다. 이 표준은 그 발견을 휘발시키지 않고 **curated journal**로 축적해 포트폴리오 가치와 재사용 가능한 원칙으로 전환하는 루프를 정의한다.

이 문서는 도메인·언어·프레임워크와 독립적이며, 어떤 프로젝트에도 동일하게 적용된다.

## Scope

적용 대상:

- 에러 메시지로 드러난 제약·호환성·환경 이슈
- 디버깅 과정에서 발견된 비자명한 원인
- 반복 작업에서 뽑아낸 추상화
- 설계·아키텍처 결정과 그 근거
- 결과 관찰(측정)이 사전 가정과 어긋났을 때

적용하지 않음:

- 단순 오타·명령어 실수 같은 trivial 이벤트
- 해결에 의미 있는 사고가 들지 않은 것
- 공식 문서에 그대로 쓰여 있는 수준의 상식

## Rule 1. Trigger — 기록은 세 분기점에서만 발생한다

자동 hook은 **쓰지 않는다**. 자동 캡처는 signal↔noise 비율이 나쁘고, 역량을 증명하는 기록이 아니라 반복 실수 리스트가 된다. 기록은 아래 세 분기점에서만 발생한다:

1. **WORKLOG `[DEV]`·`[RESEARCH]`·`[DECISION]` entry 작성 시점** — 에이전트 또는 사람이 entry를 쓰면서 "여기서 뽑을 LEARNING 후보가 있는가?"를 자기 점검한다.
2. **디버깅·에러 해결 직후** — 증상이 해소됐을 때 바로 "이 해결이 일반 원칙으로 떨어지는가?"를 회고한다.
3. **사용자가 명시적으로 기록 지시할 때** — "이거 기록" 같은 지시가 오면 해당 맥락을 담아 LEARNING 후보로 변환한다.

## Rule 2. 품질 게이트 — 4문항 중 2개 이상 YES라야 기록한다

모든 후보는 기록 전에 아래 4개 질문을 통과해야 한다.

1. **일반화 가능한가?** — 이 프로젝트 밖의 다른 언어·도메인·프레임워크에도 적용되는 원칙인가?
2. **비자명한가?** — 공식 문서·기본 튜토리얼 수준을 넘는가? 처음 본 사람에게 설명이 필요한가?
3. **해결에 의미 있는 사고가 들었나?** — 단순 타이핑 오류가 아니라 실제 원인 추적·설계 선택이 필요했나?
4. **명령형 규칙으로 진술 가능한가?** — "~할 때는 ~하라" 형태로 떨어지는가? 단순 일화가 아니라 재사용 가능한 규칙인가?

**2개 미만 YES**면 기록하지 않는다. 반복되지 않는 rare case는 기록하지 않는 것이 맞다.

## Rule 3. Buffer — 단일 파일에 최신순 누적

모든 LEARNING 후보는 **`docs/progress/LEARNINGS.md`** 한 파일에 기록한다. 프로젝트별로 인스턴스가 달라져도 이 버퍼는 한 파일로 유지한다(스캔 용이·검색 용이). entry 구조는:

```markdown
### <YYYY-MM-DD> <한 줄 제목>
- **Trigger type**: error | design | repetition | observation | request
- **Triggered by**: 구체 맥락 (에러 메시지·설계 질문·반복 장면)
- **Evidence**: 실제 로그 snippet / commit hash / file:line / 측정값
- **Rule (draft)**: "~할 때는 ~하라" 형태의 명령형 진술
- **Generality**: 어떤 언어·도메인·프레임워크에 적용되는가
- **Recurrence**: 1 (재발생 시 증가)
- **Status**: draft | validated | promoted | rejected
- **Related**: 관련 WORKLOG entry / observation / commit 링크
```

## Rule 4. Lifecycle — 재발생으로 검증된 것만 dev-standards로 승격

후보는 아래 생애주기를 거친다.

| 상태 | 기준 | 액션 |
|---|---|---|
| `draft` | 처음 기록됨 | 관찰만 함, 코드 변경 없음 |
| `validated` | 다른 맥락(다른 프로젝트·다른 파일·다른 도메인)에서 같은 원칙으로 재발생 확인 | Recurrence 증가, WORKLOG `[DECISION]` entry로 연결 검토 |
| `promoted` | 2회 이상 validated + 일반화 폭 확인 | `docs/standards/`에 전용 섹션 또는 별도 문서로 승격, LEARNINGS entry에 promotion 링크 |
| `rejected` | 후속 관찰에서 실용성 없음 확인, 또는 잘못된 추론이었다고 판명 | 사유 기재 후 삭제 금지(stale tag로 남겨 재발견 방지) |
| `stale` | 30일+ 재발생 없음 | 제거 후보, 세션 말미 정리 시 의사결정 |

승격 시에만 standards 문서가 커진다. 이 게이트가 없으면 LEARNINGS은 그냥 잡동사니가 된다.

## Rule 5. Integration — 기존 protocol과의 연계

이 표준은 이 repo의 다른 protocol과 아래와 같이 연결된다:

| 기존 protocol | 연계 방식 |
|---|---|
| **research-observation-protocol.md** | Rule 8에 따라 observation 완료 시 WORKLOG `[RESEARCH]` entry를 쓴다. 그 entry의 Found·Interpretation 섹션을 쓰면서 LEARNING 후보를 자기 점검한다. observation에서만 드러나는 일반 원칙이 특히 많다(예: bit-exact 비교). |
| **docs/progress/WORKLOG.md** | 매 entry 작성 시 Rule 1.1을 따라 LEARNING 후보 체크. 후보가 있으면 LEARNINGS.md에 draft로 append, 본 entry의 Details에 `Related LEARNING` 링크 추가. |
| **docs/progress/OPTIMIZATION-NOTES.md** | build-first 원칙 하의 최적화 후보 큐. LEARNING과 **목적이 다르다**: OPTIMIZATION-NOTES는 "build 끝난 뒤 성능 손볼 후보", LEARNINGS는 "마찰에서 뽑은 general 원칙". OPTIMIZATION 항목을 결국 act할 때 거기서 뽑히는 general 원칙이 있다면 LEARNINGS로 복사될 수 있다. |
| **docs/standards/ui-change-gate.md** 등 기존 standards | LEARNING이 promoted되면 기존 standard 문서의 섹션으로 편입되거나 신규 standard 파일로 올라간다. standards는 promoted LEARNINGS의 집합으로 성장한다. |

## Pre-record gate

기록 전에 확인한다:

- Rule 2의 4문항 중 2개 이상 YES인가
- 이미 LEARNINGS에 동일한 원칙이 있지 않은가 (중복 방지 — 중복이면 기존 entry의 Recurrence만 증가)
- 이미 standards에 승격되어 있는 원칙이 아닌가
- Trigger type / Triggered by / Evidence 세 필드를 구체적으로 채울 수 있는가

## Post-record cadence

- **매 세션 말미**: 해당 세션에 쌓인 draft를 훑고 명확히 trivial한 것이 있으면 rejected 처리
- **2주마다**: stale 후보 정리, validated 후보를 promote 할지 결정
- **신규 프로젝트 시작 시**: promoted standards만 복사해가고, draft는 두고 가면 된다(draft는 현 프로젝트 맥락이므로)

## Why this exists

프로그래밍 학습은 "책을 읽으며 규칙을 암기"가 아니라 **"마찰을 겪으며 규칙을 발견"**하는 과정이다. 발견된 규칙이 세션 종료와 함께 휘발되면, 같은 마찰을 내년에 또 겪는다. 이 표준은 그 휘발을 막는 최소 장치이다.

또한 어떤 규칙이 **"이 프로젝트 특이성"인지, "프로그래밍 일반 원칙"인지**를 검증 루프(재발생 확인)로 판별한다. 검증을 거친 것만 standards에 올라가므로, standards는 저자의 경력에 관계없이 신뢰할 만한 일반 원칙 집합으로 유지된다.
