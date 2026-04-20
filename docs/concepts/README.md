# Concepts

Concept 파일은 이 프로젝트에서 반복 등장하는 개념들의 **종합 지식 허브**다. 기초·심화 설명 한 번, 프로젝트 내 근거 축적, 관련 규칙·결정 링크 — 세 역할을 하나의 파일에서 담는다.

- **regulating standard**: [docs/standards/concepts-protocol.md](../standards/concepts-protocol.md)
- **원칙**: [learn-from-friction.md](../standards/learn-from-friction.md) · [research-observation-protocol.md Rule 9](../standards/research-observation-protocol.md)

## 이 허브가 존재하는 이유

일반 개념 설명이 여러 곳에 흩어지면 (observation report의 Interpretation, WORKLOG entry 설명, LEARNING 본문) **같은 내용을 다시 쓰거나 다른 버전으로 drift**한다. Concept 파일은 개념 설명의 **단일 진원지(single source of truth)**가 되어 다른 기록들이 **링크로만** 참조하게 한다.

## 디렉토리 구조

```
docs/concepts/
├── README.md             # 이 파일
├── ml/                   # 모델·추론·파이프라인 개념 (대부분 research-derived)
├── backend/              # FastAPI·uvicorn·SQLite·ChromaDB 등 (대부분 engineering)
├── frontend/             # React UMD·design tokens·localStorage 등 (engineering)
├── infra/                # Docker·CUDA·triton 등 (engineering)
└── testing/              # pytest·TestClient·mock patterns (engineering)
```

## 파일 템플릿

모든 concept 파일은 아래 3섹션 고정 구조를 따른다. 상단 메타 프론트매터 필수.

```markdown
---
type: research-derived | engineering | hybrid
primary_evidence: <research/observations/... | code refs | external docs>
promoted_from_learnings: <optional — LEARNING이 이 concept으로 확장된 경우>
---

# <Concept Name>

## Concept framework
일반 기초 → 심화 trade-offs. 교과서 수준 설명 + 분야에서의 advanced 판단. 이 프로젝트에 얽매이지 않는 일반론.

## Evidence in this project
이 프로젝트 안에서 이 개념이 어떻게·어디서 나타나는지. 모든 항목은 관찰·코드·문서 링크와 함께 **근거 단위**로 나열한다.

- **<구체 현상 한 줄>** — [observation:<slug>] §<섹션>
  > 인용 또는 요약 1-2줄
- **<구체 현상 한 줄>** — `file.py:line`
  > 코드·메트릭 요약

## Related rules / decisions
이 concept이 배경이 되는 LEARNING·WORKLOG 결정 링크. 정보 반복 금지, 링크만.

- [LEARNING: <title>](../progress/LEARNINGS.md#...)
- [WORKLOG entry <date>: <title>](../progress/WORKLOG.md#...)
```

## 작성·업데이트 트리거

매번 다음 순간에 concept 파일을 참조·업데이트·생성한다 ([concepts-protocol.md](../standards/concepts-protocol.md) 공식 규칙):

1. **observation report 완료 시** — Interpretation 성격의 내용을 concept의 Evidence로 이관. report.md는 Evidence contribution으로 축소.
2. **WORKLOG entry 작성 시** — entry가 기대는 개념이 이미 concept에 있으면 링크, 없으면 교육 가치 게이트 통과 시 stub 생성.
3. **LEARNING 작성 시** — 규칙의 배경 설명을 concept으로 위임. LEARNING 본문은 명령형 규칙만 남김.
4. **사용자 질문 시** — "왜 X?" 답변이 개념적 깊이를 가지면 concept으로 저장.

## 교육 가치 게이트 (concept 생성 기준)

[concepts-protocol.md](../standards/concepts-protocol.md) Rule 2 참조. 4문항 중 2개 이상 YES라야 concept 파일 생성:

1. 이 개념이 이 프로젝트 다른 곳에서도 재등장할 가능성이 있는가?
2. 처음 보는 사람에게 자명하지 않은가?
3. 나 또는 미래 기여자가 다시 찾아볼 가치가 있는가?
4. 이미 observation·코드·외부 docs에 구체적 근거가 있는가?

## LEARNINGS vs Concepts

| | LEARNING | Concept |
|---|---|---|
| 형식 | 명령형 규칙 ("~할 때는 ~하라") | 서술·설명 ("X는 Y로 동작한다") |
| 목적 | 행동 가이드 | 이해 기반 |
| 길이 | 짧게 (entry 단위) | 길게 (파일 단위) |
| 관계 | concept을 배경으로 참조 | 여러 LEARNING의 근거가 됨 |

Concept 파일은 LEARNING의 배경·메커니즘을 설명하고, LEARNING은 concept을 가리킨다. 중복 금지.

## 인덱스

현재 등록된 concept (최신순):

- [frontend/image-viewer-overlay-alignment.md](frontend/image-viewer-overlay-alignment.md) — contain-fit 이미지 + bbox/mask/label 오버레이 픽셀 정렬. CSS만으로 해결 불가한 이유, JS-measured overlay layer 패턴, 좌표 공간 3종(원본 px / 정규화 / 렌더 px). `type: engineering`.
- [backend/async-servers.md](backend/async-servers.md) — ASGI vs WSGI, uvicorn 내부, FastAPI sync/async 처리, compute-bound 시 async 한계, GPU 싱글턴과 다중 워커 충돌. `type: engineering`.
- [ml/bf16-precision.md](ml/bf16-precision.md) — BF16 vs FP32 비트 배치·Tensor Core 경로·decode argmax drift. `type: research-derived`, 근거: FP32/BF16 observation.
