# LEARNINGS

마찰(friction) 이벤트에서 뽑아낸 일반 규칙 후보들의 curated journal.

- **적용 기준**: [docs/standards/learn-from-friction.md](../standards/learn-from-friction.md)
- **목적**: 프로젝트 특이성이 아니라 **프로그래밍 일반 원칙**을 골라 draft → validated → promoted 생애주기로 검증해 `docs/standards/`에 승격
- **관련 기록소**:
  - 시간순 개발·연구 feed: [docs/progress/WORKLOG.md](WORKLOG.md)
  - 최적화 후보: [docs/progress/OPTIMIZATION-NOTES.md](OPTIMIZATION-NOTES.md)

## 생애주기 요약

| 상태 | 의미 |
|---|---|
| `draft` | 처음 기록됨, 관찰만 |
| `validated` | 다른 맥락에서 재발생 확인됨 |
| `promoted` | dev-standards에 승격됨 (관련 standards 링크 포함) |
| `rejected` | 실용성 없음 또는 잘못된 추론 (삭제 금지) |
| `stale` | 30일+ 재발생 없음 (정리 후보) |

## Entry format

각 entry는 learn-from-friction Rule 3을 따른다. 추가 시 상단에 최신순으로 삽입.

---

## Entries

### 2026-04-21 contain-fit 이미지 위의 오버레이는 JS로 측정한 레이어에 배치한다
- **Trigger type**: design + repetition
- **Triggered by**: Review 뷰어에서 사용자가 "이미지가 화면에 다 안들어오고 크게보여서 일부 잘리는 모습" 보고. 순수 CSS(`aspect-ratio + max-content`, `width:100% + object-fit: contain` 등)로 contain-fit + 오버레이 정렬을 모두 만족시키려 시도했으나 각각 다른 방식으로 실패.
- **Evidence**: `mvp_app/static/components/viewer.jsx` `measure()` 로직, `mvp_app/static/styles.css` `.viewer-canvas` / `.viewer-overlay-layer`. 상세한 기초·심화 배경은 [concepts/frontend/image-viewer-overlay-alignment.md](../concepts/frontend/image-viewer-overlay-alignment.md).
- **Rule (draft)**: 이미지에 `object-fit: contain`으로 letterbox fit을 적용하고 그 위에 `position:absolute` 오버레이(bbox, mask 등)를 **% 좌표**로 얹어야 할 때, CSS만으로 오버레이를 렌더 영역(letterbox 제외)에 정렬하는 것은 해법이 빈약하다. 대신 JS로 `img.naturalWidth/Height` + `img.getBoundingClientRect()`로 **실제 렌더 rect**를 계산해 **별도 overlay layer** 엘리먼트의 `left/top/width/height`에 설정하고 오버레이는 그 레이어 내부 %로 배치한다. 재계산 트리거는 최소 `onLoad`, `window resize`, `source 변경` 세 가지.
- **Generality**: 이미지 리뷰·어노테이션 도구(Roboflow, LabelStudio, CVAT 계열), PDF·미디어 viewer, 지도 오버레이 등 "fit된 컨텐츠 + 좌표계 보존 오버레이"가 필요한 모든 웹 UI에 동일 원칙.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [concept: image-viewer-overlay-alignment](../concepts/frontend/image-viewer-overlay-alignment.md), [WORKLOG 2026-04-21 viewer 정렬 수정](WORKLOG.md)

### 2026-04-21 SQLite 마이그레이션에서 DDL 의존성(CREATE INDEX 등)은 컬럼 ALTER 뒤로 미룬다
- **Trigger type**: error
- **Triggered by**: 서버 기동 시 `sqlite3.OperationalError: no such column: validation_status` 에러. 원인: `SCHEMA_SQL` 블록에 `CREATE INDEX IF NOT EXISTS idx_objects_validation_status ON objects(validation_status)` 를 넣었는데, 기존 DB의 `objects` 테이블에는 아직 해당 컬럼이 없었다. `CREATE TABLE IF NOT EXISTS`는 기존 테이블을 수정하지 않으므로, 같은 executescript 안에서 `CREATE INDEX`가 실패.
- **Evidence**: `mvp_app/storage.py` `SCHEMA_SQL` 주석 + `init_storage` 에서 `_ensure_column` 호출 뒤 별도 `CREATE INDEX IF NOT EXISTS ... ON objects(validation_status)` 실행. 수정 commit: TBD.
- **Rule (draft)**: SQLite처럼 "ALTER TABLE ADD COLUMN IF NOT EXISTS"가 없는 엔진에서 기존 테이블에 신규 컬럼을 도입할 때, **컬럼을 참조하는 다른 DDL**(index, trigger, view, check constraint, FK)은 **컬럼 존재 보장 후에 실행**해야 한다. 초기 스키마 블록(`CREATE TABLE IF NOT EXISTS`)과 마이그레이션 블록(`ALTER TABLE ADD COLUMN` via PRAGMA 체크)을 분리하고, 신규 컬럼을 참조하는 DDL은 마이그레이션 블록의 **뒤에** 둔다.
- **Generality**: SQLite·MySQL 구버전·PostgreSQL 9.x 등 `ADD COLUMN IF NOT EXISTS` 지원이 제한되거나 index-on-new-column 의존성이 있는 모든 SQL 엔진에 동일. 단, Postgres ≥ 9.6은 `ADD COLUMN IF NOT EXISTS`를 지원하니 그 경우 덜 중요.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [LEARNING: 스키마 변경은 idempotent + 덧붙임(additive)만으로 한다](LEARNINGS.md) — 이 LEARNING의 구체적 구현 주의사항. [WORKLOG 2026-04-21 baseline cycle 시작 서버 기동 실패 수정](WORKLOG.md)

### 2026-04-21 외부 라이브러리가 구버전 API를 사용하면 상한을 명시적으로 pin한다
- **Trigger type**: error
- **Triggered by**: `transformers==5.5.4` 자동 해석 후 Florence-2 custom code 로드 시 `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
- **Evidence**: 2026-04-20 WORKLOG `[DEV] GPU 환경 전환` entry, 원인 추적 및 `requirements.txt`의 `transformers>=4.36.0,<5.0` 수정 (commit `cc23919`)
- **Rule (draft)**: 라이브러리가 `trust_remote_code=True` 나 플러그인·확장 같은 동적 로드 경로를 통해 **외부 custom 코드**를 실행하는 경우, 해당 custom 코드가 타겟팅한 주버전(major)에 상한(`<N+1.0`)을 두어라. custom 코드는 주버전 breaking change와 호환되지 않는 경우가 많고, 상한 없이는 pip가 최신을 골라 배포 직후 터진다.
- **Generality**: Python(transformers, timm 등 커스텀 loader), JS(eslint/babel plugins), Rust(proc-macros), 어느 언어든 "동적 로드되는 third-party code + 그 안에서 SDK API 호출" 구조면 동일하게 적용.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [WORKLOG 2026-04-20 GPU 환경 전환](WORKLOG.md), commit `cc23919`

### 2026-04-21 테스트는 API 계약 레이어에서 작성하고 UI 텍스트는 E2E에서만 검증한다
- **Trigger type**: design + repetition
- **Triggered by**: SPA 전환 후 `assert "BBox Overlay" in review.text` 테스트 2건이 일괄 깨짐. 원인: 서버 렌더링 HTML 문자열이 클라이언트 렌더링으로 이동하며 소스에서 사라짐.
- **Evidence**: `tests/test_mvp_app.py:61-63`, `tests/test_mvp_e2e.py:79-80` 의 리팩토링 diff (commit `ec0d199`)
- **Rule (draft)**: 렌더링 계층이 바뀔 수 있는 프로젝트에서 단위/통합 테스트는 **안정 계층**(JSON API shape, DB 상태, 서비스 계약)에 붙인다. UI 문자열·DOM 구조는 **E2E·브라우저 테스트**에서만 검증한다. 이 두 레이어가 섞이면 presentation 변경 한 번에 테스트 대량 깨짐이 발생한다.
- **Generality**: SPA 전환, 템플릿 엔진 교체, 디자인 시스템 마이그레이션, 프론트 프레임워크 업그레이드 어느 상황이든 동일.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [WORKLOG 2026-04-20 UI 전체 구현](WORKLOG.md), commit `ec0d199`

### 2026-04-21 타입·shape mismatch는 경계에서 fail-fast로 막는다
- **Trigger type**: error
- **Triggered by**: BF16 모델 로드 후 processor가 반환한 `pixel_values(float32)`를 그대로 forward 시 `RuntimeError: Input type (float) and bias type (struct c10::BFloat16) should be the same`.
- **Evidence**: 2026-04-20 FP32/BF16 observation 재실행 중 발생, `mvp_app/detector.py` 에 `to(device=..., dtype=model_dtype)` cast 로직 추가 (commit `ec0d199`)
- **Rule (draft)**: 컴포넌트(모델·서비스·함수)가 특정 타입/shape 계약을 갖는 경우, 호출자 책임에 떠넘기지 말고 **경계에서 명시적으로 cast/verify** 한다. 조용한 dtype 변환(auto-promote)·부분 broadcast에 의존하면 디버그 비용이 훨씬 커진다. cast 실패 시 의미 있는 에러로 즉시 터트려라.
- **Generality**: 정적 타입 언어(Go/Rust/Scala)의 경계에서 `into()`/`try_into()`, 동적 타입 언어(Python/JS)에서 typeguard·runtime assertion·pydantic 모델, ML dtype 캐스트, 직렬화 boundary(JSON/Proto) 모두 동일 원칙.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [WORKLOG 2026-04-20 UI 전체 구현 (dtype cast)](WORKLOG.md), commit `ec0d199`

### 2026-04-21 동등성을 주장하기 전에 raw 샘플을 bit-exact로 비교한다
- **Trigger type**: observation
- **Triggered by**: FP32 vs BF16 관찰에서 `florence_result.labels`·bbox stats(min/max/mean/std)로는 큰 차이가 없어 보였으나, raw `output_ids.pt`를 불러와 `torch.equal(fp, bf)` 비교 결과 4개 위치의 token ID가 실제로 달랐음(decode 단계에서 divergence).
- **Evidence**: `research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md` Q2 섹션, `analysis.ipynb` output_ids diff cell (commit `8119372`, `ec0d199`)
- **Rule (draft)**: 두 파이프라인·모델·구현이 "동등하다"고 주장하기 전에 **raw 출력의 sample-wise 또는 bit-exact 비교**를 수행한다. 집계 통계(min/max/mean/std)는 분기점 근처의 분산된 작은 차이를 덮어 가린다. 두 가지 중 하나라도 어긋나면 분기점·precision·ordering 중 어디에 divergence가 있는지 찾을 때까지 파고들어라.
- **Generality**: 리팩토링 전후 동등성 검증, 모델 quantization/pruning 검증, deterministic replay, 다른 언어로 포팅한 알고리즘의 참조 구현 대조, DB query plan 변경 전후 결과 비교 모두에 적용.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md](../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md)

### 2026-04-21 관찰 코드는 외부 hook으로 주입하고 관찰 대상 코드를 수정하지 않는다
- **Trigger type**: design
- **Triggered by**: 연구 관찰 프로토콜 설계 시 "production 코드에 observation용 print/log를 심으면 코드가 오염되고 책임 경계가 흐려진다"는 문제. PyTorch `register_forward_hook` 패턴을 관찰 전용 helper(`research/tooling/hooks.py`)로 분리.
- **Evidence**: `docs/standards/research-observation-protocol.md` Rule 6, `research/tooling/hooks.py` `HookRegistry` 클래스 (commit `fa73536`)
- **Rule (draft)**: observability 코드(logging, metrics, profiling, debug probe, 연구용 trace)는 관찰 대상 시스템을 **수정하지 않고 외부에서 hook API로 주입**한다. 공식 hook API가 없다면 subclass/wrapper/decorator 등으로 바깥에서 감싸라. production 코드 안에 `print`·`logger.debug`를 직접 심는 방식은 제거 비용이 누적되고 코드 의도를 흐린다.
- **Generality**: APM/tracing 통합(OpenTelemetry), 디버그 로깅, 성능 프로파일링, 연구 instrumentation, Python decorator 기반 계측, Go middleware 기반 로깅 모두 같은 원칙. 언어 무관.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [docs/standards/research-observation-protocol.md Rule 6](../standards/research-observation-protocol.md)

### 2026-04-21 스키마 변경은 idempotent + 덧붙임(additive)만으로 한다
- **Trigger type**: design
- **Triggered by**: Run 엔티티 추가 시 기존 `data/mvp/sqlite/mvp.db`에 새 테이블을 자동 반영해야 했음. `CREATE TABLE IF NOT EXISTS runs (...)`를 사용해 init_storage 재호출만으로 migration 완료.
- **Evidence**: `mvp_app/storage.py` SCHEMA_SQL의 `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` (Wave A.1)
- **Rule (draft)**: 스키마 evolution은 **idempotent(재실행 가능)** + **forward-compatible(덧붙임만)** 원칙으로 한다. `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN ... DEFAULT ...`, `CREATE INDEX IF NOT EXISTS`만 사용하고 `DROP/RECREATE`는 명시적 deprecation 이후에만 허용한다. 이렇게 하면 기존 데이터는 보존되고 구·신 코드가 동시에 동작 가능한 전이기(transitional)를 얻는다.
- **Generality**: SQL DB(SQLite/Postgres/MySQL), NoSQL 스키마(Mongo, DynamoDB), protobuf 메시지 evolution, REST API versioning, GraphQL schema 모두 동일 원칙. "기존 데이터/클라이언트를 깨지 않는 덧붙임"은 모든 stateful 시스템의 공통 진화 전략.
- **Recurrence**: 1
- **Status**: draft
- **Related**: [WORKLOG 2026-04-21 Wave A.1](WORKLOG.md), `mvp_app/storage.py`
