# AgenticLabeling Optimization Notes

This file is a **deferred-action queue**. Items collected here are performance, efficiency, or code-detail concerns spotted **while building the baseline**. We do not act on them until after the baseline is fully assembled and one full cycle of the program has been run, so that optimization direction is driven by observed behavior rather than speculation.

## Philosophy

> 항상 어떤 조정을 통해 최적화를 시도하기 전, 기존 코드를 바탕으로 시도한 결과를 통해서 최적화 방향을 고민한다. 우선 다 구성하고나서 성능 및 프로그래밍 디테일 항목들을 기록한 뒤, 다 구성되면 프로그램 사이클 한 번 돌리고 최적화를 하나씩 디테일하게 손본다.

Rules of use:

1. **Do not pre-optimize.** Record the concern here and keep building.
2. Each entry captures: location, observation, evidence/hypothesis, priority guess, status.
3. **Priority guesses are tentative** — they will be reordered after the baseline cycle runs.
4. Entries are only promoted to WORKLOG `[DEV]` or `[RESEARCH]` entries when actually acted upon.
5. Rejected candidates stay here (with `status: rejected + reason`) so future-me does not re-add them.

## Entry template

```markdown
### <date> <short title>
- **Location**: `file:line` or feature name or subsystem
- **Observation**: what I noticed while building
- **Evidence / hypothesis**: why this might matter
- **Priority (tentative)**: low / medium / high
- **Blocked until**: which baseline milestone must complete first
- **Status**: noted | evaluating | acted → (worklog link) | rejected (reason)
```

## Entries

### 2026-04-21 이중 image encoding — Florence-2와 SAM3가 같은 이미지를 각자 별도로 인코딩
- **Location**: `mvp_app/detector.py` `_ensure_loaded` + `detect()` (Florence-2 vision tower), `mvp_app/segmenter.py` `segment()` `set_image()` (SAM3 image encoder)
- **Observation**: pipeline concept 작성 중 명시화됨. Florence-2는 `pixel_values [1,3,768,768]` 로 vision tower 통과, SAM3는 `set_image(PIL.Image)` 가 내부 image encoder(1024×1024 급)를 별도 통과. 두 feature space는 아키텍처·pretraining 다르므로 **공유 불가**. 큰 이미지(2816×1584 같은)에서는 downsample + normalize + encoder forward 비용 양쪽에서 반복.
- **Evidence / hypothesis**: 이 중복 자체는 제거 불가(모델 독립성 보장). 그러나 **단계별 완화책**:
  1. 이미지 decode(`Image.open`)는 한 번만 하고 PIL 인스턴스 공유 (현재도 그렇게 함 — OK)
  2. 공통 thumbnail을 두 모델 모두 수용 가능한 해상도로 정해 decode·normalize 중복 줄이기
  3. SAM3 image embedding을 object 수와 무관하게 한 번만 재사용(현재 `set_image` 호출 한 번, 이후 `add_geometric_prompt` 반복 — 이미 OK)
  4. Florence-2 출력 bbox가 여러 개일 때 SAM3 호출을 N회 하지 않고 batched prompt 가능성 확인
- **Priority (tentative)**: low — latency 최적화 대상이지만 "한 번 더 encode" 수준. VRAM은 영향 크지 않음.
- **Blocked until**: 실 사용자 latency 프로파일 후 bottleneck 분포 확인. 현재 warm 791 ms 중 어느 단계가 얼마 차지하는지 per-stage 측정 먼저.
- **Status**: noted
- **Related concept**: [ml/florence-sam-pipeline](../concepts/ml/florence-sam-pipeline.md) §Pipeline composition의 trade-offs

### 2026-04-21 SAM3 cold start 32s → warm 0.8s: 서버 기동 시 eager preload 검토
- **Location**: `mvp_app/segmenter.py` `SegmentationService._ensure_loaded`, `mvp_app/main.py` lifespan
- **Observation**: baseline cycle 측정. 첫 업로드 (SAM3 `uninitialized` 상태): `run_15fa2e393001` **32,658 ms** (~32s). 두 번째 업로드 (이미 warm): `run_6e849d12c6ba` **791 ms**. 즉 SAM3 cold load가 약 31.9s를 차지. Florence-2 cold load는 이미 첫 observation들에서 별도로 관측됨.
- **Evidence / hypothesis**: 현재 서비스는 lazy load 패턴(`_ensure_loaded()` at first call)이라 **첫 사용자가 기다리는 비용이 크다**. lifespan startup에서 `detector._ensure_loaded()`, `segmenter._ensure_loaded()`를 호출해 eager preload하면 보이는 지연을 전부 서버 boot로 이동 — boot 시간은 늘지만 첫 사용자 경험은 즉각적. trade-off: 배포 orchestrator가 health probe를 너무 공격적으로 치면 ready 전에 kill 당할 수 있어 startup probe / readiness probe 분리 필요.
- **Priority (tentative)**: medium — 프로덕션 UX에 직접 영향. 단 이 MVP는 single-tenant dev env라 현재는 체감 크지 않음. 여러 사용자 환경 되면 우선순위 상승.
- **Blocked until**: 한 사이클 완료 후 추가 프로파일링(모델별 개별 시간, lifespan 내 병렬 preload 가능성). Wave C 또는 Deployment Ops 단계에서 같이 다룸.
- **Status**: noted

### 2026-04-21 `sources.file_path` absolute path가 DB에 고정 → 환경 이전 불가
- **Location**: `mvp_app/registry.py:48` `register_source(... file_path=str(stored_path.resolve()))`, `mvp_app/main.py` `/api/assets/{source_id}` FileResponse
- **Observation**: 과거 WSL `/home/coffin/...` 경로가 DB에 그대로 남아 Windows에서 4개 source가 500 에러. 사용자가 Review에서 이들 이미지를 열 수 없음.
- **Evidence / hypothesis**: `register_source`는 absolute path를 저장하고, FileResponse는 그 경로를 그대로 open() 시도. 두 레이어 모두 경로 해석 추상화 없음. 해법: ① `file_path`를 `settings.assets_dir` 기준 상대 경로로 저장 → 열 때 `settings.assets_dir / file_path`로 resolve. ② 또는 파일명만 저장(`stored_path.name`)하고 디렉토리는 runtime settings에서 계산. 마이그레이션: 기존 absolute path 행들을 파일명만 추출해 갱신, 대응 파일이 없으면 `status='failed'`로 표시.
- **Priority (tentative)**: high-ish — 현재 반복 재현 중이고 baseline cycle 의 UX를 해침. 단 Wave C/Deployment Ops보다는 "배경 정리" 성격.
- **Blocked until**: migration 스크립트 작성 + 기존 broken rows 정리 정책 결정(삭제 vs failed tombstone).
- **Status**: noted
- **Related LEARNING**: [파일 경로는 상대 경로 또는 content-addressed 키로 저장한다](LEARNINGS.md)

### 2026-04-20 Florence-2 `_supports_sdpa` 패치로 BF16 속도 회복 시도
- **Location**: Florence-2 HF cache custom modeling (`configuration_florence2.py`, `modeling_florence2.py`)
- **Observation**: FP32 vs BF16 observation에서 BF16 speedup이 1.03×에 그쳤고, `attn_implementation='sdpa'` 사용은 Florence-2 custom class가 `_supports_sdpa` 속성을 선언하지 않아 transformers의 `_sdpa_can_dispatch`가 AttributeError를 던져 막힘 (2026-04-20 WORKLOG `[RESEARCH]` entry 참조).
- **Evidence / hypothesis**: SDPA로 전환하면 Ada Tensor Core의 fused BF16 matmul 경로가 열려 BF16 speedup이 이론치(1.5–2×)에 근접할 가능성.
- **Priority (tentative)**: medium — 추론 처리량이 사용자 체감에 직접 영향. 하지만 UI 기능 공백 해소가 더 우선.
- **Blocked until**: Wave A + Wave B 완료, 한 사이클 업로드→리뷰→export 실행 후.
- **Status**: noted — 원래 태스크 #29로 등록했다가 "build-first" 원칙에 따라 여기로 이관.

### 2026-04-20 Overlay 렌더링 (서버 PIL vs 클라이언트 canvas)
- **Location**: `mvp_app/main.py` `_render_source_overlay()` + `/api/assets/{id}/overlay`, `/api/assets/{id}/bbox-overlay`
- **Observation**: 현재 매 요청마다 서버 PIL로 전체 해상도 PNG를 그려 반환. 대형 이미지(2816×1584 등) + 다수 객체 시 응답 시간·CPU 부담 누적 우려. 클라이언트에서 toggle만 해도 서버 라운드트립.
- **Evidence / hypothesis**: Reference viewer는 CSS/SVG로 bbox를 그리고, mask는 `DeterministicBlob`(mock)로만 그리므로 실질적 마스크 렌더 부담은 아직 프로파일된 적 없음. 실제 사용자 인터랙션 후 어느 경로가 bottleneck인지 확인 필요.
- **Priority (tentative)**: low — 현재 이미 동작, 한 사이클 후 response time 측정으로 재평가.
- **Blocked until**: 한 사이클 후 네트워크 탭·서버 응답 시간 관찰.
- **Status**: noted

### 2026-04-20 `/api/review/workspace` 전체 페이로드 eager 로딩
- **Location**: `mvp_app/main.py` `api_workspace()` — sources 전체 + 각 source의 objects nested 반환
- **Observation**: 소스 수백 개 × 객체 수십 개 누적 시 페이로드 급격 증가. 현재는 DB 작음이라 문제 없으나 MVP 배포·실사용 초기에 빠르게 체감될 가능성.
- **Evidence / hypothesis**: 페이지네이션·lazy loading 경로(예: `/api/sources?limit=20`, source 선택 시 `/api/sources/{id}/objects`) 전환이 자연스러운 다음 단계.
- **Priority (tentative)**: low — 수 ~십 소스 레벨에서는 eager가 오히려 UX 빠름. 실측 전엔 건드리지 않음.
- **Blocked until**: 소스 수 > 100 에서 워크스페이스 초기 로드 시간 측정 후 재평가.
- **Status**: noted

### 2026-04-20 bbox 좌표 변환 이중 표현 (normalized + pixel)
- **Location**: `mvp_app/main.py` `api_workspace()` — `bbox`(normalized) + `bbox_px`(pixel) 둘 다 반환
- **Observation**: 뷰어 코드는 normalized만 사용, pixel은 디버깅·툴팁용으로 반환. 페이로드 2배 + 카테고리별 코스트.
- **Evidence / hypothesis**: 쓰는 쪽이 정해지면 하나만 두는 게 깔끔. UI 완성 후 실제로 `bbox_px` 소비처가 있는지 확인.
- **Priority (tentative)**: low — 용량상 미미.
- **Blocked until**: Wave A/B 완료 후 `bbox_px` 소비처 grep 확인.
- **Status**: noted

### 2026-04-20 `is_validated → validation_status` 전환 시 인덱스 재검토
- **Location**: `mvp_app/storage.py` `idx_objects_is_validated`
- **Observation**: Wave B.2에서 `is_validated` → `validation_status` 전환 시 인덱스도 새 컬럼으로 바꿔야 하고, `WHERE validation_status IS NULL` 쿼리가 partial index / functional index로 더 빠를 가능성.
- **Evidence / hypothesis**: SQLite의 partial index(`WHERE validation_status = 'approved'`)는 자주 쓰는 검증 목록 쿼리를 가속.
- **Priority (tentative)**: medium — 마이그레이션과 함께 갈 때 비용이 낮음. 하지만 지금 당장은 Wave B.2의 스키마 변경만 하고 인덱스는 한 사이클 후 쿼리 프로파일 보고 결정.
- **Blocked until**: Wave B.2 완료 후 `EXPLAIN QUERY PLAN` 분석.
- **Status**: noted

### 2026-04-20 `updated_at` 자동 갱신 (트리거 없음)
- **Location**: `mvp_app/storage.py` objects 테이블 `updated_at`
- **Observation**: 컬럼은 있지만 UPDATE 시 자동 갱신 트리거가 없음. 애플리케이션 코드에서 수동 설정도 안 함.
- **Evidence / hypothesis**: `validation_status` 변경·mask 교체 등에서 `updated_at`이 의미 있는 신호가 되려면 `AFTER UPDATE` 트리거 추가.
- **Priority (tentative)**: low — 기능 영향 없음, 관찰 신호 손실만.
- **Blocked until**: 감사·audit 요구가 실제로 올 때.
- **Status**: noted

---

## Review cadence

- **After each Wave** (A 완료 / B 완료): 한 번 훑고 우선순위 재조정
- **한 사이클 실행 직후**: 각 entry 재평가. 실제 측정된 성능·UX 관찰과 대조해 priority 상향/하향, status 업데이트.
- **Act 결정 시**: WORKLOG에 `[DEV]` 또는 `[RESEARCH]` entry 작성 + 이 파일 `Status: acted → <worklog anchor>` 업데이트.
