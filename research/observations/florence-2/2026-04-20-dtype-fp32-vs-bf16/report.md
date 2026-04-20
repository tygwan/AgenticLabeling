# Florence-2 FP32 vs BF16 — Observation Report

## Summary

Florence-2-base 모델을 FP32와 BF16 두 compute dtype으로 실행해 같은 입력 이미지·클래스 프롬프트에서의 **추론 속도, VRAM 사용량, 출력 안정성**을 비교했다. **VRAM은 이론치대로 정확히 절반**으로 줄었고(모델 930.86 → 474.53 MB, peak 1270.55 → 650.08 MB), **출력은 6개 모두 레이블 동일·bbox 미세 차이(최대 ~20 px / 2816 px ≈ 0.7 %)**로 실용상 동일했다. 그러나 **속도 이득은 1.03×에 그쳤다** — 모델이 작고 `attn_implementation="eager"`가 Tensor Core 경로를 우회한 점이 유력한 원인으로 추정된다. BF16으로의 전환은 VRAM 관점에서 명확히 이득이며, 속도 이득을 확보하려면 attention 구현 교체를 다음 관찰에서 검증해야 한다.

## Setup

- 입력: `data/images/test_street.jpg` (1584×2816, sha256 `8414df42…`)
- 프롬프트 클래스: `car, person, road, building, sky`
- 모델: `microsoft/Florence-2-base`
- GPU: NVIDIA GeForce RTX 4090, 24 GB, CC 8.9
- 소프트웨어: torch 2.11.0+cu126, transformers 4.57.6, triton-windows 3.6.0.post26
- Generation 설정: `do_sample=False, num_beams=1, max_new_tokens=1024, use_cache=False`
- Attention 구현: `attn_implementation="eager"` (현 MVP 기본값)
- 측정: warmup 1회 + measured 3회, median / min / max 보고, `torch.cuda.synchronize()` 브래킷

Source observations:
- `../2026-04-20-florence2-base-fp32/summary.json`
- `../2026-04-20-florence2-base-bf16/summary.json`

## Observations

### 1. VRAM

| 항목 | FP32 | BF16 | BF16/FP32 |
|---|---:|---:|---:|
| 모델 VRAM (로드 직후, `torch.cuda.memory_allocated`) | 930.86 MB | 474.53 MB | **0.510×** |
| Peak VRAM (generate 중, `max_memory_allocated`) | 1270.55 MB | 650.08 MB | **0.512×** |

### 2. Wall-clock 추론 시간 (`model.generate(...)` 한 회)

| 항목 | FP32 | BF16 |
|---|---:|---:|
| Run 1 | 176.14 ms | 170.57 ms |
| Run 2 | 175.08 ms | 170.41 ms |
| Run 3 | 179.55 ms | 164.54 ms |
| **Median** | **176.14 ms** | **170.41 ms** |
| Min | 175.08 ms | 164.54 ms |
| Max | 179.55 ms | 170.57 ms |

BF16 median speedup: **1.03×** (FP32 대비 ~3.3 % 단축)

### 3. 출력 안정성

**레이블**: 6개 모두 동일 — `['car','person','road','building','sky','sky']`

**Bounding box (xyxy, px)**:

| # | label | FP32 | BF16 | Δ (max 변화) |
|---|---|---|---|---:|
| 0 | car | `[0.79, 1.41, 1580.04, 2811.78]` | `[0.79, 1.41, 1580.04, 2811.78]` | 0.00 |
| 1 | person | `[731.02, 1175.68, 814.97, 1333.38]` | `[731.02, 1175.68, 813.38, 1330.56]` | 2.82 |
| 2 | road | `[0.79, 1071.49, 1581.62, 2811.78]` | `[0.79, 1071.49, 1581.62, 2811.78]` | 0.00 |
| 3 | building | `[0.79, 1.41, 407.88, 1395.33]` | `[0.79, 1.41, 407.88, 1392.51]` | 2.82 |
| 4 | sky | `[247.90, 1.41, 1581.62, 863.10]` | `[247.90, 1.41, 1581.62, 863.10]` | 0.00 |
| 5 | sky | `[249.48, 1.41, 1581.62, 606.85]` | `[249.48, 1.41, 1581.62, 587.14]` | **19.71** |

최대 drift는 box#5(sky)의 하단 경계에서 **19.71 px** (≈ 0.70 % of 2816 px 높이).

### 4. 공통 메타데이터

- 입력 토큰 수 (`input_ids`): 27
- 생성 토큰 수 (`output_ids`): 38 (두 실행 모두 동일)
- `pixel_values` shape `[1, 3, 768, 768]` (FP32 측정 기준, BF16 실행 시 모델 dtype으로 cast)
- `post_process/bboxes` 길이 두 실행 모두 7 (내부), 그 중 6개가 요청 클래스로 매칭되어 `output/boxes`로 전달

## Interpretation

해석(개념 수준의 설명)은 [docs/concepts/ml/bf16-precision.md](../../../../docs/concepts/ml/bf16-precision.md) 의 Concept framework 섹션으로 위임됨. 이 observation의 수치·측정은 같은 concept 파일의 `Evidence in this project` 섹션 1-4 번에 기여한다.

이 observation에 **국한된 결정**만 아래에 남긴다:

- 이 cycle 이후 제품 dtype 기본값을 BF16로 전환 결정 (DECISION 링크: [WORKLOG 2026-04-20 FLORENCE_DTYPE](../../../../docs/progress/WORKLOG.md))
- 속도 이득 원인 추적을 위한 후속 observation: attn_implementation 변경 · Florence-2-large 확장

## Evidence contribution

이 observation이 `docs/concepts/ml/bf16-precision.md` 에 기여하는 evidence:

- **§Evidence 1 — VRAM 절반** : 모델 로드 후 930.86 → 474.53 MB (0.510×), peak 1270.55 → 650.08 MB (0.512×)
- **§Evidence 2 — 속도 이득 미미** : generate median 176.14 ms → 170.41 ms (1.03×), eager attention 원인 가설
- **§Evidence 3 — decode argmax boundary 분기** : output_ids 4 토큰 위치([0,16], [0,17], [0,27], [0,36])에서 다른 `<loc_N>` 선택. sky 박스의 20 px bbox drift와 연결.
- **§Evidence 4 — 입력 dtype cast 필요** : Input type (float) and bias type (BFloat16) should be the same 에러 → detector가 model_dtype으로 cast (관련 evidence는 코드에서 오는 것이지만 본 observation에서 문제 재현됨)

## Decision

향후 Florence-2 추론을 **BF16 기본값**으로 전환한다. 근거:

- VRAM 절반(930 → 475 MB) 확보 → 더 큰 배치 / 더 큰 이미지 / SAM3 동시 로드 여지
- 출력 품질 손실 실용적으로 없음 (labels 동일, bbox drift 0.7 % 미만)
- 속도는 현재 조건에서 이득이 작지만, attention 구현 전환 후 재측정 예정

실행 방법: `FLORENCE_DTYPE=bfloat16` 환경변수 설정. `mvp_app/detector.py`에 이미 이 경로가 연결되어 있음(`_resolve_dtype`). 기본값은 호환성을 위해 `float32`로 유지하고, 배포 환경에서 env로 override한다.

## Follow-ups

- [ ] **Q1 검증**: `attn_implementation="sdpa"` 로 바꿔 BF16 speedup 재측정. 기대 1.5–2×.
- [x] **Q2 검증**: `generate.output_ids.pt` raw 덤프를 FP32/BF16 쌍으로 로드해 bit-exact 비교 (위 "Q2 — 확인" 섹션).
- [ ] **Florence-2-large** 로 같은 비교 반복. 모델이 커질수록 BF16 이득 커질 것으로 예상.
- [ ] **SAM3 쪽도 BF16 경로** 가능한지 확인. 현재는 float32만 테스트됨.
- [ ] **배치/해상도 확장 실험**: BF16으로 확보된 VRAM을 활용해 batch=N, 더 큰 해상도를 넣었을 때 throughput 증가 측정.
- [x] **`analysis.ipynb` 에 시각화**: bbox 오버레이 비교, 토큰 시퀀스 diff, VRAM/시간 bar chart (실행·outputs 포함 commit).
