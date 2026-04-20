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

### H1 (memory) — **확인**
BF16이 VRAM을 정확히 절반으로 줄인다는 이론치(parameter = 2 bytes)가 그대로 관찰됐다. 모델 로드 시점뿐 아니라 peak VRAM도 같은 비율로 감소한다. 이는 activation/KV-cache도 BF16로 산출되기 때문이다. `use_cache=False` 설정임에도 generate 중 중간 텐서들이 모두 BF16이라 peak가 절반으로 떨어졌다.

### H2 (speed) — **기각에 가까움. 본 조건에서는 이득이 미미**
예상한 1.5–2× speedup 대신 1.03×만 관찰됐다. 유력한 원인 세 가지:

1. **`attn_implementation="eager"` 로 설정됨**. HuggingFace의 eager attention은 vanilla PyTorch matmul/softmax로 구현되며, FlashAttention / SDPA와 달리 Tensor Core 친화적인 fused kernel을 사용하지 않는다. BF16의 실제 속도 이점은 Tensor Core의 BF16 matmul 처리량이 FP32의 2–8×에서 나오는데, eager 구현은 이 경로를 거의 활용하지 못할 가능성이 있다.
2. **Florence-2-base는 작은 모델** (~230M params, VRAM 930 MB 수준). RTX 4090의 이론 FP32 성능이 82 TFLOPs로 이미 매우 빠르고, 작은 모델의 decode loop는 memory-bound보다 compute-bound가 될 여지가 있다. 모델이 커질수록 BF16 speedup이 실질화될 가능성.
3. **Autoregressive decode가 sequential**. `num_beams=1, use_cache=False`로 KV-cache도 없는 순수 sequential 생성이고, `max_new_tokens=1024`지만 실제 출력은 38 토큰(early stop). 토큰마다 overhead가 있어 matmul 시간의 절대량 자체가 작다.

→ 이 셋 중 어느 것이 주된 원인인지는 후속 관찰에서 attention 구현을 바꿔가며 측정해야 확인 가능하다.

### H3 (output stability) — **확인**
레이블은 bit-exact로 동일. bbox는 최대 ~20 px(전체 높이의 0.7 %) 드리프트. 이는 BF16의 mantissa 7 bit 정밀도가 post_process_generation 중 `<loc_N>` 토큰(0..999 양자화) → 원본 이미지 좌표로의 scale-up에서 rounding boundary 인근에서만 다른 bucket으로 넘어가기 때문으로 해석된다. Labeling 품질상 실용적으로 무시 가능한 수준.

재미있는 점: **가장 큰 drift(20 px)가 "두 번째 sky" 박스**에서 발생했다. sky는 경계가 모호(명확한 에지 없이 gradient로 섞임)해서 post_processor가 선택하는 <loc_N> 토큰이 BF16 precision에서는 다른 bucket으로 쉽게 넘어갔을 가능성. 이는 모델이 확신 없는 영역일수록 dtype 차이에 민감해진다는 가설로 연결된다.

### Q1 — 다음 관찰 후보
`attn_implementation="sdpa"` 또는 `attn_implementation="flash_attention_2"`(설치 시)로 바꿔 BF16 speedup이 실제 기대치대로 올라오는지 검증해야 한다. 제품 관점에서도 eager attention 고정은 RTX 4090의 Tensor Core를 낭비한다.

### Q2 — **확인: drift는 decode 단계에서 이미 발생**

`decoder/generate.output_ids`의 raw `.pt` 덤프를 로드해 bit-exact 비교했다. 결과: 두 실행의 output_ids는 **동일하지 않다**. 길이는 38 토큰으로 같지만, 인덱스 [16, 17, 27, 36]의 4개 토큰에서 차이가 있다:

| idx | FP32 | BF16 | Δ |
|---:|---:|---:|---:|
| 16 | 50783 | 50782 | 1 |
| 17 | 50742 | 50741 | 1 |
| 27 | 50764 | 50763 | 1 |
| 36 | 50484 | 50477 | 7 |

네 토큰 모두 Florence-2의 위치 토큰 범위(`<loc_N>`, N ∈ [0, 999], 토큰 id ≈ 50269 + N). 즉 드리프트는 post_process 단계의 scale-up 연산에서만 발생한 것이 **아니라, decode 중 softmax argmax 경계 근처에서 BF16 정밀도로는 다른 위치 토큰이 선택된 결과**다.

인덱스 36의 Δ=7은 다른 세 위치(Δ=1)보다 유독 크다. 이 토큰이 최종 sky box의 `y2` 경계를 결정하는 위치라면, 앞서 관찰된 bbox 최대 drift 19.71 px(0.7%)의 직접적인 원인이다. 모델이 경계에 **확신이 없는 영역일수록 BF16 precision 손실이 다른 output token으로 이어지는 경향**이 이 관찰로 뒷받침된다.

함의: Florence-2 계열 모델의 BF16 drift는 "post-processing 양자화 잡음"이 아니라 "decode 분기점 변경"으로 이해해야 한다. 드리프트를 줄이고 싶다면 post_process를 건드리기보다 decoder 단계에서(예: 더 높은 temperature·beam search·stop-token 조건) 경계 근처 안정화 전략을 고려해야 한다.

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
