---
type: research-derived
primary_evidence: research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/
promoted_from_learnings: raw 샘플 bit-exact 비교; fail-fast type/shape cast
---

# BF16 Precision in Inference

## Concept framework

### IEEE 754 비트 배치 비교

| 포맷 | 총 비트 | Sign | Exponent | Mantissa | 표현 범위 | 정밀도(ULP) |
|---|---|---|---|---|---|---|
| **FP32** | 32 | 1 | 8 | 23 | ±1.2e-38 ~ 3.4e38 | ~1.2e-7 |
| **FP16** | 16 | 1 | **5** | 10 | ±6e-5 ~ 65504 | ~1e-3 |
| **BF16** | 16 | 1 | **8** | **7** | FP32와 **동일** | ~8e-3 |

핵심 통찰:

- **BF16 = FP32의 exponent를 그대로 + mantissa만 절단**. 값 범위는 그대로 유지하면서 정밀도만 희생한다.
- **FP16은 범위가 좁아** attention logits의 큰 값·softmax 분모에서 overflow/NaN이 쉽게 발생. BF16은 이 문제가 없음.
- **BF16 정밀도는 FP32 대비 ~2^16배 거침**. 수치 계산이 분기(argmax, threshold, assignment) 근처에서 다른 답을 낼 수 있음.

### Tensor Core 가속 경로

NVIDIA Ampere(A100), Ada(RTX 40), Hopper(H100) 아키텍처의 Tensor Core는 BF16 matmul을 FP32 대비 **2~8배 throughput**으로 처리한다. 그러나 이득은 attention 구현이 Tensor Core-친화적일 때만 실현된다:

- `attn_implementation="eager"` (vanilla PyTorch matmul/softmax): 일반 CUDA kernel. Tensor Core 활용 제한적.
- `attn_implementation="sdpa"` (torch.nn.functional.scaled_dot_product_attention): fused kernel에 dispatch, Tensor Core 활용.
- `attn_implementation="flash_attention_2"`: 가장 공격적 fusion, Tensor Core 최대 활용.

**→ BF16 속도 이득은 attention 구현에 크게 의존한다**. 모델 가중치를 BF16으로 바꾸는 것만으로는 자동으로 2배 빨라지지 않는다.

### Autoregressive decode에서의 drift

VLM/LLM 의 `generate()` 는 매 step 마다 logits → softmax → argmax(혹은 sampling) 를 반복하는 sequential 과정이다. 각 step에서:

- Logits가 거의 같은 두 token 사이의 **argmax boundary**에 있으면, BF16 정밀도 손실이 **다른 token 선택**으로 이어질 수 있다.
- 한 번 다른 token이 선택되면 **그 뒤의 context가 달라져** 누적 divergence가 생긴다.
- 그러나 대부분의 step은 argmax가 확실해 drift가 없다. **drift는 model이 확신이 없는 boundary case에 국한**된다.

### 왜 production에서 BF16이 주류인가

- VRAM **절반** (parameters = 2 bytes)
- Tensor Core 경유 시 **속도 2-8×**
- FP16 대비 **overflow/NaN 회피**
- 추론 품질 실용적으로 **FP32와 구분 불가**인 경우가 대다수 (boundary case만 drift)
- 학습·미세조정에서도 mixed precision(BF16 matmul + FP32 master weights)이 표준

→ NVIDIA 최근 세대, HuggingFace 기본 권장 dtype, vLLM/TGI 기본 설정 모두 BF16.

## Evidence in this project

### 1. VRAM 절반 — 이론치 그대로 달성

- **Florence-2-base 모델 로드 후 VRAM**: FP32 930.86 MB → BF16 474.53 MB (**0.510×**)
- **Peak VRAM (generate 중)**: FP32 1270.55 MB → BF16 650.08 MB (**0.512×**)
- 출처: [obs:2026-04-20-dtype-fp32-vs-bf16](../../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md) §VRAM

### 2. 속도 이득 미미 — eager attention 우회 확인

- median `generate()` 시간: FP32 176.14 ms → BF16 170.41 ms (**1.03×**에 그침)
- 예상 1.5~2× speedup과 큰 격차.
- 동일 observation에서 `attn_implementation="sdpa"`를 시도했으나 Florence-2 custom modeling(`Florence2ForConditionalGeneration`)이 `_supports_sdpa` 속성을 선언하지 않아 transformers가 dispatch를 거부.
- → Tensor Core 경로가 열리지 않아 BF16의 계산 이득이 실현되지 않음.
- 출처: [obs:2026-04-20-dtype-fp32-vs-bf16](../../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md) §Timing, [WORKLOG 2026-04-21 SDPA RESEARCH](../../progress/WORKLOG.md)

### 3. 출력 drift는 decode argmax boundary에서 발생

- 레이블은 FP32/BF16 **6개 모두 동일**.
- bbox 드리프트 **최대 19.71 px** (전체 height 2816의 0.7%), 특히 경계 모호한 "sky" 박스에서 집중.
- `generate.output_ids` bit-exact 비교 시 4개 위치 [0,16], [0,17], [0,27], [0,36] 에서 다른 token 선택 — 모두 `<loc_N>` 위치 토큰 범위(~50000번대).
- → drift는 post_process 단계가 아니라 **decode 단계의 argmax 분기에서** 발생. mantissa 정밀도 손실이 logit argmax의 경계를 넘기는 케이스에서만 다른 token이 선택됨.
- 출처: [obs:2026-04-20-dtype-fp32-vs-bf16 §Q2](../../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md)

### 4. 모델에 BF16을 주입할 때 입력도 cast해야 한다

- BF16 모델 로드 후 processor가 반환한 `pixel_values(float32)`를 그대로 forward 시 `RuntimeError: Input type (float) and bias type (BFloat16) should be the same`.
- 해법: `mvp_app/detector.py`의 `.detect()` 내부에서 model의 param dtype을 읽어 floating point 입력을 그 dtype으로 cast.
- 출처: `mvp_app/detector.py:125-139` (`model_dtype = next(self._model.parameters()).dtype; v.to(device=self.device, dtype=model_dtype)`)

## Related rules / decisions

- [LEARNING: raw 샘플을 bit-exact로 비교한다](../../progress/LEARNINGS.md) — 본 concept의 Evidence 3이 이 규칙의 근거 사례
- [LEARNING: type/shape mismatch는 경계에서 fail-fast](../../progress/LEARNINGS.md) — Evidence 4가 직접 사례
- [LEARNING: 외부 라이브러리 상한 pin](../../progress/LEARNINGS.md) — `_supports_sdpa` 누락이 이 규칙과 다른 맥락(업스트림 custom code의 SDK 호환성 이슈)에서 관련
- [WORKLOG 2026-04-20 [DECISION] FLORENCE_DTYPE 도입](../../progress/WORKLOG.md)
- [WORKLOG 2026-04-21 [RESEARCH] SDPA attention 시도 → 불가](../../progress/WORKLOG.md)

## Open follow-ups

- Florence-2-large 로 같은 비교 반복 — 모델이 커질수록 BF16 속도 이득이 eager 경로에서도 더 크게 드러나는지 확인
- HF cache의 Florence-2 custom modeling에 `_supports_sdpa` 패치 시도 — 성공 시 Tensor Core 경로 열려 2× speedup 기대
- SAM3 경로의 BF16 동작 확인 — 현재 FP32만 테스트됨
