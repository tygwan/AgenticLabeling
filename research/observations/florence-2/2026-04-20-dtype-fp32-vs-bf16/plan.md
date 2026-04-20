# Observation Plan — Florence-2 FP32 vs BF16

- **Date**: 2026-04-20
- **Slug**: 2026-04-20-dtype-fp32-vs-bf16
- **Model**: `microsoft/Florence-2-base`
- **Hardware**: NVIDIA GeForce RTX 4090 (Ada, CC 8.9), 24 GB VRAM
- **Software**: torch 2.11.0+cu126, transformers 4.57.6, triton-windows 3.6.0.post26

## Inputs

- `data/images/test_street.jpg` sha256=`8414df421ee3ea605fdd94cf7ee267cf67c5634b76a92aac9b228f281353942d`
- Classes prompt: `car, person, road, building, sky`

## Observation layer

- [x] pipeline
- [x] model (compute dtype swap)
- [ ] module
- [ ] tensor (covered separately by per-dtype observations)

## Source observations (to be compared)

- `../2026-04-20-florence2-base-fp32/summary.json`
- `../2026-04-20-florence2-base-bf16/summary.json`

Each source observation runs the same image through the same pipeline, differing only in `FLORENCE_DTYPE`.

## Hypotheses / questions

1. **H1 (memory)**: BF16 halves the model VRAM footprint because parameters are 2 bytes instead of 4. *Expected ratio ≈ 0.5.*
2. **H2 (speed)**: BF16 gives a 1.5–2× wall-clock speedup on Ada GPUs via Tensor Cores for the dominant matmul-heavy workload.
3. **H3 (output stability)**: Labels remain identical; bbox coordinates drift by a small, bounded amount (mantissa precision loss).
4. **Q1**: Does `attn_implementation="eager"` (set in `mvp_app/detector.py`) prevent Tensor Core usage and hence suppress the BF16 speedup?
5. **Q2**: Where exactly does BF16 drift show up — post-processor `<loc_N>` tokens, or in the continuous bbox values after `post_process_generation`?

## Measurement protocol

- GPU warmup: 1 generate() call before timing (kernel compile / cache warm).
- Measured runs: 3 generate() calls, reported as median / min / max.
- `torch.cuda.synchronize()` bracketing every timed call.
- `torch.cuda.reset_peak_memory_stats()` before first measured run; `torch.cuda.max_memory_allocated()` after.
- Input image, classes, and seed-free inference (do_sample=False, num_beams=1) held constant.
