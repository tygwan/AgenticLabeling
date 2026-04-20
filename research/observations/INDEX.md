# Observations Index

정식 관찰 기록의 최신순 인덱스. 새 observation을 완료할 때마다 이 파일 상단에 한 줄을 추가한다.

각 observation은 `docs/progress/WORKLOG.md` 에 `[RESEARCH]` entry로 맥락(왜 시작했는지·무엇이 이 실험을 유발했는지)이 함께 기록되어야 한다. 이 인덱스는 빠른 탐색용이고, 스토리는 WORKLOG를 본다.

- [전체 작업 타임라인 → docs/progress/WORKLOG.md](../../docs/progress/WORKLOG.md)

형식:

```
- [YYYY-MM-DD <model> <slug>](<path-to-report.md>) — 한 줄 요약
```

## Entries

- [2026-04-20 florence-2 dtype-fp32-vs-bf16](florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md) — VRAM 절반 확인(0.51x), 속도 이득 1.03x에 그침(eager attention 유력), 출력 drift 최대 0.7%. BF16 전환 결정.
- [2026-04-20 florence-2 florence2-base-fp32](florence-2/2026-04-20-florence2-base-fp32/summary.json) — 위 비교의 FP32 source run. 41 observation points.
- [2026-04-20 florence-2 florence2-base-bf16](florence-2/2026-04-20-florence2-base-bf16/summary.json) — 위 비교의 BF16 source run. 41 observation points.
