# AgenticLabeling Research Observation Protocol

## Purpose

AgenticLabeling은 제품이자 연구 산출물이다. 연구의 핵심은 Florence-2, SAM3 등 모델 파이프라인 내부에서 발생·이동하는 데이터의 타입·형식·형태를 **관찰하고 기록**하는 것이다.

이 문서는 그 관찰 행위가 일관되고 재현 가능하며 포트폴리오 자산이 되도록 규칙을 정한다.

## Scope

아래 작업에 이 프로토콜이 적용된다:

- 모델 아키텍처를 계층 단위로 분해해 데이터 흐름을 추적하는 작업
- 특정 입력(이미지/텍스트/프롬프트)을 넣어 단계별 출력을 기록하는 작업
- 중간 텐서, attention map, 임베딩을 시각화·저장하는 작업
- 모델 간(예: Florence-2 → SAM3) 데이터 경계를 문서화하는 작업

제품 기능 개발(UI, API 확장, 리팩토링)에는 적용되지 않는다. 단, 기능 개발이 관찰 포인트를 깨트리면 이 프로토콜이 사전 검토 대상이 된다.

## Rule 1. 관찰 단위는 4계층 중 명시한다

모든 관찰 작업은 다음 단위 중 어느 계층에서 수행되는지 리포트에 선언한다:

1. **pipeline** — 모델 전체 I/O (Florence-2 → SAM3 같은 체인 단위)
2. **model** — 단일 모델 I/O (예: Florence-2 `generate()` 입력/출력)
3. **module** — 모델 내부 서브모듈 (예: vision encoder, text decoder, mask head)
4. **tensor** — forward hook 기반 개별 레이어 출력, attention weight, 중간 임베딩

연구의 기본값은 **tensor 단위까지 내려간다**. 더 얕은 계층만 기록하는 경우 사유를 리포트에 남긴다.

## Rule 2. 관찰 포인트는 사전에 등록한다

관찰 스크립트를 실행하기 전, `research/observations/<model>/<date>-<slug>/plan.md`에 다음을 적는다:

- 어떤 입력을 사용하는가 (파일 경로, hash)
- 어느 계층에서 관찰하는가 (Rule 1의 단위)
- 어떤 포인트를 관찰하는가 (예: `vision_tower.layer[4].self_attn.attn_weights`)
- 왜 관찰하는가 (가설, 의문, 확인 대상)

등록 없이 실행된 관찰은 임시 실험(`research/experiments/`) 영역에 남기고, 정식 observation으로 승격하려면 계획서를 작성한다.

## Rule 3. 기록 형식은 이중 트랙이다

모든 observation은 아래 두 산출물을 함께 생산한다:

**Markdown 리포트 (`report.md`)**
- 관찰 목적과 가설
- 관찰 포인트 리스트
- 각 포인트의 type/shape/dtype/stats 표
- 관찰 결과에 대한 해석 (무엇이 의외였는가, 어떤 적용을 결정했는가)
- 관련 코드 변경이나 설계 의사결정으로의 링크

**Jupyter 노트북 (`analysis.ipynb`) — outputs 포함**
- 관찰 포인트별 시각화 (attention heatmap, tensor stats, 분포 히스토그램)
- 배열/텐서를 인라인으로 렌더링
- 생각의 흐름 (markdown 셀) 포함
- GitHub 상에서 바로 렌더되도록 outputs를 **지우지 않는다**

두 산출물은 같은 관찰 포인트를 가리키되, 리포트는 서술·의사결정을 담고, 노트북은 시각적 근거를 담는다.

## Rule 4. 재현성 메타데이터를 함께 기록한다

모든 observation 폴더는 `summary.json`을 포함한다:

```json
{
  "model": "florence-2",
  "model_id": "microsoft/Florence-2-base",
  "commit": "<git rev-parse HEAD>",
  "date": "2026-04-20",
  "inputs": [{"path": "data/images/test_street.jpg", "sha256": "..."}],
  "observation_points": [
    {
      "name": "pixel_values",
      "layer": "processor output",
      "type": "torch.Tensor",
      "shape": [1, 3, 768, 768],
      "dtype": "float32",
      "device": "cpu",
      "stats": {"min": -2.11, "max": 2.64, "mean": 0.12, "std": 0.98}
    }
  ],
  "python_version": "3.12.10",
  "key_packages": {"torch": "2.11.0", "transformers": "5.5.4"}
}
```

이 파일은 기계 판독 가능해야 하며, 나중에 포트폴리오 페이지나 인덱스 스크립트가 이를 파싱한다.

## Rule 5. 원본 텐서 덤프는 git에 넣지 않는다

다음은 **커밋한다**:

- `report.md`
- `analysis.ipynb` (outputs 포함)
- `summary.json`
- `plan.md`
- 리포트·노트북에 임베드되는 핵심 시각화 (`assets/*.png`, `assets/*.svg`)

다음은 **ignore한다** (재현 스크립트로 복원 가능):

- `*.npz`, `*.pt`, `*.safetensors`
- `outputs/raw/**` 대용량 중간 덤프

대용량 시각화를 반드시 보존해야 하면 그때 Git LFS 도입을 검토한다. 기본값은 LFS 없이 간다.

## Rule 6. 관찰은 프로덕션 코드를 오염시키지 않는다

observation 전용 코드는 다음 위치에만 둔다:

- `scripts/inspect_*.py` — 실행 가능한 관찰 도구
- `research/tooling/` — 관찰 전용 유틸리티 (hook 등록 helper 등)

`mvp_app/detector.py`, `mvp_app/segmenter.py` 등 프로덕션 코드에는 관찰 목적의 로깅·훅·print를 직접 넣지 않는다. 필요하면 PyTorch의 `register_forward_hook`, `register_module_forward_pre_hook` 등 외부 접근 API를 써서 스크립트 쪽에서 주입한다.

## Rule 7. 해석은 관찰과 분리한다

`report.md`는 두 섹션을 명확히 나눈다:

- **Observations** — 측정된 사실 (shape, dtype, 값의 분포 등)
- **Interpretation** — 그 사실이 의미하는 바, 내가 어떤 고민을 했는지, 어떤 설계 결정으로 이어졌는지

관찰 사실과 해석을 섞어 쓰면 나중에 가설이 틀렸다는 걸 알아도 어느 부분을 고쳐야 할지 불분명해진다.

## Pre-observation gate

관찰 스크립트를 실행하기 전:

- `plan.md`가 작성되었는가
- 입력 파일이 존재하고 hash가 계산되었는가
- 관찰 포인트가 Rule 1의 계층 중 어디인지 선언되었는가
- raw 덤프 경로가 `.gitignore` 규칙에 부합하는가

## Post-observation gate

observation 폴더가 완료되었다고 표시하기 전:

- `report.md`, `analysis.ipynb`, `summary.json`이 모두 존재하는가
- 시각화가 노트북에 inline으로 남아 있는가 (outputs 제거 금지)
- `summary.json`의 `commit`이 관찰 시점의 git HEAD인가
- Interpretation 섹션에 "무엇을 알게 되었고 무엇을 다음에 볼지"가 쓰여 있는가
- `research/observations/INDEX.md`에 한 줄 등록되었는가
- **`docs/progress/WORKLOG.md`에 `[RESEARCH]` entry가 추가되고 선행 entry와 `Triggered by` 링크가 걸려 있는가** (Rule 8)

## Rule 8. observation은 WORKLOG에 [RESEARCH] entry로도 등록한다

observation 폴더를 완성하는 것만으로는 **왜 그 실험을 했는지** 의 맥락이 시간축 위에서 끊긴다. 정식 observation 완료 시 반드시 `docs/progress/WORKLOG.md`에 `[RESEARCH]` 태그 entry 하나를 추가해야 한다. entry는 아래를 포함한다:

- **What** — 한두 줄로 무엇을 관찰했는가
- **Why** — 이 실험을 촉발한 개발·질문·선행 결정
- **Found** — 핵심 발견 3-5 bullet
- **Details** — `report.md` 링크 + Related LEARNING 링크 (있으면)
- **Triggered by** — 이 실험을 유발한 선행 WORKLOG entry (DEV, RESEARCH, DECISION)
- **Triggers** — 이 실험이 만들 후속 작업 (생기면 이 필드 갱신)

이 규칙이 지켜지면 포트폴리오·회고 시 "어떤 개발 → 어떤 실험 → 어떤 결정 → 다음 개발" 흐름을 한 파일에서 읽어낼 수 있다. 실험만 따로 기록되면 자료로서 가치가 급격히 떨어진다.

또한 observation 단계는 **LEARNING 후보가 발견되는 대표 분기점**이다 (특히 결과가 가설과 어긋날 때, bit-exact 비교로 divergence가 드러날 때, 새로운 제약/호환성 이슈가 발견될 때). Interpretation 섹션과 WORKLOG [RESEARCH] entry를 쓰면서 반드시 [learn-from-friction.md](learn-from-friction.md) Rule 2의 4문항 게이트로 자기 점검하고, 통과하는 항목은 [docs/progress/LEARNINGS.md](../progress/LEARNINGS.md) 에 draft로 append한다.

## 인덱싱

`research/observations/INDEX.md`가 모든 관찰 항목을 최신순으로 모은다. 새 observation을 끝낼 때마다 한 줄을 추가한다:

```
- [2026-04-20 florence-2 initial pipeline trace](florence-2/2026-04-20-initial-pipeline-trace/report.md) — Florence-2 processor → generate → post-process까지 텐서 단위 관찰
```

INDEX.md는 빠른 탐색용이다. **맥락이 필요한 경우 WORKLOG를 먼저 본다.**

## Why this document exists

이 프로젝트는 모델의 데이터 흐름 관찰 자체가 연구 기여이다. 한 번에 여러 명의 에이전트·사람이 같은 관찰을 다른 형식으로 남기면 자료로서 가치가 빠르게 바랜다. 이 프로토콜은 관찰 기록을 **시간이 지나도 참조 가능한 포트폴리오 자산**으로 유지하기 위한 최소 규칙이다.
