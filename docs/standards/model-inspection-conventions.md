# Model Inspection Conventions

## Purpose

`research-observation-protocol.md`가 "어떻게 기록할지"를 정한다면, 이 문서는 **"무엇을 어떤 이름으로 관찰할지"**를 정한다. Florence-2와 SAM3의 아키텍처 계층에 공통된 네이밍·선정 기준을 부여해 여러 관찰 세션의 결과가 서로 비교 가능하게 한다.

## Observation point 네이밍 규칙

관찰 포인트 이름은 `<model>/<stage>/<component>[.subpath]` 형식으로 적는다.

예:
- `florence-2/processor/inputs.pixel_values`
- `florence-2/encoder/vision_tower.layer[4].self_attn.attn_weights`
- `florence-2/decoder/generate.output_ids`
- `florence-2/post_process/bboxes`
- `sam3/processor/state.image_embed`
- `sam3/mask_head/masks_logits`
- `sam3/output/mask_best.uint8`

규칙:
- 소문자, 하이픈·점 사용. 공백 금지.
- `<stage>`는 아래 모델별 스테이지 중 하나에서 선택.
- `<component>`는 코드상 실제 attribute 경로에 가깝게. 임의 별칭은 금지.
- 리스트 인덱스는 `[i]`.

## Florence-2 스테이지 구조

Florence-2는 텍스트-조건 VLM이다. 관찰 스테이지를 아래 5단계로 정의한다.

| Stage | 포함 대상 | 관찰 가치 |
|---|---|---|
| `processor` | `AutoProcessor` 입력 토큰화·이미지 전처리 | 입력이 모델 입장에서 어떻게 보이는지 |
| `encoder` | vision tower, text embedding, encoder blocks | 이미지·텍스트 표현의 결합 방식 |
| `decoder` | language model decoder, generate loop | 생성 중 attention과 token 흐름 |
| `post_process` | `processor.post_process_generation` | raw 토큰 → 구조화된 bbox/label |
| `output` | `detector.detect()` 최종 반환 | API 경계로 드러나는 계약 |

**관찰 기본 권장 포인트** (최소 세트):
- `processor/inputs.input_ids` — shape, dtype, 토큰 수
- `processor/inputs.pixel_values` — shape, dtype, stats, 이미지 사이즈 대응
- `decoder/generate.output_ids` — 생성된 토큰 길이, 분포
- `decoder/generate.cross_attention` — 이미지·텍스트 정합 시각화 (forward hook 필요)
- `post_process/result` — 원본 dict 구조
- `output/boxes`, `output/labels` — 최종 계약 필드

## SAM3 스테이지 구조

SAM3는 프롬프트 기반 세그멘테이션 모델이다. 관찰 스테이지를 아래 5단계로 정의한다.

| Stage | 포함 대상 | 관찰 가치 |
|---|---|---|
| `processor` | `Sam3Processor.set_image`, 입력 정규화 | 박스·이미지가 모델 내부 좌표로 어떻게 변환되는지 |
| `image_encoder` | 이미지 임베딩 생성 | 이미지 표현의 shape, dtype, 압축 비율 |
| `prompt_encoder` | 박스/포인트/라벨 프롬프트 임베딩 | 프롬프트가 임베딩으로 변환되는 과정 |
| `mask_head` | mask decoder, logits, score 후보 | 멀티 마스크 생성과 score 선택 기준 |
| `output` | `segment()` 최종 반환 | base64 PNG 인코딩 이전/이후 비교 |

**관찰 기본 권장 포인트** (최소 세트):
- `processor/state.image_embed` — shape (feature map 크기), dtype
- `processor/state.box_norm` — 정규화 좌표 변환 전후 비교
- `mask_head/masks_logits` — 후보 마스크 전체의 shape
- `mask_head/scores` — 각 후보의 score 분포
- `output/mask_best.uint8` — 최종 선택 마스크와 score

## 관찰 대상 선정 기준 (Decision table)

모든 텐서를 기록하지 않는다. 다음 질문을 차례로 물어 선정한다:

| 질문 | 예 | 기본 우선순위 |
|---|---|---|
| 모델 간 경계에서 오가는가? (Florence → SAM 전달 등) | boxes, image_size | **필수** |
| API 계약에 드러나는가? | output/* | **필수** |
| shape/dtype이 다른 모델 버전에서 바뀔 가능성이 높은가? | attention_weights, embedding | **필수** |
| 연구 가설의 직접 근거가 되는가? | 특정 attention head, 특정 레이어 출력 | **필수** |
| 디버깅에만 유용한가? (한 번 보고 버릴 것) | 내부 연산 중간값 | 선택 |
| 단순 scalar 상수이거나 config? | config dict | 기록 불필요 |

"필수"로 분류된 포인트는 `plan.md`에 반드시 올린다. "선택"은 노트북에서 탐색하다 가치를 확인한 뒤에 observation으로 승격한다.

## 텐서 기록 표준 필드

observation 리포트·`summary.json`에 각 포인트는 최소 다음을 포함한다:

- `name` — 네이밍 규칙을 따른 전체 경로
- `type` — `torch.Tensor`, `np.ndarray`, `list`, `dict` 등
- `shape` — 튜플
- `dtype` — `float32`, `int64`, 등
- `device` — `cpu`, `cuda:0` 등 (해당되는 경우)
- `stats` — `{min, max, mean, std}` (수치형 텐서)
- `sample` — 앞/뒤 몇 값 (너무 크면 요약)

이미지·마스크처럼 공간 구조를 갖는 텐서는 노트북에서 **반드시 시각화**하고 `assets/` 에 PNG로 저장한다.

## Forward hook 패턴

텐서 단위 관찰에는 PyTorch forward hook을 쓴다. 관찰 스크립트는 다음 패턴을 따른다:

```python
captured = {}

def make_hook(name):
    def hook(module, inputs, output):
        captured[name] = output.detach().cpu() if hasattr(output, "detach") else output
    return hook

handle = model.vision_tower.layers[4].self_attn.register_forward_hook(
    make_hook("florence-2/encoder/vision_tower.layer[4].self_attn")
)
# ... run inference ...
handle.remove()
```

훅 등록은 `research/tooling/hooks.py` 같은 공용 모듈에 helper를 두고 재사용한다. 훅을 달면 반드시 `remove()`로 해제한다.

## 이 모델들을 선택한 이유

- **Florence-2**: 단일 모델이 grounding·captioning·detection을 통합 처리하는 구조가 데이터 흐름 관찰에 풍부한 대상이다.
- **SAM3**: 프롬프트 기반 세그멘테이션이 Florence-2의 grounding 출력을 그대로 소비하는 경계가 흥미롭다. 두 모델의 경계에서 일어나는 좌표계 변환, 임베딩 차이는 연구의 중심 질문 중 하나다.

이 두 모델이 바뀌거나 추가되면 이 문서의 스테이지 표를 먼저 갱신한다.
