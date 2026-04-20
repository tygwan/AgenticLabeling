---
type: hybrid
primary_evidence: mvp_app/detector.py, mvp_app/segmenter.py, research/observations/florence-2/
---

# Florence-2 + SAM3 Pipeline — Grounded Detection + Promptable Segmentation

## Concept framework

### 1. Grounded detection / phrase grounding

Object detection 은 오래된 과제다. 패러다임이 세 시기로 나뉜다:

**Closed-set detection (고전)** — 모델을 **고정된 클래스 집합**(예: COCO 80, VOC 20) 위에서 학습. 학습 시 본 클래스만 검출 가능. 새 클래스 추가하려면 재학습. 대표: YOLO 계열, Faster R-CNN, DETR.

**Open-vocabulary detection** — 자연어 설명으로 **어떤 클래스든 즉석에서 지정**. 재학습 없이 "traffic cone", "construction worker holding a clipboard" 같은 새 클래스 사용 가능. 대표: OWL-ViT, GLIP, Grounding DINO, Florence-2.

**Phrase grounding** — open-vocabulary의 구체적 태스크 중 하나. 주어진 **caption**에서 **noun phrase**들을 추출해 각각을 이미지 영역에 매핑(ground). "A dog next to a red car" → `[dog]`에 bbox 하나, `[red car]`에 bbox 하나. 모델이 문장 구조를 이해해 다중 개체를 동시 localization.

**Florence-2의 task prompts**: 모델이 여러 task를 하나의 weight set으로 처리하기 위해 **special token**으로 task를 구분한다. 주요 prompts:

| Prompt | 동작 |
|---|---|
| `<OD>` | 이미지 전체에서 pretrained ontology의 모든 object 검출 |
| `<CAPTION>` | 이미지 전체 설명 생성 |
| `<CAPTION_TO_PHRASE_GROUNDING>` | 주어진 caption의 phrase를 이미지에 ground |
| `<REFERRING_EXPRESSION_SEGMENTATION>` | "the red car on the left" 같은 지시 표현의 영역 마스크 |
| `<REGION_TO_SEGMENTATION>` | 주어진 bbox 영역의 segmentation mask |

우리 프로젝트는 **`<CAPTION_TO_PHRASE_GROUNDING>`** 사용. 이유: open-vocabulary 유연성 + caption 형태가 다중 클래스 동시 처리에 자연스러움.

**Caption 프리픽스 관례**: open-vocabulary 모델은 훈련 분포상 "A photo of X" 같은 stock prefix에서 성능이 더 나오는 경향이 있다(CLIP prompt engineering 연구 기원). Florence-2도 비슷해 우리 detector가 `"A photo of " + ", and ".join(classes) + "."` 형태를 구성. 이 표현 자체가 prompt engineering의 결과물이고, 논문·실험에서 **prompt template**이 성능에 유의미 영향.

### 2. Promptable segmentation (SAM family)

SAM (Segment Anything Model)은 **foundation segmentation model** 이다. 핵심 철학:

- **Class-agnostic**: 모델은 "무엇인지" 모르고 "어디인지"만 안다. 입력된 prompt가 가리키는 물체의 경계를 찾는다.
- **Promptable**: 입력이 **프롬프트 + 이미지**. prompt는 아래 중 하나 또는 조합:

| Prompt type | 의미 |
|---|---|
| **point** | (x, y) 좌표 + label(foreground/background) |
| **box** | 대상 영역의 직사각형 |
| **mask** | 대략적 초기 마스크 (coarse → fine refinement) |
| **text** (SAM3부터) | 자연어 설명 |

- **Multi-mask 출력**: 하나의 point prompt는 여러 해석 가능(예: 사람의 얼굴 vs 상체 vs 전신). SAM은 **여러 후보 마스크** + 각각의 score를 출력하고, 클라이언트가 **argmax**하거나 user가 선택.

SAM 시리즈:

- **SAM 1** (2023): point, box, mask prompts. ViT-H backbone.
- **SAM 2** (2024): 이미지 + 비디오 통합, memory attention으로 frame-wise tracking.
- **SAM 3** (2024): text prompt 추가, 성능/효율 개선.

우리 프로젝트는 SAM3 사용하되 **box prompt만** 활용. Florence-2의 bbox를 geometric prompt로 넘겨 "이 영역의 물체 경계를 찾아줘" 지시.

### 3. Pipeline composition — detect-then-segment

**Grounded SAM 패턴**은 open-vocabulary detector와 promptable segmenter를 조합해 **"텍스트로 지정한 어떤 클래스든 픽셀 단위 마스크까지"** 얻는 워크플로우. 변형이 여럿 있지만 구조는 동일:

```
Text prompt (classes) ─┐
                       ├─► Detector (Florence-2 / Grounding DINO / OWL-ViT)
Image ─────────────────┤              │
                       │              ▼ bboxes + labels + scores
                       │
                       └─► Segmenter (SAM1/2/3) ─► per-object masks
                                     ▲
                                     └── box prompts
```

왜 두 모델로 나누나? **전문화의 경제**:

- Detector는 "어디에 무엇이 있나"에 최적화. Localization + classification이 목표.
- Segmenter는 "주어진 영역의 정확한 픽셀 경계"에 최적화. Class 이해 책임 없음.
- 두 모델은 **독립 진화** 가능. Detector를 OWL-ViT으로 바꿔도 segmenter는 그대로. Segmenter를 SAM2 → SAM3 로 업그레이드해도 detector 영향 없음.

**Trade-offs**:

- **이중 image encoding**: 이미지가 detector의 vision encoder와 segmenter의 image encoder를 모두 통과. Feature 공유 불가능(서로 다른 아키텍처·pretraining). → latency·VRAM 중복.
- **좌표계 변환**: detector 출력(pixel xyxy) → segmenter 입력(normalized cxcywh 등) 변환 필수. 에러 포인트.
- **개체 수에 비례한 segment 호출**: detector가 N개 bbox 반환하면 segmenter는 N번 호출. batched inference로 줄일 수 있지만 SAM 표준 API는 대부분 single-prompt per call.
- **장점**: 개별 모델의 state-of-the-art를 자유롭게 조합. open-vocabulary + pixel-perfect가 한 모델로는 drained가 크다.

### 4. 클래스 정보의 흐름

중요한 관점 — **class name은 파이프라인을 일직선으로 통과하지 않는다**:

```
classes=["car", "person"] ──► Florence-2 prompt ──► Florence-2 output
                                                          │
                                                   ┌──────┴──────┐
                                                   ▼             ▼
                                             boxes[], labels[], scores[]
                                                   │             │
                                                   ▼             ▼
                                     ┌─────────────┘             │
                                     ▼                           │
                              SAM3 box prompt only               │
                                     │                           │
                                     ▼                           │
                                  masks[]                        │
                                     │                           │
                                     └────────────┬──────────────┘
                                                  ▼
                                         Registry.register_objects_batch
                                         (label + mask 다시 묶음)
```

- Class name은 **detector 입력**이자 **registry 저장값**이지만 **segmenter 입력은 아니다**.
- 결과적으로 segmenter 계층은 **class ontology에 독립**. 다른 labeling system(COCO, VOC, 커스텀)과도 호환.
- 이 분리가 "**모델은 교체 가능, ontology는 독립**"이라는 설계 원칙을 보장.

## Evidence in this project

### 1. Florence-2 prompt 구성 — phrase grounding

`mvp_app/detector.py:107-114`:

```python
prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
text_input = "A photo of " + ", and ".join(classes) + "."
inputs = self._processor(
    text=prompt + text_input,
    images=image,
    return_tensors="pt",
)
```

실제 입력 예시(classes=["car", "person", "road"]): `<CAPTION_TO_PHRASE_GROUNDING>A photo of car, and person, and road.`

이것이 "class word 하나 입력이 아님, caption 구조" 의 증거. prompt template은 prompt engineering 영역이라 다른 stock 형태(예: "Objects: car. person. road.") 비교 실험 여지 있음(Open follow-up).

### 2. Florence-2 출력 포맷

`mvp_app/detector.py:119-128`:

```python
generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
result = self._processor.post_process_generation(
    generated_text,
    task=prompt,
    image_size=(image.width, image.height),
)
parsed = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
raw_boxes = parsed.get("bboxes", [])
raw_labels = parsed.get("labels", [])
```

- raw 생성 토큰에 `<loc_N>` (N ∈ [0, 999]) 형태의 **location token**이 포함. post-processor가 이들을 실제 pixel xyxy로 변환.
- bbox와 label이 동수로 정렬되어 반환. 클래스 매칭은 `_match_requested_class()` 에서 수행(모델이 출력한 raw label이 requested class와 정확히 일치 안 할 때 fuzzy match: plural → singular, 토큰 교집합 등).
- 근거 observation: [FP32 vs BF16 비교의 `output_ids` 4 token drift](../../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md) — `<loc_N>` 토큰들이 정확도 차이로 다른 bucket을 선택한 경우.

### 3. SAM3 box-only geometric prompt

`mvp_app/segmenter.py:122-135`:

```python
state = self._processor.set_image(image)
for box in boxes:
    self._processor.reset_all_prompts(state)
    box_norm = self._xyxy_to_cxcywh_norm(box, image.width, image.height)
    state = self._processor.add_geometric_prompt(
        box=box_norm, label=True, state=state,
    )
    if "masks" in state and len(state["masks"]) > 0:
        scores = state["scores"]
        best_idx = int(scores.argmax())
        mask_tensor = state["masks"][best_idx][0]
```

- `set_image()` 가 이미지를 한 번 SAM3 image encoder에 통과시켜 embedding 캐시.
- 각 box에 대해 `add_geometric_prompt(box=box_norm, label=True)` 호출 — **class name 없음**. `label=True`는 foreground 지시자(SAM의 point prompt에서 foreground/background 구분용 semantics가 box에도 전이).
- `box` 좌표는 `_xyxy_to_cxcywh_norm` 으로 변환: Florence-2의 `[x1,y1,x2,y2]` (pixel) → SAM3의 `[cx,cy,w,h]` (normalized [0,1]).
- **Multi-mask 선택**: `state["masks"]` 가 여러 후보를 반환할 수 있고, `scores.argmax()` 로 최고 점수 선택. observation에서 box#3(building)만 2개 후보 반환 사례 확인 — 모델이 boundary 애매하면 multi-hypothesis 출력([첫 파이프라인 trace observation](../../../research/observations/florence-2/2026-04-20-florence2-base-bf16/summary.json) 참고).

### 4. 이중 image encoding — 두 모델 모두 이미지 독립 처리

관련 observation에서 확인된 두 경로:

- **Florence-2**: `inputs.pixel_values shape = [1, 3, 768, 768]` (ImageNet normalization 적용, float32 또는 bf16). vision tower가 이미지 feature 생성.
- **SAM3**: `set_image(PIL.Image)` 호출 시 내부 image encoder가 별도 feature map 생성 (SAM image encoder는 일반적으로 1024×1024 input).

두 feature space는 **공유 불가** (pretraining·아키텍처 다름). 이미지 크기 2816×1584의 경우 양쪽 downsample·normalize 비용 각각 부담. OPTIMIZATION-NOTES에 "이중 image encoding 제거 또는 캐싱" 후보 등록 대상.

### 5. 클래스 ontology 독립성

`mvp_app/main.py:197-207` (auto-label):

```python
objects_data.append(
    {
        "category": labels[idx] if idx < len(labels) else "object",
        "bbox": xywh,
        "confidence": scores[idx] if idx < len(scores) else None,
        "detection_model": "florence2",
        "mask_base64": masks[idx]["mask"] if idx < len(masks) else None,
    }
)
```

- `category` 는 Florence-2 label 그대로(또는 fuzzy matched requested class).
- `mask_base64` 는 SAM3 출력.
- **두 정보가 object_id 아래 합쳐지는 시점은 registry write 단계.** SAM3는 `category`를 보지 못한 채 mask만 반환.

이 구조 덕에 **detector·segmenter 둘 중 하나를 교체**해도 ontology 계약(`registry`의 `category_id` 테이블) 은 불변.

## Related rules / decisions

- [LEARNING: bit-exact 샘플 비교 (decode divergence 발견 방법)](../../progress/LEARNINGS.md) — Florence-2의 `output_ids` 4 token drift는 phrase grounding token stream 안에서 발생한 divergence
- [LEARNING: dtype 경계에서 fail-fast cast](../../progress/LEARNINGS.md) — detector.py의 `model_dtype` cast 로직이 caption input을 BF16 모델로 넘길 때 일어나는 mismatch 해소 경로
- [Concept: bf16-precision](bf16-precision.md) — 이 파이프라인의 decoder 정밀도 논의는 해당 concept의 Evidence 섹션에 귀속
- [Observation: 2026-04-20 initial pipeline trace](../../../research/observations/florence-2/) — 파이프라인 각 단계 텐서 단위 관찰. `<loc_N>` token, box-prompt 좌표 변환, masks_logits shape `[2, 1, H, W]` 사례 등 본 concept의 Evidence 4·5 바탕.
- [WORKLOG 2026-04-21 baseline cycle 실사용](../../progress/WORKLOG.md) — warm pipeline 791 ms 측정

## Open follow-ups

- **Prompt template 비교 실험**: "A photo of X, and Y." vs "Objects: X. Y." vs "<OD>" task 자체. 클래스별 recall·precision 변화 측정 → prompt engineering 효과 정량화.
- **Grounded SAM 변형 조사**: Florence-2 → SAM3 대신 Grounding DINO → SAM2, OWL-ViT → SAM3 비교. 각 조합의 VRAM·latency·정확도 trade-off.
- **이중 image encoding 최적화**: Florence-2 vision tower feature를 SAM3 image encoder에 전달 불가능하더라도, **thumbnail 공유**(예: 둘 다 동일 downsampled 이미지 쓰면 decode 비용 감소) 혹은 **한 모델의 feature를 다른 모델 adapter로 투영** 연구.
- **SAM3 text prompt 활용**: 현재 box prompt만 쓰는데, SAM3는 text prompt도 받을 수 있다. Florence-2 class name을 SAM3에 **추가 힌트**로 전달하면 boundary ambiguity(box#3 multi-mask case) 개선 가능성.
- **Multi-mask 선택 정책**: 지금은 `scores.argmax()` 단순. 후보 2개 이상일 때 IoU·크기·class 가능성 등으로 재정렬할 수 있는가?
