---
type: engineering
primary_evidence: mvp_app/static/components/viewer.jsx, mvp_app/static/styles.css
---

# Image Viewer — Contain-Fit + Overlay Alignment

## Concept framework

이미지 리뷰·어노테이션 UI의 흔한 요구사항:

1. 원본 종횡비를 유지하면서 가용 영역에 **contain-fit**(잘림 없이 전부 보이기, 여백은 letterbox).
2. bbox·mask·label 같은 **오버레이가 이미지 실제 렌더 영역에 픽셀 단위로 정렬**.

이 둘을 동시에 만족하는 것이 까다롭다. 흔한 실패 패턴들:

### 실패 패턴 A — `aspect-ratio` + `max-content` (컨텐츠 사이즈 기반)

```css
.canvas { aspect-ratio: W/H; max-width: 100%; max-height: 100%;
          width: max-content; height: max-content; }
.img   { width: 100%; height: 100%; object-fit: fill; }
```

`max-content`는 "컨텐츠의 선호 너비"다. 그런데 `<img>`에 `width`/`height` **HTML 속성이 없으면** 로드 전 intrinsic size = 0. 결과적으로 canvas가 0으로 붕괴 → 화면에 아무것도 안 보임. img에 `width={w} height={h}` 속성을 붙이면 intrinsic size가 known이라 동작하지만, 여전히 canvas가 img 원본 해상도 기준 max-content라 작은 뷰포트에서 `max-*` 클램핑 시 종횡비 유지 실패 가능성.

### 실패 패턴 B — `width: 100%; height: 100%` + `object-fit: contain`

```css
.canvas { width: 100%; height: 100%; }
.img   { width: 100%; height: 100%; object-fit: contain; }
```

img는 contain-fit 되지만 canvas는 뷰어 영역 전체 크기. bbox 오버레이가 `position: absolute` + `left: ${x*100}%` 식으로 canvas 기준 %를 쓰면, **letterbox(검은 여백)까지 포함한 canvas 좌표계**에 배치되어 실제 이미지 위가 아닌 여백에 걸린다. 이미지가 정사각형이 아닌 한 bbox 위치가 틀어진다.

### 실패 패턴 C — `object-fit`에 의존한 CSS-only 정렬

CSS만으로 img의 **렌더 박스**(letterbox 제외한 실제 이미지 영역)를 다른 요소의 positioning 기준으로 쓸 방법이 현재 제한적이다. `@container` + `cqi` 단위나 SVG `<foreignObject>` 등 우회는 가능하지만 브라우저 호환성·복잡도 비용.

### 작동 패턴 — JS-measured overlay layer

CSS는 img contain-fit만 담당. 오버레이는 **별도 레이어**로 분리하고 JS가 img의 실제 렌더 rect를 측정해 그 레이어의 `left/top/width/height`에 적용. 레이어 안의 오버레이는 레이어 기준 %로 배치.

핵심 계산:

```js
// img는 object-fit: contain 상태. img.getBoundingClientRect()는 img ELEMENT의 box
// (= 컨테이너 크기)를 반환. 실제 렌더된 이미지 영역을 구하려면 naturalWidth/Height로 스케일 계산.
const { naturalWidth: nw, naturalHeight: nh } = img;
const { width: boxW, height: boxH } = img.getBoundingClientRect();
const scale = Math.min(boxW / nw, boxH / nh);
const renderedW = nw * scale;
const renderedH = nh * scale;
const offsetX = (boxW - renderedW) / 2;   // letterbox 여백
const offsetY = (boxH - renderedH) / 2;
```

`object-fit: contain`은 `min` 기반. `cover`라면 `max`. `object-position`이 center가 아니면 offset 계산 수정 필요.

### 재계산 트리거

`scale`은 **이미지 원본 크기**와 **컨테이너 크기** 둘 다에 의존. 재계산 시점:

1. img `onLoad` — naturalWidth/Height가 처음 결정되는 순간
2. 뷰포트·컨테이너 resize — `window.addEventListener('resize', measure)` 또는 `ResizeObserver` (더 정확)
3. source 변경 — 새 이미지 naturalWidth/Height 반영
4. container query·사이드바 토글 같은 레이아웃 변경

`ResizeObserver`가 resize event보다 견고하다 (layout 변경 반응, 리사이징 성능 최적화).

### 좌표 공간 정리

이미지 오버레이 시스템은 세 개의 좌표 공간을 구분해야 한다:

| 공간 | 단위 | 사용처 |
|---|---|---|
| **원본 이미지 픽셀** | `(px_x, px_y)` | DB·API 저장, export 형식(YOLO는 정규화, COCO는 px) |
| **정규화** | `(nx, ny) ∈ [0, 1]` | API 응답·프론트 내부 전달, 해상도 무관 |
| **렌더 스크린 픽셀** | `(screen_x, screen_y)` | 오버레이 포지셔닝, 클릭 이벤트 좌표 변환 |

오버레이 레이어가 렌더 이미지 rect와 같으면 **정규화 좌표 → 렌더 픽셀** 변환이 `nx * layerWidth`로 단순해진다. 이 단순함 때문에 JS-measured 레이어가 실용적 해법이다.

## Evidence in this project

### 1. 실패 패턴 반복 후 JS 측정 채택

초기 `max-content + aspect-ratio` 시도에서 img가 너무 작게 안 보이거나 반대로 전체 크기로 overflow하여 잘리는 문제 발생. 사용자 피드백 "이미지가 화면에 다 안들어오고 크게보여서 일부 잘리는 모습" 이 직접적 트리거.

- **변경 전**: [`mvp_app/static/styles.css`](../../../mvp_app/static/styles.css) `.viewer-canvas { aspect-ratio: W/H; max-width:100%; max-height:100%; width: max-content; height: max-content; }`
- **변경 후**: canvas는 `width:100%; height:100%`, img는 `object-fit: contain`, 별도 `.viewer-overlay-layer`가 JS로 측정한 rect를 차지.

### 2. 측정 로직

`mvp_app/static/components/viewer.jsx`:

```jsx
const measure = React.useCallback(() => {
  const img = imgRef.current; if (!img) return;
  const parent = img.parentElement; if (!parent) return;
  const imgR = img.getBoundingClientRect();
  const parentR = parent.getBoundingClientRect();
  const nat = { w: img.naturalWidth || source.width,
                h: img.naturalHeight || source.height };
  if (!nat.w || !nat.h) return;
  const boxW = imgR.width, boxH = imgR.height;
  const scale = Math.min(boxW / nat.w, boxH / nat.h);
  const renderedW = nat.w * scale;
  const renderedH = nat.h * scale;
  const left = (imgR.left - parentR.left) + (boxW - renderedW) / 2;
  const top  = (imgR.top  - parentR.top)  + (boxH - renderedH) / 2;
  setRect({ left, top, width: renderedW, height: renderedH });
}, [source.width, source.height]);

React.useEffect(() => {
  measure();
  const onResize = () => measure();
  window.addEventListener('resize', onResize);
  return () => window.removeEventListener('resize', onResize);
}, [measure, source.url]);
```

오버레이 div·SVG는 `.viewer-overlay-layer` (left/top/width/height 설정됨) 안에서 `%` 위치로 배치되어 결과적으로 이미지 위에 정확히 올라간다.

### 3. Fallback — 측정 전에 오버레이 숨김

`{rect && imgOk && ( ... overlay layer ... )}` 패턴으로 rect 계산되기 전에는 오버레이 렌더 안 함. 초기 깜빡임·오정렬 방지.

## Related rules / decisions

- [LEARNING: contain-fit 이미지 + 오버레이 정렬은 JS-measured 레이어로](../../progress/LEARNINGS.md)
- [WORKLOG 2026-04-21 viewer 픽셀 정렬 수정](../../progress/WORKLOG.md)

## Open follow-ups

- `window.resize` → `ResizeObserver`로 교체 — `.viewer-canvas` 자체 리사이즈만 관찰하므로 viewport 변경 외 사이드바 토글·density 변경 등 모든 레이아웃 변경에 자동 반응.
- mask가 현재 mock `DeterministicBlob`으로 그려짐. 실제 `/api/masks/{object_id}` PNG를 canvas(2D context)로 합성하는 단계로 업그레이드 시, 마스크 이미지도 같은 overlay-layer 좌표계 쓰면 된다 (OPTIMIZATION-NOTES 항목 참조).
- 줌·팬 기능 추가 시 rect 계산 + CSS transform 결합이 필요. overlay-layer의 위치는 CSS transform 밖에서 결정하고 transform은 layer 내부 scale로 처리하는 패턴이 깔끔.
