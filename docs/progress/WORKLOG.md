# AgenticLabeling WORKLOG

개발·실험·결정을 시간순으로 기록한 단일 피드. 각 entry는 태그로 구분한다.

## 읽는 법

- 최신 entry가 **위에** 있다.
- 태그:
  - `[DEV]` — 코드·환경·인프라·문서 등 개발 작업
  - `[RESEARCH]` — 모델 파이프라인 관찰·실험 (상세는 `research/observations/`)
  - `[DECISION]` — 이 프로젝트의 방향·기본값을 바꾸는 결정
- 인과 관계:
  - `**Triggered by**`: 이 entry를 유발한 이전 entry
  - `**Triggers**`: 이 entry가 유발한 후속 entry (후속이 완료되면 업데이트)

## 쓰는 법

각 entry는 아래 네 필드 이상을 포함한다:

- **What**: 무엇을 했는지 2-3줄
- **Why**: 동기. 무엇이 궁금했거나 무엇이 막혔었는지
- **Result / Found / Decision**: 핵심 산출물·관찰·결정
- **Details**: 자세한 자료·코드·리포트 링크

상세 리포트는 `docs/` 또는 `research/observations/` 아래에 두고, WORKLOG에는 링크만 남긴다.

상세 프로토콜·규칙은:

- [docs/standards/research-observation-protocol.md](../standards/research-observation-protocol.md)
- [docs/standards/model-inspection-conventions.md](../standards/model-inspection-conventions.md)

---

## 2026-04-20

### [DECISION] `FLORENCE_DTYPE` 환경변수 도입, 배포 기본값 BF16 채택

**What**: `mvp_app/config.py`에 `florence_dtype` 설정 추가, `mvp_app/detector.py`에서 dtype resolver(`_resolve_dtype`) 통해 float32 / bfloat16 / float16 지원. 모델 로드 시 이 dtype으로 캐스트, 입력 `pixel_values`도 model dtype에 맞춰 캐스트. 테스트 202 pass 유지.
**Why**: 아래 RESEARCH에서 BF16이 VRAM을 정확히 절반으로 줄이고 출력 품질을 거의 보존함을 실측으로 확인. GPU 운용 시 VRAM 확보가 더 큰 배치·동시 SAM3 로드·더 큰 이미지를 위해 중요.
**Decision**:
- 코드 기본값은 **float32** 유지 (CPU CI, fake-models 테스트 호환성)
- 배포·관찰 환경에서는 `FLORENCE_DTYPE=bfloat16` 으로 override
- 속도 이득(1.03×만)은 **미해결** — 후속 실험에서 attention 구현 교체로 재측정
**Details**:
- 변경 파일: `mvp_app/config.py`, `mvp_app/detector.py`
- 근거 리포트: [research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md](../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md)

**Triggered by**: 아래 RESEARCH (FP32 vs BF16 비교)
**Triggers**: (후속) `attn_implementation='sdpa'` 비교 observation, Florence-2-large 동일 비교

---

### [RESEARCH] Florence-2 FP32 vs BF16 비교

**What**: 같은 이미지(`data/images/test_street.jpg`)·같은 클래스 프롬프트(`car,person,road,building,sky`)를 FP32 / BF16 두 dtype으로 나눠 실행. 각각 warmup 1회 + measured 3회, `torch.cuda.synchronize()` 브래킷. VRAM·wall-clock·출력 bbox·레이블 비교.
**Why**: GPU 전환 직후, 앞으로 Florence-2 추론을 어떤 compute dtype으로 굴릴지 결정 필요. BF16의 VRAM·속도 이득이 실제로 얼마나 되는지, 출력 품질을 해치지는 않는지 실측.
**Found**:
- **VRAM 절반** 확인 (모델 0.510×, peak 0.512×). 이론치 일치. ✅
- **속도 1.03×에 그침**. 예상한 1.5-2×와 큰 격차. 유력 원인: `attn_implementation="eager"` 설정이 Tensor Core 경로를 우회. ❌
- **출력 레이블 완전 동일** (6개), bbox drift 최대 19.71 px (0.7%), 정보 손실 실용적으로 없음. ✅
- **Bonus (Q2)**: `output_ids` bit-exact 비교 결과 4개 토큰이 실제로 달랐음. drift는 post_process만이 아니라 decode argmax boundary 근처에서 이미 다른 `<loc_N>` 토큰이 선택된 결과. 경계 모호한 box(sky)에서 drift 집중.
**Details**:
- 비교 report: [research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md](../../research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/report.md)
- 시각화 포함 노트북 + outputs: `research/observations/florence-2/2026-04-20-dtype-fp32-vs-bf16/analysis.ipynb`
- Source observations: `research/observations/florence-2/2026-04-20-florence2-base-fp32/`, `.../2026-04-20-florence2-base-bf16/`

**Triggered by**: 위 DEV (GPU 전환)
**Triggers**: 위 DECISION (`FLORENCE_DTYPE` 도입)

---

### [RESEARCH] 첫 정식 Florence-2 + SAM3 파이프라인 trace

**What**: GPU 환경이 준비되자마자 `scripts/inspect_pipeline.py`로 `test_street.jpg`를 Florence-2-base + SAM3 전체 파이프라인에 한 번 통과시키고 41개 observation point를 기록. raw `.pt` dump는 `outputs/raw/`에 저장(gitignored).
**Why**: 프로토콜 제정 직후 첫 실제 파이프라인 관찰 기준선을 확보. 이후 실험의 비교점이자 연구 기록의 0번 entry로 기능.
**Found**:
- Florence-2 processor: `pixel_values [1,3,768,768] float32`, `input_ids [1,27] int64`, ImageNet 정규화 분포 확인
- Florence-2 decoder: `output_ids [1,38]`, 생성 text에 `<loc_N>` 위치 토큰 포함
- SAM3 `masks_logits` shape이 **box마다 다름**: 대부분 `[1,1,2816,1584]` (후보 1개), **box#3 building만 `[2,1,2816,1584]`** (후보 2개). argmax 선택 로직이 실제로 의미 있는 유일한 케이스.
- SAM3 `masks_logits` dtype은 이름과 달리 **`bool`** (이미 이진화됨). 진짜 float logit은 SAM3 내부 더 안쪽 지점에서 hook 걸어야 관찰 가능.
**Details**:
- Source observation: 위 RESEARCH(FP32 vs BF16)가 이 trace 이후 같은 형식으로 생성한 두 run으로 대체됨.
- 관찰 스크립트: `scripts/inspect_pipeline.py`

**Triggered by**: 위 DEV (연구 관찰 인프라 완비)
**Triggers**: 아래 follow-up 후보 — SAM3 mask_head 더 깊이(진짜 logit float), building multi-mask 원인 가설 검증

---

### [DEV] GPU 환경 전환 — CUDA torch + triton-windows + SAM3 전체 의존성

**What**:
- torch CPU wheel 제거 → `torch 2.11.0+cu126 / torchvision 0.26.0+cu126` 설치 (RTX 4090, CUDA 12.6 드라이버)
- `triton-windows 3.6.0.post26` 설치 (SAM3 perflib가 `triton` 직접 import)
- SAM3가 요구하는 누락 의존성 발견·설치: `einops`, `timm`, `iopath`, `ftfy`, `pycocotools`
- transformers 5.x → **4.57.6 다운그레이드** (Florence-2 custom code가 4.x API 기준)
- `requirements.txt`에 누락 의존성·상한 반영, triton은 플랫폼 종속성 주석만 남기고 제외
- `docs/setup/gpu-setup.md` 신규 작성 — GPU 전환 절차·triton 플랫폼 차이·검증 스크립트
**Why**: MVP·research 모두 실제 모델 경로로 가야 가치가 생긴다. 처음 실행 시 CPU torch만 있고 triton·iopath 등 SAM3 트랜스이티브 의존성이 미설치여서 `backend_name='box-fallback'` 으로 떨어짐. 사용자가 RTX 4090 GPU 환경임을 명시하며 BF16/GPU 경로 전환 요청.
**Result**:
- 첫 실행 시 Florence-2 only 동작, SAM3 fallback. 트랜시트 의존성 해결 후 **SAM3 `backend_name == 'sam3'`** 동작 확인
- `torch.cuda.is_available() == True`, RTX 4090 인식
- 테스트 202 pass 유지
**Details**:
- 변경 파일: `requirements.txt`, `requirements-mvp.txt`, `docs/setup/gpu-setup.md`
- `mvp_app/detector.py`의 `dtype=torch.float32`는 원래 CPU 호환 목적, 이후 DECISION에서 env var로 교체

**Triggered by**: 위 DEV (연구 관찰 인프라) 첫 실행 시 SAM3 경로 실패
**Triggers**: 위 RESEARCH (첫 파이프라인 trace), 이어서 FP32 vs BF16 비교

---

### [DEV] 연구 관찰 인프라 도입 — 프로토콜·컨벤션·`research/` 구조·관찰 도구

**What**:
- **규칙 문서**: `docs/standards/research-observation-protocol.md` (관찰 단위·포인트 등록·이중 트랙 기록·재현성·사전/사후 gate) + `docs/standards/model-inspection-conventions.md` (Florence-2 / SAM3 스테이지 정의, 네이밍 규칙, 관찰 대상 decision table, forward hook 패턴)
- **디렉토리**: `research/{observations,experiments,tooling}/` + README + INDEX
- **도구**: `research/tooling/{stats.py,hooks.py,io.py}` — describe 헬퍼, forward hook registry, 관찰 폴더 레이아웃·덤프·commit hash·sha256 유틸
- **관찰 스크립트**: `scripts/inspect_pipeline.py` — 이미지 + 클래스 → Florence-2·SAM3 파이프라인 trace → `summary.json` + plan/report/notebook 템플릿 자동 생성. fake-models smoke test 통과
- **에이전트 진입점**: `CLAUDE.md`, `AGENTS.md`, `docs/agents/START-HERE.md`에 연구 작업 트리거 링크
- **`.gitignore`**: raw tensor 덤프(`*.npz`, `*.pt`, `*.safetensors`, `outputs/raw/**`) ignore. 리포트·노트북·요약·시각화는 커밋.
**Why**: AgenticLabeling은 사용자의 연구논문에서 파생된 프로젝트이고, 핵심 연구 관심사가 Florence-2·SAM3 파이프라인의 **데이터 타입·형식·흐름 관찰**이다. 일회성 스크립트만으로는 포트폴리오 자산·재현성·시간순 비교 가치가 휘발된다.
**Result**: 향후 모든 관찰은 이 프로토콜을 따라 기록됨. smoke test로 구조 검증 완료 (7 observation points + 템플릿 생성 확인).
**Details**:
- 신규 파일: `docs/standards/research-observation-protocol.md`, `docs/standards/model-inspection-conventions.md`, `research/README.md`, `research/tooling/*.py`, `scripts/inspect_pipeline.py`, `research/observations/INDEX.md`
- 변경: `CLAUDE.md`, `AGENTS.md`, `docs/agents/START-HERE.md`, `.gitignore`

**Triggered by**: 사용자가 "이 프로젝트는 연구논문에서 시작. 데이터 타입 관찰이 핵심" 라는 맥락 공유
**Triggers**: 위 DEV (GPU 전환), 이후 모든 RESEARCH entry

---

### [DEV] Windows Python 3.12 환경 구축 + 크로스플랫폼 인코딩 버그 수정

**What**:
- `py -3.12 -m venv .venv312` 생성, `requirements.txt` 전체 설치 (CPU torch로 시작)
- **버그 수정**: `services/object-registry/app/registry.py:130` `open(schema_path, "r")` 에 `encoding="utf-8"` 추가. Windows 기본 로케일 CP949로 한국어 포함 SQL 파일을 읽다 `UnicodeDecodeError` 발생 → 27개 레지스트리 테스트 실패의 근본 원인.
- 테스트용 이미지 파일 배치 (`data/images/test_street.jpg`)
- `tests/` 202 pass 확보
**Why**: 기존 `.venv/`는 WSL에서 만든 Linux 스텁이라 Windows에서 사용 불가. 첫 테스트 실행 시 의존성·인코딩·테스트 데이터 3종 이슈 모두 노출. 이후 모든 작업의 기반.
**Result**:
- 202 passed, 0 errors, 0 failed (CP949 버그 수정 + 테스트 이미지 배치로)
- 크로스플랫폼 결함 하나 영구 제거
**Details**:
- 변경 파일: `services/object-registry/app/registry.py`
- 신규 venv: `.venv312/` (gitignored)

**Triggered by**: 새 Windows 개발 환경 시작
**Triggers**: 위 DEV (연구 관찰 인프라)

---

## 기록 규칙 (요약)

1. 개발·실험·결정은 **시간순 단일 피드**로 이 파일에 쌓는다. 가장 최근 entry가 위.
2. Observation이 완료되면 **반드시 `[RESEARCH]` entry 하나를 이 파일에 추가**하고 `report.md` 로 링크한다 — `research-observation-protocol.md` Rule 8.
3. 제품 기본값·환경 설정·API 계약 등을 바꾸는 변경은 **별도 `[DECISION]` entry**로 남긴다. 근거 RESEARCH·DEV를 `Triggered by`로 묶는다.
4. entry 본문은 간결하게. 상세는 반드시 링크로.
