# GPU Setup (CUDA / triton)

`requirements.txt`는 크로스 플랫폼(CPU/Linux/Windows)에서 공통으로 쓸 수 있도록 작성되어 있다. CUDA 빌드 torch와 triton은 플랫폼 종속이라 여기에 별도로 정리한다.

## GPU 환경 가정

- OS: Windows (WSL / Linux도 가능)
- GPU: NVIDIA (예: RTX 4090, Ada Lovelace, compute 8.9)
- CUDA 드라이버: 12.x (필요한 CUDA runtime은 torch wheel이 자체 번들)

`nvidia-smi` 로 드라이버·CUDA 버전을 먼저 확인한다. torch wheel의 `cu12x` 접미사는 빌드된 CUDA runtime이고, 드라이버가 그보다 같거나 높으면 작동한다.

## 1. torch CUDA 빌드 설치

CPU 전용으로 설치된 상태면 먼저 제거 후 CUDA 빌드로 교체한다.

Windows PowerShell / bash:

```bash
.venv312/Scripts/python.exe -m pip uninstall -y torch torchvision
.venv312/Scripts/python.exe -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
```

Linux:

```bash
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
```

설치 후 검증:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`torch.cuda.is_available()` 가 `True` 여야 하고 GPU 이름이 찍혀야 한다.

## 2. triton 설치

SAM3의 `sam3.perflib.*`, `sam3.model.edt`는 `triton`을 직접 import 한다.

Linux + CUDA:

```bash
python -m pip install triton
```

Windows + CUDA:

```bash
.venv312/Scripts/python.exe -m pip install triton-windows
```

CPU only:

> triton은 CPU 환경에서 설치할 수 없다. SAM3의 실제 경로 대신 `segmenter._processor is False` 폴백이 타서 `backend_name=='box-fallback'` 로 동작한다 (바운딩 박스 영역만 채워진 마스크). 이 상태에서는 SAM3 mask_head 관찰이 불가능하다.

## 3. SAM3 체크포인트

기본값은 HuggingFace에서 자동 다운로드(`load_from_HF=True`). 로컬 체크포인트를 쓰려면 환경변수:

```bash
export SAM3_CHECKPOINT=/path/to/sam3.pt
```

## 4. 검증 스크립트

GPU 경로가 제대로 잡혔는지 빠르게 확인:

```bash
FLORENCE_MODEL_ID=microsoft/Florence-2-base .venv312/Scripts/python.exe scripts/inspect_pipeline.py \
    --image data/images/test_street.jpg \
    --classes "car,person,road" \
    --slug gpu-smoke \
    --model-slug pipeline
```

실행 후 `research/observations/pipeline/<date>-gpu-smoke/summary.json`의:

- `key_packages.torch` → `x.y.z+cu126` (CPU가 아니어야 함)
- `observation_points[]` 에 `sam3/processor/backend_name` 의 `value`가 `sam3` (box-fallback 아님)
- `sam3/mask_head/masks_logits[...]`, `sam3/mask_head/scores[...]` 포인트가 존재

## 알려진 경로

- `vendor/sam3/sam3/perflib/fused.py` 는 이 프로젝트에서 Turing (RTX 2070 Super) 호환성을 위해 패치된 버전이다. RTX 4090 (Ada) 에서도 동일하게 동작한다.
- SAM3 첫 로드 시 HuggingFace에서 체크포인트 다운로드 (~수백 MB).
