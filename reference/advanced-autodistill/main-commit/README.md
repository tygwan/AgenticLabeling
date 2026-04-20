# PROJECT-AGI - AI 기반 이미지 처리 및 Few-Shot Learning 플랫폼

이 프로젝트는 이미지 처리, 객체 탐지, 세그멘테이션 및 Few-Shot Learning을 위한 AI 기반 플랫폼입니다.

## 주요 기능

1. **이미지 처리 파이프라인**
   - Autodistill, Florence-2, SAM2를 활용한 객체 탐지 및 세그멘테이션
   - 마스크 및 바운딩 박스 좌표 기반 이미지 전처리
   - 데이터셋 생성 및 관리

2. **Few-Shot Learning 플랫폼**
   - 코사인 유사도 기반 Few-Shot 분류기
   - 다양한 N-shot 및 threshold 설정으로 실험 자동화
   - 결과 평가 및 시각화

3. **Ground Truth 관리 및 평가**
   - Ground Truth 생성 및 관리
   - 다양한 평가 지표 계산 (Accuracy, Balanced Accuracy, F1, MCC 등)
   - Binary/Multi-class Confusion Matrix 생성

## 설치 방법

### 요구 사항

- Python 3.10 이상
- CUDA 지원 GPU (선택 사항이지만 권장)
- 충분한 디스크 공간 (모델 파일 및 데이터용)

### 설치 단계

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/project-agi.git
cd project-agi
```

2. 설치 스크립트 실행:
```bash
./install_project.sh
```

이 스크립트는 다음을 수행합니다:
- 가상 환경 생성
- 필요한 패키지 설치
- 디렉토리 구조 생성
- segment-anything-2 저장소 클론 및 설치

### 수동 설치

자동 설치 스크립트를 사용하지 않는 경우, 다음 단계를 따르세요:

1. 가상 환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. segment-anything-2 설치:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e ".[demo]"
cd ..
```

## 프로젝트 구조

```
project-agi/
├── data/               # 데이터 및 분석 결과
├── docs/               # 문서 파일
├── logs/               # 로그 파일
├── models/             # 모델 파일
├── scripts/            # 스크립트 (워크플로우 단계별 정리)
│   ├── 01_data_preparation/  # 데이터 준비 및 초기 처리
│   ├── 02_preprocessing/     # 데이터 전처리 및 구조화
│   ├── 03_classification/    # Few-Shot Learning 및 분류
│   ├── 04_ground_truth/      # Ground Truth 생성 및 관리
│   ├── 05_yolo_training/     # YOLO 학습 및 데이터셋 생성
│   ├── 06_utilities/         # 공통 유틸리티 및 도구
│   └── 99_deprecated_debug/  # 사용하지 않는 파일들
└── (기타 설정 파일 및 폴더)
```

## 사용 방법

### 이미지 처리 파이프라인

1. 데이터 디렉토리 구조 설정:
```bash
mkdir -p data/my_category/{1.images,2.support-set,3.box,4.mask,5.dataset,6.preprocessed,7.results,8.refine-dataset}
```

2. 원본 이미지를 `data/my_category/1.images/` 디렉토리에 복사

3. 메인 런처 실행:
```bash
python scripts/main_launcher.py --category my_category --plot --preprocess
```

### Few-Shot Learning 플랫폼

1. 웹 인터페이스 모드:
```bash
python scripts/03_classification/run_few_shot_platform.py --webapp
```

2. CLI 모드:
```bash
python scripts/03_classification/run_few_shot_platform.py --cli --category my_category --model resnet
```

3. 커스텀 옵션:
```bash
python scripts/03_classification/run_few_shot_platform.py --cli --category my_category --model dino --shots 1,5,10 --thresholds 0.7,0.8,0.9
```

## 명령줄 옵션

### 메인 런처 옵션

- `--category <n>`: 카테고리 이름 설정 (기본값: test_category)
- `--debug`: 디버그 모드 활성화
- `--verbose`: 상세 로깅 활성화
- `--plot`: 탐지 결과 시각화 활성화
- `--save-npy`: NPY 파일로 마스크 데이터 저장 (기본값: True)
- `--preprocess`: 좌표 데이터를 사용한 이미지 전처리 수행
- `--target-size WxH`: 전처리된 이미지의 대상 크기 설정 (기본값: 640,640)
- `--no-crop`: 전처리 중 크롭 비활성화
- `--no-mask`: 전처리 중 마스크 적용 비활성화
- `--prepare-classify`: 분류를 위한 디렉토리 구조 생성
- `--classification-methods`: 쉼표로 구분된 분류 방법 목록

### Few-Shot 플랫폼 옵션

- `--webapp`: 웹 애플리케이션 모드로 실행
- `--cli`: CLI 모드로 실행
- `--category`: 실험 대상 카테고리 이름
- `--model`: 특징 추출에 사용할 모델 (resnet, clip, dino)
- `--shots`: N-shots 값 (쉼표로 구분)
- `--thresholds`: 임계값 목록 (쉼표로 구분)

## 문제 해결

- **모델 다운로드 문제**: 모델이 자동으로 다운로드되지 않는 경우, 첫 실행 시 다운로드됩니다.
- **CUDA 문제**: GPU 가속을 사용하는 경우 NVIDIA 드라이버가 올바르게 설치되어 있는지 확인하세요.
- **이미지 로딩 오류**: 이미지 형식을 확인하세요 (지원: jpg, png, jpeg).
- **시각화 문제**: "시각화 결과가 유효하지 않습니다" 경고가 표시되면 이미지가 유효한지 확인하고 --plot 옵션을 사용하세요.
- **좌표 파일 없음**: 전처리 전에 탐지 단계를 실행하세요.
- **분류 오류**: 전처리가 완료되었고 메타데이터 파일이 존재하는지 확인하세요.

## 참고 사항

- 이 저장소에는 데이터 파일과 모델 파일이 포함되어 있지 않습니다.
- 필요한 모델 파일은 첫 실행 시 자동으로 다운로드됩니다.
- 대규모 데이터셋을 처리하는 경우 충분한 디스크 공간과 메모리가 필요합니다.

## 라이선스

[MIT License](LICENSE)
