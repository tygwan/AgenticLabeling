# 프로젝트 파일 재정리 요약

## 완료된 작업

### 1. 스크립트 폴더 구조화
- 워크플로우 단계별로 스크립트 정리
- `organize_scripts.py`를 통해 6개의 주요 카테고리로 분류:
  - `01_data_preparation/`: 데이터 준비 및 초기 처리
  - `02_preprocessing/`: 데이터 전처리 및 구조화
  - `03_classification/`: Few-Shot Learning 및 분류
  - `04_ground_truth/`: Ground Truth 생성 및 관리
  - `05_yolo_training/`: YOLO 학습 및 데이터셋 생성
  - `06_utilities/`: 공통 유틸리티 및 도구
  - `99_deprecated_debug/`: 사용하지 않는 파일들

### 2. 문서 파일 정리
- 문서 파일을 `docs/` 폴더로 이동 및 카테고리화:
  - `docs/guides/`: 워크플로우 가이드 및 상세 프로세스 문서
  - `docs/readme/`: 다양한 README 파일
  - `docs/setup/`: 설치 및 설정 관련 문서
  - `docs/cleanup/`: 정리 및 제거 관련 문서

### 3. 로그 파일 정리
- 로그 파일을 `logs/app/` 폴더로 이동:
  - `fsl_platform.log`
  - `main_webapp.log`
  - `eventerrorlog.txt`
  - `mcp.log`

### 4. 분석 데이터 정리
- 메트릭 및 분석 파일을 `data/analytics/` 폴더로 이동:
  - `dino_metrics.csv`
  - `resnet_metrics.csv`

### 5. 유틸리티 스크립트 이동
- 프로젝트 관리 스크립트를 `scripts/06_utilities/` 폴더로 이동:
  - `organize_scripts.py`
  - `organize_project.py`
  - `execute_backup.py`
  - `cleanup_project.sh`
  - `track_modified_packages.sh`

## 폴더 구조

```
project-agi/
├── data/
│   ├── analytics/           # 분석 데이터 및 메트릭
│   └── test_category/       # 테스트 카테고리 데이터
├── docs/
│   ├── guides/              # 워크플로우 가이드
│   ├── readme/              # README 파일
│   ├── setup/               # 설치 가이드
│   └── cleanup/             # 정리 관련 문서
├── logs/
│   ├── app/                 # 애플리케이션 로그
│   └── model_comparison/    # 모델 비교 로그
├── models/                  # 모델 파일
├── scripts/
│   ├── 01_data_preparation/ # 데이터 준비 스크립트
│   ├── 02_preprocessing/    # 전처리 스크립트
│   ├── 03_classification/   # 분류 스크립트
│   ├── 04_ground_truth/     # Ground Truth 관련 스크립트
│   ├── 05_yolo_training/    # YOLO 학습 스크립트
│   ├── 06_utilities/        # 유틸리티 스크립트
│   └── 99_deprecated_debug/ # 더 이상 사용되지 않는 스크립트
└── (기타 설정 파일 및 폴더)
```

## 다음 단계

1. 스크립트 내 상대 경로 수정
   - 폴더 구조 변경으로 인한 import 경로 문제 해결 필요

2. 환경 설정 파일 정리
   - 환경 설정 파일들을 `config/` 폴더로 통합 고려

3. 불필요한 파일 삭제
   - `scripts/99_deprecated_debug/` 폴더의 파일 검토 후 삭제

4. 문서 업데이트
   - 새로운 폴더 구조를 반영하도록 주요 문서 업데이트 