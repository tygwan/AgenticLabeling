# Project AGI - 완전한 워크플로우 가이드

## 📋 개요

이 프로젝트는 이미지 분석부터 AI 모델 학습까지의 완전한 파이프라인을 제공합니다:
**원본 이미지 → Box/Mask 추출 → 데이터 전처리 → Few-Shot Learning → Ground Truth 생성 → YOLO 세그먼테이션 학습**

---

## 🎯 전체 워크플로우 개요

### Phase 1: 데이터 준비 및 초기 처리
- 원본 이미지 준비
- Support Set 구성
- Autodistill + SAM2를 이용한 자동 객체 탐지 및 마스크 생성

### Phase 2: 데이터 전처리 및 분류
- 마스크 데이터 전처리
- 객체 크롭 및 정규화
- Few-Shot Learning을 통한 초기 분류

### Phase 3: Ground Truth 생성 및 검증
- 분류 결과 검토 및 수정
- Ground Truth 데이터셋 생성
- 품질 평가 및 최적화

### Phase 4: 최종 모델 학습
- YOLO 세그먼테이션 데이터셋 생성
- 데이터 증강을 포함한 모델 학습
- 성능 평가 및 모델 배포

---

## 🔧 Phase별 상세 가이드

## Phase 1: 데이터 준비 및 초기 처리

### 1.1 환경 설정
```bash
# 의존성 설치
./install_dependencies.sh

# 기본 디렉토리 구조 생성
mkdir -p data/test_category/1.images
mkdir -p data/test_category/2.support-set
```

### 1.2 데이터 준비
```bash
# 원본 이미지를 1.images/에 배치
cp /path/to/your/images/* data/test_category/1.images/

# 각 클래스별 예시 이미지를 2.support-set/에 배치
# data/test_category/2.support-set/class_0/
# data/test_category/2.support-set/class_1/
# data/test_category/2.support-set/class_2/
# data/test_category/2.support-set/class_3/
```

### 1.3 자동 객체 탐지 및 마스크 생성
```bash
# 전체 파이프라인 실행 (Autodistill + SAM2)
python scripts/01_data_preparation/main_launcher.py \
    --category test_category \
    --plot \
    --preprocess

# 또는 bash 스크립트 사용
./run_pipeline.sh -c test_category
```

**생성되는 결과물:**
- `3.box/`: 바운딩 박스 좌표 데이터
- `4.mask/`: 마스크 이미지 및 JSON 좌표 데이터
- `6.preprocessed/`: 전처리된 객체 이미지들
- `7.results/`: 시각화 결과

---

## Phase 2: 데이터 전처리 및 분류

### 2.1 Support Set 구조화
```bash
# N-shot별 Support Set 생성
python scripts/02_preprocessing/restructure_support_set.py \
    --category test_category \
    --shots 1,5,10,30
```

### 2.2 Few-Shot Learning 분류 실험

#### 웹 인터페이스 사용
```bash
# Few-Shot Learning 웹 플랫폼 실행
python scripts/03_classification/run_few_shot_platform.py --webapp
```

#### CLI 배치 실험
```bash
# 다양한 N-shot/threshold 조합으로 자동 실험
python scripts/03_classification/run_shot_threshold_experiments.py \
    --category test_category \
    --models resnet,dino \
    --shots 1,5,10,30 \
    --thresholds 0.5,0.6,0.7,0.8,0.9
```

### 2.3 분류 결과 분석
```bash
# 실험 결과 종합 분석
python scripts/03_classification/analyze_experiment_metrics.py \
    --category test_category

# 모델별 성능 비교
python scripts/03_classification/run_model_comparison.py \
    --category test_category
```

---

## Phase 3: Ground Truth 생성 및 검증

### 3.1 Ground Truth 라벨링
```bash
# 대화형 라벨링 도구 실행
python scripts/04_ground_truth/ground_truth_labeler.py \
    --category test_category

# 또는 bash 스크립트 사용
./scripts/04_ground_truth/run_ground_truth_labeler.sh
```

**주요 기능:**
- 배치 선택 및 라벨링
- 클래스별 필터링 및 검색
- 실시간 통계 및 진행률 표시
- Ground Truth 기준선 설정 및 적용

### 3.2 Ground Truth 평가 및 검증
```bash
# Ground Truth 품질 평가
python scripts/04_ground_truth/evaluate_ground_truth.py \
    --category test_category

# 실험 결과와 Ground Truth 비교 분석
python scripts/04_ground_truth/run_ground_truth_evaluator.sh
```

### 3.3 분류 결과 정리
```bash
# 분류 결과 체계적 정리
python scripts/04_ground_truth/organize_classification_results.py \
    --category test_category

# 또는 bash 스크립트 사용
./scripts/04_ground_truth/run_organize_results.sh
```

---

## Phase 4: 최종 모델 학습

### 4.1 YOLO 세그먼테이션 데이터셋 생성

#### Ground Truth 기반 데이터셋 생성
```bash
# 정제된 Ground Truth로부터 YOLO 데이터셋 생성
python scripts/05_yolo_training/create_yolo_from_ground_truth_fixed.py \
    --category test_category \
    --output data/test_category/8.refine-dataset
```

#### 원본 마스크 기반 데이터셋 생성 (모든 윤곽점 보존)
```bash
# 모든 윤곽선 포인트를 유지하며 YOLO 데이터셋 생성
python scripts/05_yolo_training/create_yolo_segmentation_dataset.py \
    --category test_category \
    --output data/test_category/8.yolo-dataset \
    --verbose
```

### 4.2 YOLO 세그먼테이션 모델 학습

#### 기본 학습 (Copy-Paste 증강 포함)
```bash
python scripts/05_yolo_training/train_yolo_segmentation.py \
    --data data/test_category/8.refine-dataset/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --copy-paste 0.3 \
    --name refined_augmented_train
```

#### 고급 증강 설정으로 학습
```bash
python scripts/05_yolo_training/train_yolo_segmentation.py \
    --data data/test_category/8.refine-dataset/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --copy-paste 0.3 \
    --degrees 10 \
    --translate 0.1 \
    --scale 0.5 \
    --shear 2.0 \
    --flipud 0.5 \
    --fliplr 0.5 \
    --mixup 0.1 \
    --hsv-h 0.015 \
    --hsv-s 0.7 \
    --hsv-v 0.4 \
    --mosaic 1.0 \
    --name advanced_augmented_train
```

---

## 📁 데이터 디렉토리 구조

```
data/test_category/
├── 1.images/                      # 원본 입력 이미지
├── 2.support-set/                 # 클래스별 예시 이미지
│   ├── class_0/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
├── 2.support-set-structured/      # 구조화된 Support Set
│   ├── 1-shot/
│   ├── 5-shot/
│   ├── 10-shot/
│   └── 30-shot/
├── 3.box/                         # 바운딩 박스 데이터
├── 4.mask/                        # 마스크 및 좌표 데이터
├── 5.dataset/                     # YOLO 원본 데이터셋
├── 6.preprocessed/                # 전처리된 이미지 (클래스별)
│   ├── Class_0/
│   ├── Class_1/
│   ├── Class_2/
│   └── Class_3/
├── 7.results/                     # 분류 및 분석 결과
│   ├── analysis_results/
│   ├── dino/
│   ├── resnet/
│   ├── ground_truth/
│   ├── model_comparison/
│   └── visualizations/
└── 8.refine-dataset/             # 정제된 YOLO 세그먼테이션 데이터셋
    ├── train/
    ├── val/
    └── dataset.yaml
```

---

## 🎛️ 주요 설정 파라미터

### Few-Shot Learning 설정
- **N-shot 값**: 1, 5, 10, 30 (클래스당 사용할 예시 이미지 수)
- **Threshold 값**: 0.50 ~ 0.95 (유사도 임계값)
- **분류기 모델**: ResNet, DINOv2

### YOLO 학습 설정
- **Epochs**: 100-200 (데이터셋 크기에 따라 조정)
- **Batch Size**: 16 (GPU 메모리에 따라 조정)
- **Image Size**: 640 (표준 YOLO 입력 크기)
- **Data Augmentation**: Copy-paste, Mosaic, Mixup 등

---

## 🚨 주의사항 및 팁

### 데이터 품질
- 원본 이미지는 고해상도로 준비 (최소 640x640 권장)
- Support Set은 각 클래스당 최소 30장 이상 준비
- 클래스 간 명확한 구분이 가능한 예시 이미지 선택

### 성능 최적화
- GPU 메모리에 따라 배치 크기 조정
- 대용량 데이터셋의 경우 단계별로 처리
- 실험 결과를 기반으로 최적 파라미터 선택

### 문제 해결
- 메모리 부족 시 배치 크기 감소
- 분류 성능이 낮을 시 Support Set 품질 점검
- 학습이 느릴 시 이미지 크기 또는 모델 크기 조정

---

## 🔄 반복 개선 프로세스

1. **초기 실행**: 기본 설정으로 전체 파이프라인 실행
2. **결과 분석**: Few-Shot 분류 및 Ground Truth 품질 평가
3. **데이터 개선**: Support Set 보강 및 Ground Truth 수정
4. **파라미터 튜닝**: 최적 N-shot/threshold 조합 찾기
5. **최종 학습**: 최적화된 설정으로 YOLO 모델 학습

이 가이드를 따라 진행하면 원본 이미지부터 배포 가능한 세그먼테이션 모델까지 완성할 수 있습니다! 