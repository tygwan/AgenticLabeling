# Scripts 폴더 정리 완료 보고서

## 📁 새로운 폴더 구조

```
scripts/
├── 01_data_preparation/     # 데이터 준비 및 초기 처리
├── 02_preprocessing/        # 데이터 전처리 및 구조화  
├── 03_classification/       # Few-Shot Learning 및 분류
├── 04_ground_truth/         # Ground Truth 생성 및 관리
├── 05_yolo_training/        # YOLO 학습 및 데이터셋 생성
├── 06_utilities/            # 공통 유틸리티 및 도구
└── 99_deprecated_debug/     # 사용하지 않는 파일들
```

## 🚀 워크플로우별 실행 가이드

### Phase 1: 데이터 준비
```bash
python scripts/01_data_preparation/main_launcher.py --category test_category --plot --preprocess
```

### Phase 2: 전처리
```bash
python scripts/02_preprocessing/restructure_support_set.py --category test_category --shots 1,5,10,30
```

### Phase 3: 분류
```bash
python scripts/03_classification/run_few_shot_platform.py --webapp
```

### Phase 4: Ground Truth
```bash
python scripts/04_ground_truth/ground_truth_labeler.py --category test_category
```

### Phase 5: YOLO 학습
```bash
python scripts/05_yolo_training/train_yolo_segmentation.py --data data/test_category/8.yolo-dataset/dataset.yaml --epochs 100 --copy-paste 0.3
```

## 📋 정리 완료 항목

✅ 워크플로우에 따른 폴더 구조 생성
✅ 파일들을 적절한 폴더로 이동
✅ 각 폴더별 README.md 생성
✅ 사용하지 않는 파일들 분리

## 🎯 다음 단계

1. 각 Phase별 README.md 내용 보완
2. 상대 경로 import 문제 해결
3. 사용하지 않는 파일들 검토 후 삭제
4. 워크플로우 가이드 업데이트
