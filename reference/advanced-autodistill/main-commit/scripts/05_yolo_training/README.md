# Phase 5: YOLO 학습 및 데이터셋 생성

## 개요
YOLO 세그먼테이션 데이터셋을 생성하고 모델을 학습하는 단계입니다.

## 주요 스크립트
- `create_yolo_segmentation_dataset.py`: YOLO 데이터셋 생성
- `train_yolo_segmentation.py`: YOLO 모델 학습

## 실행 방법
```bash
# 데이터셋 생성
python create_yolo_segmentation_dataset.py --category test_category --output data/test_category/8.yolo-dataset

# 모델 학습
python train_yolo_segmentation.py --data data/test_category/8.yolo-dataset/dataset.yaml --epochs 100 --copy-paste 0.3
```
