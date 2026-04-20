# Phase 3: Few-Shot Learning 및 분류

## 개요
Few-Shot Learning을 통한 분류 및 성능 분석 단계입니다.

## 주요 스크립트
- `run_few_shot_platform.py`: Few-Shot 웹 플랫폼
- `run_shot_threshold_experiments.py`: 배치 실험 수행
- `analyze_experiment_metrics.py`: 결과 분석

## 실행 방법
```bash
# 웹 인터페이스
python run_few_shot_platform.py --webapp

# CLI 배치 실험
python run_shot_threshold_experiments.py --category test_category --models resnet,dino
```
