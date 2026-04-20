# Phase 1: 데이터 준비 및 초기 처리

## 개요
원본 이미지에서 객체를 탐지하고 마스크를 생성하는 초기 데이터 처리 단계입니다.

## 주요 스크립트
- `main_launcher.py`: 메인 파이프라인 런처
- `autodistill_runner.py`: Autodistill + SAM2 실행기
- `advanced_preprocessor.py`: 고급 이미지 전처리

## 실행 방법
```bash
python main_launcher.py --category test_category --plot --preprocess
```
