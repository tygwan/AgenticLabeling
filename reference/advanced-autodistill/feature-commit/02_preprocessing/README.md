# Phase 2: 데이터 전처리 및 구조화

## 개요
Support Set을 구조화하고 데이터를 정제하는 단계입니다.

## 주요 스크립트
- `restructure_support_set.py`: Support Set N-shot별 구조화
- `support_set_manager.py`: Support Set 관리
- `refine_dataset.py`: 데이터셋 정제

## 실행 방법
```bash
python restructure_support_set.py --category test_category --shots 1,5,10,30
```
