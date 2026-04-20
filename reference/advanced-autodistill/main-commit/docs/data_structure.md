# 카테고리 기반 폴더 구조 설계

이 문서는 AGI 이미지 처리 파이프라인을 위한 표준화된 카테고리 기반 폴더 구조를 정의합니다.

## 개요

각 카테고리는 표준화된 하위 폴더 세트를 포함하며, 이를 통해 데이터 관리와 프로세스 파이프라인이 일관되게 유지됩니다.

## 기본 폴더 구조

```
[프로젝트 루트]/data
├── [카테고리 1]
│   ├── 1.images           # 원본 이미지
│   ├── 2.support-set      # 지원 세트 이미지
│   ├── 3.box              # 바운딩 박스 데이터
│   ├── 4.mask             # 마스크 데이터
│   ├── 5.preprocessed     # 전처리된 이미지
│   ├── 6.results          # 결과 데이터
│   └── 7.dataset          # YOLO 형식 데이터셋
├── [카테고리 2]
│   ├── 1.images
│   ├── 2.support-set
│   └── ...
└── ...
```

## 각 폴더의 용도

1. **1.images**: 원본 이미지가 저장되는 폴더
   - 하위 폴더: `train`, `test`, `val`
   
2. **2.support-set**: 대조군으로 사용되는 이미지 
   - 각 클래스별 예시 이미지 포함
   
3. **3.box**: 바운딩 박스 정보
   - YOLO 형식 어노테이션 (.txt)
   - COCO 형식 어노테이션 (.json)
   
4. **4.mask**: 세그멘테이션 마스크
   - 픽셀 단위 마스크 이미지
   
5. **5.preprocessed**: 전처리된 이미지
   - 크기 조정, 정규화, 증강 등 적용된 이미지
   
6. **6.results**: 분석 결과
   - 예측 결과
   - 시각화 이미지
   - 평가 메트릭
   
7. **7.dataset**: YOLO 형식 데이터셋
   - YOLO 학습용 데이터셋
   - `data.yaml` 구성 파일

## 경로 구성 예시

프로젝트에서 폴더에 접근하는 방법:

```python
import os

# 기본 데이터 디렉토리
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def get_category_path(category_name):
    """지정된 카테고리의 경로를 반환합니다."""
    return os.path.join(BASE_DATA_DIR, category_name)

def get_images_path(category_name, subset="train"):
    """지정된 카테고리의 이미지 경로를 반환합니다."""
    category_path = get_category_path(category_name)
    return os.path.join(category_path, "1.images", subset)

def get_results_path(category_name):
    """지정된 카테고리의 결과 경로를 반환합니다."""
    category_path = get_category_path(category_name)
    return os.path.join(category_path, "6.results")
```

## 경로 해결 전략

데이터 경로 해결을 위해 다음과 같은 전략을 사용합니다:

1. **상대 경로 사용**: 프로젝트의 이식성을 위해 절대 경로 대신 상대 경로 사용
2. **환경 변수**: 필요한 경우 기본 경로를 재정의하기 위한 환경 변수 지원
3. **구성 파일**: 경로 구성 설정을 위한 JSON 또는 YAML 구성 파일

## 확장성

이 폴더 구조는 필요에 따라 확장할 수 있습니다:

- 새로운 카테고리 추가 가능
- 각 카테고리 내에 추가 전문 폴더 생성 가능 