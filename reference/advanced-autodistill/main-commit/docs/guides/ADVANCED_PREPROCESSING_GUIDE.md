# Advanced Preprocessing Guide for Project-AGI

이 가이드는 기존 마스크전처리_class.py의 고품질 이미지 추출 로직을 project-agi에 통합한 고급 전처리 시스템의 사용법을 설명합니다.

## 주요 개선사항

### 1. 기존 방식 vs 새로운 방식

| 구분 | 기존 마스크전처리_class.py | 새로운 Advanced Preprocessor |
|------|---------------------------|------------------------------|
| **마스크 처리** | RLE(Run Length Encoding) 정밀 마스크 | RLE + 폴리곤 좌표 지원 |
| **데이터 형식** | 별도 box.txt, mask.json | 통합 JSON 형식 |
| **처리 방식** | 클래스별 개별 처리 | 배치 처리 + 클래스별 선택 |
| **품질 제어** | 수동 GUI 기반 | 자동화된 고품질 추출 |
| **메모리 관리** | 단일 이미지 처리 | 배치 처리 + 메모리 최적화 |

### 2. 핵심 기능

- **RLE 마스크 디코딩**: 정밀한 객체 경계 추출
- **컨투어 기반 크롭**: 객체 형태에 맞는 최적 크롭
- **클래스별 필터링**: 특정 클래스만 선택적 처리
- **품질 보장**: 패딩, 리사이징, 마스크 적용 최적화
- **배치 처리**: 대용량 데이터 효율적 처리

## 설치 및 설정

### 1. 필요한 라이브러리 확인
```bash
# 기본 라이브러리들이 설치되어 있는지 확인
pip install opencv-python numpy tqdm
```

### 2. 프로젝트 구조 확인
```
project-agi/
├── data/
│   └── test_category/
│       ├── 1.images/          # 원본 이미지
│       ├── 3.box/             # 박스 데이터 (JSON)
│       ├── 4.mask/            # 마스크 데이터 (JSON)
│       └── 6.preprocessed/    # 전처리 결과
│           ├── Class_0/       # 클래스별 디렉토리
│           ├── Class_1/
│           ├── Class_2/
│           └── Class_3/
└── scripts/
    ├── advanced_preprocessor.py
    ├── data_converter.py
    └── main_launcher.py
```

## 사용법

### 1. 기존 데이터 변환 (필요한 경우)

기존 마스크전처리_class.py에서 사용하던 데이터가 있다면 먼저 변환:

```bash
# 기존 데이터를 project-agi 형식으로 변환
python scripts/data_converter.py \
    --input_dir /path/to/original/data \
    --category test_category \
    --output_report conversion_report.txt
```

### 2. 고품질 전처리 실행

#### 방법 1: main_launcher.py를 통한 통합 실행
```bash
# 전체 파이프라인 + 고품질 전처리
python scripts/main_launcher.py \
    --category test_category \
    --advanced-preprocess \
    --target-size 224,224 \
    --max-images 5000 \
    --batch-size 100

# 특정 클래스만 처리
python scripts/main_launcher.py \
    --category test_category \
    --advanced-preprocess \
    --preprocess-class-id 0 \
    --target-size 224,224
```

#### 방법 2: 독립적인 고품질 전처리
```bash
# 모든 클래스 처리
python scripts/advanced_preprocessor.py \
    --category test_category \
    --target_size 224,224 \
    --quality high

# 특정 클래스만 처리 (예: fence_person = class_id 0)
python scripts/advanced_preprocessor.py \
    --category test_category \
    --class_id 0 \
    --target_size 224,224 \
    --max_images 1000 \
    --output_report preprocessing_report.txt
```

### 3. 처리 결과 확인

```bash
# 결과 디렉토리 확인
ls -la data/test_category/6.preprocessed/

# 클래스별 객체 수 확인
find data/test_category/6.preprocessed/ -name "*.png" | wc -l

# 클래스별 분포 확인
for class_dir in data/test_category/6.preprocessed/Class_*; do
    echo "$(basename $class_dir): $(ls $class_dir/*.png 2>/dev/null | wc -l) objects"
done
```

## 고급 옵션

### 1. 품질 레벨 설정
```bash
# 고품질 (기본값) - RLE 마스크 + 정밀 크롭
python scripts/advanced_preprocessor.py --quality high

# 중품질 - 박스 기반 + 마스크 적용
python scripts/advanced_preprocessor.py --quality medium

# 저품질 - 박스 기반만
python scripts/advanced_preprocessor.py --quality low
```

### 2. 메모리 최적화
```bash
# 메모리 모니터링 활성화
python scripts/main_launcher.py \
    --category test_category \
    --advanced-preprocess \
    --memory-monitor \
    --batch-size 50

# 작은 배치 크기로 메모리 사용량 줄이기
python scripts/advanced_preprocessor.py \
    --category test_category \
    --max_images 100
```

### 3. 클래스별 처리
```bash
# fence_person (class_id: 0)만 처리
python scripts/advanced_preprocessor.py \
    --category test_category \
    --class_id 0

# sidewalk (class_id: 1)만 처리  
python scripts/advanced_preprocessor.py \
    --category test_category \
    --class_id 1

# car (class_id: 2)만 처리
python scripts/advanced_preprocessor.py \
    --category test_category \
    --class_id 2

# traffic cone (class_id: 3)만 처리
python scripts/advanced_preprocessor.py \
    --category test_category \
    --class_id 3
```

## 데이터 형식 상세

### 1. 입력 데이터 형식

#### Box 데이터 (3.box/*.json)
```json
{
  "image_path": "A_1_1_frame_0001.png",
  "boxes": [[x1, y1, x2, y2], ...],
  "class_ids": [0, 1, 2, 3],
  "classes": ["fence_person", "sidewalk", "car", "traffic cone"],
  "confidence": [1.0, 1.0, 1.0, 1.0]
}
```

#### Mask 데이터 (4.mask/*.json)
```json
[
  {
    "class_id": 0,
    "class_name": "fence_person",
    "confidence": 1.0,
    "mask_rle": [17, 53, 1601, 53, ...]
  }
]
```

### 2. 출력 데이터 형식

전처리된 이미지는 다음과 같은 명명 규칙으로 저장됩니다:
```
{원본이미지명}_obj{객체번호}_cls{클래스ID}_{클래스명}.png
```

예시:
```
A_1_1_frame_0001_obj0_cls0_fence_person.png
A_1_1_frame_0001_obj1_cls1_sidewalk.png
A_1_1_frame_0001_obj2_cls2_car.png
```

## 성능 최적화 팁

### 1. 배치 크기 조정
- **메모리 16GB 이상**: `--batch-size 200`
- **메모리 8-16GB**: `--batch-size 100` (기본값)
- **메모리 8GB 미만**: `--batch-size 50`

### 2. 이미지 크기 최적화
- **분류용**: `--target-size 224,224`
- **검출용**: `--target-size 640,640`
- **고해상도**: `--target-size 512,512`

### 3. 클래스별 순차 처리
메모리가 부족한 경우 클래스별로 나누어 처리:
```bash
for class_id in 0 1 2 3; do
    python scripts/advanced_preprocessor.py \
        --category test_category \
        --class_id $class_id \
        --target_size 224,224
done
```

## 문제 해결

### 1. 메모리 부족 오류
```bash
# 배치 크기 줄이기
python scripts/advanced_preprocessor.py --max_images 100

# 클래스별 개별 처리
python scripts/advanced_preprocessor.py --class_id 0
```

### 2. RLE 디코딩 오류
- 마스크 데이터 형식 확인
- 이미지 크기와 RLE 데이터 일치성 확인

### 3. 빈 결과 디렉토리
- 입력 데이터 경로 확인
- 클래스 매핑 파일 존재 확인
- 로그 메시지에서 오류 확인

## 품질 비교

### 기존 방식 대비 개선점

1. **정밀도 향상**: RLE 마스크로 픽셀 단위 정확한 객체 추출
2. **자동화**: GUI 없이 배치 처리로 대용량 데이터 처리
3. **일관성**: 동일한 품질 기준으로 모든 이미지 처리
4. **효율성**: 메모리 최적화로 안정적인 대용량 처리
5. **확장성**: 새로운 클래스나 데이터 형식 쉽게 추가

### 예상 결과

- **4959장 원본 이미지** → **약 20,000개 고품질 객체 이미지**
- **처리 시간**: 약 30-60분 (하드웨어에 따라)
- **메모리 사용량**: 2-8GB (배치 크기에 따라)
- **품질**: 기존 마스크전처리_class.py와 동등 이상

이 시스템을 통해 기존의 고품질 전처리 결과를 project-agi 환경에서 자동화되고 확장 가능한 방식으로 재현할 수 있습니다. 