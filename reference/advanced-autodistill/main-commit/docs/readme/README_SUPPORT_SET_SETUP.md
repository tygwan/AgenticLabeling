# Support Set 설정 가이드

이 문서는 Few-Shot Learning을 위한 Support Set을 체계적으로 구성하고 관리하는 방법을 설명합니다.

## 개요

Few-Shot Learning 시스템은 Support Set의 이미지를 기반으로 새로운 이미지를 분류합니다. Shot 수(1, 5, 10, 30)는 각 클래스당 사용되는 예제 이미지의 수를 의미합니다. 실험 결과의 일관성과 재현성을 보장하기 위해, 이 가이드는 명확하게 구조화된 Support Set을 생성하는 방법을 제공합니다.

## 1. Support Set 구조

### 1.1 기존 구조 (원본)

기존 구조는 단일 Support Set 폴더 내에 모든 이미지가 클래스별로 저장되어 있습니다:

```
data/test_category/2.support-set/
  ├── class_0/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ... (30+ 이미지)
  ├── class_1/
  │   ├── image1.jpg
  │   └── ... (30+ 이미지)
  └── ...
```

이 구조에서는 매 실행마다 각 클래스에서 상위 N개(shot 수)의 이미지만 선택하여 사용합니다. 이 방식은 실행 시마다 동일한 이미지가 선택되지만, 어떤 이미지가 선택되는지 명확하지 않을 수 있습니다.

### 1.2 구조화된 형식 (새로운 구조)

새로 도입된 구조화된 형식은 각 shot 수에 대해 별도의 디렉토리를 생성합니다:

```
data/test_category/2.support-set-structured/
  ├── 1-shot/
  │   ├── class_0/
  │   │   └── 01_image1.jpg
  │   ├── class_1/
  │   │   └── 01_image1.jpg
  │   └── ...
  ├── 5-shot/
  │   ├── class_0/
  │   │   ├── 01_image1.jpg
  │   │   ├── 02_image2.jpg
  │   │   └── ... (5개 이미지)
  │   └── ...
  ├── 10-shot/
  │   └── ...
  ├── 30-shot/
  │   └── ...
  └── support_set_info.json (구조화된 정보 파일)
```

이 구조는 다음과 같은 장점을 제공합니다:
- 각 shot 수마다 사용되는 이미지가 명확히 정의됨
- 실험 결과의 일관성과 재현성 보장
- 이미지 선택에 대한 불확실성 제거

## 2. Support Set 관리 도구 사용법

### 2.1 구조화된 Support Set 생성

```bash
# 기본 사용법
./scripts/run_support_set_manager.sh test_category --create

# 기존 구조화 디렉토리를 덮어쓰려면
./scripts/run_support_set_manager.sh test_category --create --force
```

이 명령은 원본 Support Set에서 이미지를 읽어 각 shot 수(1, 5, 10, 30)에 맞게 구조화된 디렉토리를 생성합니다.

### 2.2 Support Set 상태 확인

```bash
# Support Set 유효성 검증
./scripts/run_support_set_manager.sh test_category --validate

# 상세 보고서 생성
./scripts/run_support_set_manager.sh test_category --report
```

유효성 검증은 모든 클래스에 충분한 이미지(최대 shot 수인 30개)가 있는지 확인합니다. 보고서는 클래스별 이미지 수와 각 shot 수에 대한 커버리지 정보를 제공합니다.

## 3. Few-Shot 실험 실행

구조화된 Support Set을 사용하여 실험을 실행하는 방법:

```bash
# CLI 모드로 실행
python run_few_shot_platform.py --cli --category test_category

# 웹 앱 모드로 실행
python run_few_shot_platform.py --webapp
```

시스템은 자동으로 구조화된 Support Set을 감지하고 사용합니다. 구조화된 디렉토리가 없으면 원본 Support Set을 사용합니다.

## 4. Support Set 준비 체크리스트

1. 각 클래스에 최소 30개의 대표 이미지 준비 (더 많을수록 좋음)
2. 이미지를 `data/<카테고리>/2.support-set/<클래스>/` 디렉토리에 저장
3. 구조화된 Support Set 생성: `./scripts/run_support_set_manager.sh <카테고리> --create`
4. 유효성 검증: `./scripts/run_support_set_manager.sh <카테고리> --validate`
5. 필요한 경우 부족한 클래스에 이미지 추가 후 다시 구조화

## 5. 고급 사용법

### 5.1 Support Set 분석

```bash
# 상세 보고서 생성
./scripts/run_support_set_manager.sh test_category --report
```

보고서는 다음 정보를 제공합니다:
- 각 클래스의 이미지 수
- 각 shot 수에 대한 클래스 커버리지 비율
- 이미지가 부족한 클래스 목록
- 전체 Support Set의 유효성 여부

### 5.2 이미지 준비 팁

- 각 클래스당 최소 30개, 이상적으로 50개 이상의 이미지 준비
- 다양한 각도, 조명, 배경에서 촬영된 이미지 포함
- 이미지 파일명에 의미 있는 정보 포함 (예: `object_front_01.jpg`, `object_side_02.jpg`)
- 불필요한 배경이 없는 깨끗한 이미지 사용
- 모든 클래스에 비슷한 품질과 수의 이미지 제공

## 6. 문제 해결

### 6.1 "클래스에 이미지가 부족합니다" 경고

이 경고는 특정 클래스에 최대 shot 수(일반적으로 30)보다 적은 이미지가 있을 때 표시됩니다. 해당 클래스에 더 많은 이미지를 추가한 후 구조화된 Support Set을 다시 생성하세요.

### 6.2 "구조화된 support set 디렉토리가 이미 존재합니다" 경고

기존 구조화된 디렉토리를 덮어쓰려면 `--force` 옵션을 사용하세요:

```bash
./scripts/run_support_set_manager.sh test_category --create --force
```

### 6.3 구조화된 Support Set이 인식되지 않음

구조화된 디렉토리 경로가 올바른지 확인하세요:
```bash
ls -la data/test_category/2.support-set-structured/
```

디렉토리 이름이 정확히 `2.support-set-structured`인지 확인하세요. 