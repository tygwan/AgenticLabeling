# Shot*Threshold 결과 폴더 조직화 가이드

이 문서는 Few-Shot 분류 결과를 클래스별 폴더로 조직화하고, Ground Truth를 관리하는 새로운 기능에 대해 설명합니다.

## 개요

Few-Shot 분류 결과를 효과적으로 시각화하고 평가하기 위해 아래와 같은 기능이 추가되었습니다:

1. **클래스별 폴더 조직화**: 각 shot*threshold 조합마다 class_0, class_1, class_2, class_3, unknown 폴더 생성
2. **Ground Truth 생성**: 최적의 shot*threshold 조합을 Ground Truth 폴더로 복사
3. **Ground Truth 평가**: 모든 shot*threshold 조합의 성능을 Ground Truth와 비교

## 1. 클래스별 폴더 조직화

Few-Shot 분류 결과 JSON 파일을 기반으로 이미지를 클래스별 폴더로 복사합니다.

### 사용 방법:

```bash
# 이미 실행된 Few-Shot 분류 결과에 대해 클래스별 폴더 생성
./scripts/run_organize_results.sh test_category

# 기존 폴더를 덮어쓰려면 --force 옵션 사용
./scripts/run_organize_results.sh test_category --force

# 특정 shot 값과 threshold 값만 처리
./scripts/run_organize_results.sh test_category --shot-values=5,10 --threshold-values=0.7,0.8
```

### 결과:

각 shot*threshold 조합마다 다음과 같은 폴더 구조가 생성됩니다:

```
data/
└── test_category/
    └── 7.results/
        ├── shot_1/
        │   ├── threshold_0.5/
        │   │   ├── class_0/          # class_0으로 분류된 이미지
        │   │   ├── class_1/          # class_1으로 분류된 이미지
        │   │   ├── class_2/          # class_2으로 분류된 이미지
        │   │   ├── class_3/          # class_3으로 분류된 이미지
        │   │   ├── unknown/          # unknown으로 분류된 이미지
        │   │   └── results.json      # 원본 결과 파일
        │   ├── threshold_0.6/
        │   │   └── ...
        │   └── ...
        ├── shot_5/
        │   └── ...
        └── ...
```

또한 모든 조합의 분류 결과에 대한 요약 정보가 `classification_summary.json` 파일에 저장됩니다.

## 2. Ground Truth 생성

특정 shot*threshold 조합의 결과를 Ground Truth 폴더로 복사하여 Ground Truth 초안을 생성할 수 있습니다.

### 방법 1: Few-Shot Platform과 함께 실행

```bash
# Few-Shot 분류 실행 및 Ground Truth 생성
python scripts/run_few_shot_platform.py --category test_category --ground-truth 10,0.7

# 기존 Ground Truth 폴더를 덮어쓰려면 --force 옵션 추가
python scripts/run_few_shot_platform.py --category test_category --ground-truth 10,0.7 --force

# 사용자 지정 Ground Truth 디렉토리 지정
python scripts/run_few_shot_platform.py --category test_category --ground-truth 10,0.7 --ground-truth-dir data/test_category/custom_ground_truth
```

### 방법 2: 직접 Ground Truth 생성

이미 생성된 shot*threshold 폴더에서 Ground Truth 폴더를 생성할 수도 있습니다.

```bash
# 수동으로 ground_truth 폴더를 만들고 각 class 폴더를 복사
mkdir -p data/test_category/7.results/ground_truth
cp -r data/test_category/7.results/shot_10/threshold_0.7/class_* data/test_category/7.results/ground_truth/
```

## 3. Ground Truth 수동 편집

생성된 Ground Truth 폴더는 다음과 같이 수동으로 편집할 수 있습니다:

1. 잘못 분류된 이미지를 올바른 클래스 폴더로 이동
2. 불필요한 이미지 삭제
3. unknown으로 분류된 이미지를 적절한 클래스 폴더로 이동

예시:
```bash
# unknown 폴더에서 class_1 폴더로 이미지 이동
mv data/test_category/7.results/ground_truth/unknown/image123.jpg data/test_category/7.results/ground_truth/class_1/

# class_0 폴더에서 class_2 폴더로 잘못 분류된 이미지 이동
mv data/test_category/7.results/ground_truth/class_0/image456.jpg data/test_category/7.results/ground_truth/class_2/
```

## 4. Ground Truth 평가

수동으로 편집된 Ground Truth 폴더를 기준으로 모든 shot*threshold 조합의 성능을 평가합니다.

```bash
# Ground Truth 평가 실행
./scripts/run_ground_truth_evaluator.sh test_category

# 시각화 그래프 생성
./scripts/run_ground_truth_evaluator.sh test_category --visualize

# 사용자 지정 Ground Truth 폴더 지정
./scripts/run_ground_truth_evaluator.sh test_category data/test_category/custom_ground_truth --visualize
```

### 평가 결과:

평가 결과는 다음 파일에서 확인할 수 있습니다:

- **요약 보고서**: `data/{category}/7.results/evaluation_summary.txt`
- **상세 데이터**: `data/{category}/7.results/evaluation_report.json`
- **정확도 그래프**: `data/{category}/7.results/accuracy_comparison.png`
- **혼동 행렬**: `data/{category}/7.results/best_confusion_matrix.png`

## 5. 권장 워크플로우

효율적인 Ground Truth 생성 및 평가를 위한 권장 워크플로우:

1. Few-Shot 분류 실행 및 자동 폴더 구성:
   ```bash
   python scripts/run_few_shot_platform.py --category test_category
   ```

2. 생성된 분류 결과 확인 및 최적 조합 선택:
   ```bash
   ./scripts/run_organize_results.sh test_category
   ```

3. 선택한 최적 조합을 Ground Truth 초안으로 복사:
   ```bash
   python scripts/run_few_shot_platform.py --category test_category --ground-truth 10,0.7
   ```

4. Ground Truth 폴더 수동 편집 (파일 탐색기 또는 명령줄 사용)

5. Ground Truth 평가 실행:
   ```bash
   ./scripts/run_ground_truth_evaluator.sh test_category --visualize
   ```

6. 필요한 경우 Ground Truth 조정 및 평가 반복

## 6. 참고 사항

- 원본 이미지는 `data/{category}/5.dataset/val` 디렉토리에서 가져옵니다.
- 클래스별 폴더에는 이미지 파일만 복사되며, 원본 파일은 변경되지 않습니다.
- Ground Truth 폴더를 수정한 후에는 반드시 평가를 다시 실행하여 변경 사항이 반영되었는지 확인하세요. 