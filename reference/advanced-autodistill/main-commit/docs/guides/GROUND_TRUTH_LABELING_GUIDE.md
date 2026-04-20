# 폴더 기반 Ground Truth Labeling 가이드

이 가이드는 폴더 기반 Ground Truth Labeling 시스템을 사용하여 이미지 데이터셋에 Ground Truth 레이블을 효율적으로 할당하는 방법을 설명합니다.

## 1. 개요

폴더 기반 Ground Truth Labeling 시스템은 기존의 이미지별 선택 방식 대신 다음과 같은 개선점을 제공합니다:

- **폴더 탐색기 스타일 인터페이스**: 이미지를 직관적으로 탐색하고 분류
- **배치 선택 및 처리**: 여러 이미지를 한 번에 선택하여 동일한 클래스로 지정
- **듀얼 뷰 모드**: 원본 이미지와 Ground Truth 폴더 모두 탐색 가능
- **클래스 필터링**: 특정 클래스에 속한 이미지만 빠르게 보기
- **자동 통계**: Ground Truth 레이블링 진행 상황 및 클래스 분포 확인

## 2. 새로운 폴더 기반 워크플로우

### 2.1 Few-Shot 분류 결과에 클래스별 폴더 자동 생성

Few-Shot 분류 과정이 완료된 후, 각 shot*threshold 조합마다 class_0, class_1, class_2, class_3, unknown 폴더가 자동으로 생성됩니다. 이를 통해 분류 결과를 시각적으로 쉽게 확인할 수 있습니다.

#### 실행 방법:

```bash
# 이미 실행된 Few-Shot 분류 결과에 대해 클래스별 폴더 생성
./scripts/run_organize_results.sh test_category --force
```

옵션:
- `--shot-values=1,5,10,30`: 처리할 shot 값 지정
- `--threshold-values=0.5,0.6,0.7,0.8,0.9`: 처리할 threshold 값 지정
- `--force`: 기존 폴더가 있다면 덮어쓰기

### 2.2 Ground Truth 폴더 생성

최적의 shot*threshold 조합을 선택하여 Ground Truth 폴더로 쉽게 복사할 수 있습니다.

#### 방법 1: Few-Shot Platform과 함께 실행

```bash
python scripts/run_few_shot_platform.py --category test_category --ground-truth 10,0.7
```

#### 방법 2: 이미 생성된 분류 결과에서 직접 Ground Truth 생성

```bash
python scripts/evaluate_ground_truth.py --category test_category --ground-truth-dir data/test_category/7.results/ground_truth
```

### 2.3 Ground Truth 폴더 수동 편집

생성된 Ground Truth 폴더에서 다음과 같이 이미지를 수동으로 편집할 수 있습니다:

1. 잘못 분류된 이미지를 올바른 클래스 폴더로 이동
2. 불필요한 이미지 삭제
3. unknown으로 분류된 이미지를 적절한 클래스 폴더로 이동

### 2.4 Ground Truth 평가

수동으로 편집된 Ground Truth 폴더를 기준으로 모든 shot*threshold 조합의 성능을 평가합니다.

```bash
./scripts/run_ground_truth_evaluator.sh test_category --visualize
```

## 3. 기존 Labeling 도구 사용

### 시스템 실행

```bash
# 터미널에서 실행
chmod +x scripts/run_ground_truth_labeler.sh
./scripts/run_ground_truth_labeler.sh
```

실행 후 웹 브라우저에서 Gradio 인터페이스가 열립니다(기본 주소: http://localhost:7860).

### 3.1 데이터 로드

1. 좌측 패널에서 **카테고리 선택** 드롭다운을 사용하여 데이터 카테고리 선택
2. **실험 선택** 드롭다운에서 레이블링할 실험 선택
3. **실험 로드** 버튼 클릭하여 데이터 로드
4. 로드가 완료되면 이미지 갤러리에 이미지가 표시됩니다

### 3.2 이미지 탐색 및 필터링

- **뷰 선택**: 원본 이미지 또는 Ground Truth 이미지 중 확인할 뷰 선택
- **클래스 필터링**: 특정 클래스에 속한 이미지만 표시하도록 필터링
- **페이지 이동**: 갤러리 하단의 페이지 버튼으로 이미지 페이지 간 이동
- **정렬 옵션**: 다양한 정렬 기준으로 이미지 정렬

### 3.3 이미지 레이블링

#### 단일 이미지 레이블링:
1. 이미지를 클릭하여 선택
2. 클래스 드롭다운에서 클래스 선택
3. **레이블 적용** 버튼 클릭

#### 배치 레이블링:
1. 체크박스로 여러 이미지 선택
2. 클래스 드롭다운에서 공통 클래스 선택
3. **배치 레이블 적용** 버튼 클릭

### 3.4 Ground Truth 관리

- **Ground Truth 저장**: 현재의 레이블링 상태를 Ground Truth로 저장
- **Ground Truth 로드**: 기존 Ground Truth 데이터 로드
- **Ground Truth 내보내기**: Ground Truth 데이터를 JSON 형식으로 내보내기
- **Ground Truth 통계**: 현재 Ground Truth 클래스 분포 확인

## 4. 평가 결과 해석

Ground Truth 평가 결과는 다음 파일에서 확인할 수 있습니다:

- **요약 보고서**: `data/{category}/7.results/evaluation_summary.txt`
- **상세 데이터**: `data/{category}/7.results/evaluation_report.json`
- **정확도 그래프**: `data/{category}/7.results/accuracy_comparison.png`
- **혼동 행렬**: `data/{category}/7.results/best_confusion_matrix.png`

주요 평가 지표:
- **정확도(Accuracy)**: 전체 이미지 중 올바르게 분류된 비율
- **정밀도(Precision)**: 특정 클래스로 분류된 이미지 중 실제로 해당 클래스인 비율
- **재현율(Recall)**: 실제 특정 클래스인 이미지 중 올바르게 해당 클래스로 분류된 비율
- **F1-Score**: 정밀도와 재현율의 조화 평균
- **클래스별 성능**: 각 클래스마다의 개별 성능 지표

## 5. 워크플로우 예시

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

4. 필요한 경우 Ground Truth 폴더 수동 편집

5. Ground Truth 평가 실행:
   ```bash
   ./scripts/run_ground_truth_evaluator.sh test_category --visualize
   ```

6. 평가 결과 확인 및 필요시 Ground Truth 조정 반복 