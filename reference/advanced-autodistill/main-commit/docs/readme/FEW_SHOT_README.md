# Few-Shot Learning Experiment & Evaluation Platform (FSL-EEP)

이 플랫폼은 코사인 유사도 기반 Few-Shot 분류기를 다양한 N-shot 및 threshold 설정으로 자동 실험하고 결과를 평가하는 기능을 제공합니다.

## 주요 기능

1. 다양한 N-shot 및 threshold 조합으로 실험 자동화
2. 각 실험 결과의 체계적 저장 및 관리
3. Ground Truth 생성 및 관리
4. 다양한 평가 지표 계산 (Accuracy, Balanced Accuracy, F1, MCC 등)
5. Binary/Multi-class Confusion Matrix 생성
6. Annotation 정보 동기화

## 설치 방법

필요한 모든 의존성 패키지를 설치하기 위해 제공된 설치 스크립트를 실행하세요:

```bash
./install_few_shot_deps.sh
```

또는 직접 의존성 패키지를 설치할 수 있습니다:

```bash
pip install -r few_shot_requirements.txt
```

## 사용 방법

### 웹 애플리케이션 모드

웹 인터페이스를 통해 실험을 설정하고 결과를 시각화하는 방법:

```bash
python run_few_shot_platform.py --webapp
```

웹 인터페이스가 브라우저에서 자동으로 열립니다 (기본 포트: 7860).

### CLI 모드

명령줄에서 직접 실험을 실행하는 방법:

```bash
python run_few_shot_platform.py --cli --category test_category --model resnet
```

기본 옵션을 커스터마이즈하려면 다음과 같은 추가 인자를 사용할 수 있습니다:

```bash
python run_few_shot_platform.py --cli --category test_category --model dino --shots 1,5,10 --thresholds 0.7,0.8,0.9
```

## 디렉토리 구조

Few-Shot Learning 플랫폼은 다음과 같은 디렉토리 구조를 사용합니다:

```
data/
  <category_name>/
    1.images/            - 원본 이미지
    2.support-set/       - Few-Shot Learning용 지원 이미지 (클래스별 폴더 포함)
    3.box/               - 객체 바운딩 박스 정보
    4.mask/              - 객체 마스크/폴리곤 정보
    5.dataset/           - 데이터셋
    6.preprocessed/      - 전처리된 이미지 (분류 대상)
    7.results/           - 실험 결과
      shot_<n>/          - N-shot 설정별 결과
        threshold_<t>/   - 임계값별 결과
          predictions.csv             - 예측 결과
          ground_truth.csv           - Ground Truth 데이터 (레이블링 후 생성)
          evaluation_metrics.json    - 평가 지표
          confusion_matrix.png       - 혼동 행렬 시각화
          binary_confusion_matrices/ - 클래스별 이진 혼동 행렬
    8.refine-dataset/    - 정제된 데이터셋
    class_mapping.json   - 클래스 ID 매핑 정보
```

## 모델 옵션

현재 지원되는 모델:

1. **ResNet-50** (`resnet`): 가장 빠르고 가벼운 모델
2. **CLIP ViT-B/32** (`clip`): OpenAI의 CLIP 모델, 텍스트-이미지 이해에 강함
3. **DINOv2 ViT-B/14** (`dino`): Facebook의 DINO 모델, 고품질 특징 추출에 강함

## 문제 해결

### 일반적인 문제

1. **모듈 가져오기 오류**: 필요한 패키지가 설치되어 있는지 확인하세요.
   ```
   pip install -r few_shot_requirements.txt
   ```

2. **CUDA 메모리 부족**: GPU 메모리가 부족한 경우 다음과 같이 조치할 수 있습니다:
   - 더 가벼운 모델(`resnet`) 사용
   - 환경 변수 설정: `export CUDA_VISIBLE_DEVICES="0"`

3. **CLIP 모델 오류**: CLIP 패키지가 올바르게 설치되어 있는지 확인하세요.
   ```
   pip install 'clip @ git+https://github.com/openai/CLIP.git'
   ```

4. **실험 중단/멈춤**: 처음 모델을 로드할 때 시간이 오래 걸릴 수 있습니다. 특히 transformers 모델(DINO)은 첫 실행 시 다운로드가 필요합니다.

### 디버깅 팁

실행 문제가 발생할 경우 다음과 같이 디버깅할 수 있습니다:

1. 모델 초기화 및 특징 추출 확인:
   ```python
   # scripts/debug_model.py
   from scripts.classifier_cosine import FeatureExtractor
   extractor = FeatureExtractor(model_name="resnet")  # 또는 "clip", "dino"
   print("모델 초기화 성공!")
   ```

2. 지원 세트 로드 확인:
   ```bash
   ls -la data/<category_name>/2.support-set/
   ```

3. 전처리된 이미지 확인:
   ```bash
   ls -la data/<category_name>/6.preprocessed/
   ```

## 참고 사항

- 대규모 데이터셋을 처리하는 경우 충분한 디스크 공간과 메모리가 필요합니다.
- GPU가 있는 경우 자동으로 사용되며, 없는 경우 CPU로 실행됩니다(느릴 수 있음).
- 웹 앱 모드는 실험 실행, 결과 시각화, Ground Truth 레이블링을 위한 통합 인터페이스를 제공합니다. 