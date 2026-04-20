# Few-Shot Learning Ground Truth Optimization

웹앱의 Ground Truth 레이블링 기능을 최적화하여 수천, 수만 개의 이미지를 효율적으로 처리할 수 있는 개선사항을 구현했습니다.

## 구현된 기능

### 1. 최적화된 Ground Truth Labeler 모듈

새로운 `ground_truth_labeler.py` 모듈을 구현하여 다음과 같은 기능을 제공합니다:

- **페이지네이션**: 대량의 이미지를 페이지 단위로 효율적으로 탐색
- **일괄 레이블링**: 여러 이미지를 한 번에 선택하고 동일한 클래스로 레이블링
- **다양한 필터링 옵션**: 클래스, 신뢰도, 수정 상태에 따라 이미지 필터링
- **Ground Truth 관리**: 특정 실험을 Ground Truth 기준으로 설정하고 저장/로드 기능
- **성능 지표 계산**: 다양한 실험 결과를 Ground Truth와 비교하여 정확도 계산

### 2. 웹앱 통합

기존 `main_webapp.py`에 새로운 "Ground Truth Labeling" 탭을 추가하여 다음과 같은 UI 구성 요소를 제공합니다:

- 실험 선택 및 로드 인터페이스
- 페이지 이동 버튼이 있는 이미지 갤러리
- 이미지 선택 및 일괄 작업 기능
- 이미지 상세 정보 및 레이블 수정 인터페이스
- Ground Truth 관리 및 성능 지표 분석 도구

### 3. 결과 변환 도구

`convert_few_shot_results.py` 스크립트를 구현하여 다양한 형식의 Few-Shot Learning 결과 파일을 Ground Truth Labeler에서 사용할 수 있는 표준화된 형식으로 변환합니다:

- 다양한 실험 방법(method), shot, threshold 조합 지원
- 여러 결과 파일 형식 자동 인식 및 변환
- 표준화된 출력 디렉토리 구조 생성

### 4. 실행 스크립트 및 문서화

사용자 편의성을 위한 추가 파일:

- `run_ground_truth_labeler.sh`: Ground Truth Labeling 시스템을 쉽게 실행할 수 있는 스크립트
- `GROUND_TRUTH_LABELING_GUIDE.md`: 시스템 사용 방법에 대한 상세 가이드

## 주요 개선사항

1. **사용성 향상**:
   - 이전: 한 장씩 이미지 확인 및 클래스 지정
   - 개선: 페이지 단위로 여러 이미지 일괄 처리 가능

2. **클래스 할당 문제 해결**:
   - 이전: "unknown" 클래스에 대해 제한된 클래스 옵션
   - 개선: 모든 이미지에 대해 모든 클래스로 재할당 가능

3. **스크롤 문제 해결**:
   - 이전: 갤러리 스크롤 기능 비정상
   - 개선: 페이지네이션 기반 탐색으로 대체

4. **대량 이미지 처리 효율성**:
   - 이전: 개별 이미지 처리로 인한 낮은 효율성
   - 개선: 일괄 선택 및 처리, 필터링을 통한 효율적인 워크플로우

5. **Ground Truth 관리**:
   - 이전: Ground Truth 설정 및 관리 기능 부재
   - 개선: 특정 실험을 Ground Truth로 지정하고 저장/로드 가능

6. **성능 평가**:
   - 이전: 다양한 shot/threshold 조합 간 비교 어려움
   - 개선: Ground Truth 기반 성능 지표 생성 및 비교 기능

## 사용 방법

1. `run_ground_truth_labeler.sh` 스크립트를 실행하여 웹앱 시작
2. `convert_few_shot_results.py`를 사용하여 결과 파일 변환
3. 웹앱에서 "Ground Truth Labeling" 탭 선택 및 실험 로드
4. 이미지 확인 및 필요한 클래스 재할당
5. Ground Truth로 저장 및 성능 지표 계산

자세한 사용 방법은 `GROUND_TRUTH_LABELING_GUIDE.md` 문서를 참조하세요. 