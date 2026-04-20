# 프로젝트 설정 및 기능 요약

## 구현 완료된 기능

1. **경로 구조 표준화**
   - `data/{category}/` 아래 순차적으로 번호가 매겨진 폴더 구조 구현
   - 각 단계별 데이터 저장 위치 명확화: 이미지, 지원 세트, 박스, 마스크, 데이터셋, 전처리, 결과, 정제 데이터세트

2. **마스크 처리 향상**
   - NPY 바이너리 배열 외에 JSON/TXT 좌표 형식으로 마스크 데이터 저장
   - 윤곽선 및 경계 상자 추출 기능 구현
   - 마스크 없이 경계 상자 데이터만으로도 작업 가능
   - 시각화 오류 해결

3. **코드 개선**
   - tqdm을 사용한 진행 상황 표시 기능 추가
   - 레벨별 로깅(INFO, DEBUG, WARNING) 구현으로 터미널 출력 최적화
   - 이미지 유효성 검사 개선으로 시각화 오류 해결
   - NPY 마스크 저장을 선택적으로 설정 가능

4. **전처리 기능**
   - 좌표 기반 이미지 처리 기능 구현
   - 경계 상자 기반 크롭 기능
   - 윤곽선 정보를 이용한 마스크 적용
   - 기존 파이프라인과의 호환성 유지

5. **명령줄 인터페이스 개선**
   - `--preprocess` 옵션 추가로 전처리 기능 실행 가능
   - `--plot` 옵션을 통한 시각화 제어
   - `--target-size`, `--no-crop`, `--no-mask` 등 전처리 세부 조정 옵션 추가
   - `--save-npy` 옵션으로 NPY 파일 저장 여부 선택 가능

## 사용 방법

### 기본 실행 (탐지만)
```bash
python scripts/main_launcher.py --category test_category
```

### 탐지 + 시각화
```bash
python scripts/main_launcher.py --category test_category --plot
```

### 탐지 + 전처리
```bash
python scripts/main_launcher.py --category test_category --preprocess
```

### 전체 파이프라인 실행 (탐지 + 시각화 + 전처리)
```bash
python scripts/main_launcher.py --category test_category --plot --preprocess
```

### 전처리 옵션 조정
```bash
python scripts/main_launcher.py --category test_category --preprocess --target-size 800,800 --no-mask
```

## 다음 단계 제안

1. **데이터 증강 및 정제**
   - 전처리된 이미지에 대한 데이터 증강 기능 추가
   - 정제된 데이터셋 생성 자동화

2. **모델 훈련 통합**
   - 전처리된 데이터를 이용한 YOLOv8 훈련 파이프라인 통합
   - 훈련 매개변수 조정 및 최적화 기능

3. **웹 인터페이스**
   - 전체 파이프라인을 위한 간단한 웹 UI 개발
   - 결과 시각화 및 분석 도구 통합

4. **배치 처리 최적화**
   - 대규모 데이터셋 처리를 위한 멀티프로세싱 지원
   - 증분 처리 기능 추가 (이미 처리된 파일 건너뛰기)

5. **오류 처리 및 복구**
   - 더 강력한 예외 처리 및 복구 메커니즘 구현
   - 처리 로그 및 오류 보고 기능 향상 