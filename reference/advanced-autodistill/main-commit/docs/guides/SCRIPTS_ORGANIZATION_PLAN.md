# Scripts 폴더 정리 계획

## 📁 Phase별 폴더 구조

### 01_data_preparation (데이터 준비 및 초기 처리)
**핵심 기능**: 원본 이미지 처리, 객체 탐지, 마스크 생성

#### 메인 스크립트
- `main_launcher.py` - 메인 파이프라인 런처
- `autodistill_runner.py` - Autodistill 실행기
- `advanced_preprocessor.py` - 고급 전처리기

#### 유틸리티 
- `custom_helpers.py` - 모델 로딩 및 패치 헬퍼
- `data_utils.py` - 데이터 관리 유틸리티
- `preprocess_utils.py` - 이미지 전처리 유틸리티
- `mask_utils.py` - 마스크 처리 유틸리티
- `metadata_utils.py` - 메타데이터 처리

#### 변환 도구
- `mask_converter.py` - 마스크 형식 변환
- `data_converter.py` - 데이터 형식 변환
- `show_mask_info.py` - 마스크 정보 시각화

---

### 02_preprocessing (데이터 전처리 및 구조화)
**핵심 기능**: Support Set 구조화, 데이터 정제

#### 메인 스크립트
- `restructure_support_set.py` - Support Set 구조화
- `support_set_manager.py` - Support Set 관리
- `run_support_set_manager.sh` - Support Set 관리 스크립트

#### 전처리 도구
- `autodistill_dataset_resizer.py` - 데이터셋 크기 조정
- `high_resolution_converter.py` - 고해상도 변환
- `refine_dataset.py` - 데이터셋 정제

---

### 03_classification (Few-Shot Learning 및 분류)
**핵심 기능**: Few-Shot 분류, 실험 수행, 성능 분석

#### 분류기 및 실험
- `classifier_cosine.py` - 코사인 유사도 분류기
- `classifier_cosine_experiment.py` - 코사인 분류 실험
- `classifier_vlm.py` - VLM 분류기
- `run_shot_threshold_experiments.py` - Shot/Threshold 실험

#### 웹 인터페이스
- `run_few_shot_platform.py` - Few-Shot 플랫폼 런처
- `few_shot_webapp.py` - Few-Shot 웹앱
- `main_webapp.py` - 메인 웹앱

#### 분석 및 비교
- `analyze_experiment_metrics.py` - 실험 결과 분석
- `run_model_comparison.py` - 모델 성능 비교
- `run_classifier_comparison.sh` - 분류기 비교 스크립트
- `convert_few_shot_results.py` - Few-Shot 결과 변환

#### 스크립트
- `start_classification.py` - 분류 시작
- `run_full_analysis.sh` - 전체 분석 스크립트

---

### 04_ground_truth (Ground Truth 생성 및 관리)
**핵심 기능**: 라벨링, Ground Truth 생성, 품질 평가

#### 라벨링 도구
- `ground_truth_labeler.py` - Ground Truth 라벨링 도구
- `folder_based_labeler.py` - 폴더 기반 라벨링
- `run_ground_truth_labeler.sh` - 라벨링 도구 실행 스크립트

#### 평가 및 검증
- `evaluate_ground_truth.py` - Ground Truth 평가
- `run_ground_truth_evaluator.sh` - 평가 도구 실행 스크립트

#### 결과 정리
- `organize_classification_results.py` - 분류 결과 정리
- `run_organize_results.sh` - 결과 정리 스크립트

#### 정확도 분석
- `analyze_autodistill_accuracy.py` - Autodistill 정확도 분석

---

### 05_yolo_training (YOLO 학습 및 데이터셋 생성)
**핵심 기능**: YOLO 데이터셋 생성, 모델 학습

#### 데이터셋 생성
- `create_yolo_segmentation_dataset.py` - YOLO 세그먼테이션 데이터셋 생성
- `create_yolo_from_ground_truth_fixed.py` - Ground Truth 기반 YOLO 데이터셋 (수정됨)
- `create_yolo_from_ground_truth.py` - Ground Truth 기반 YOLO 데이터셋 (원본)
- `create_yolo_dataset_corrected.py` - 수정된 YOLO 데이터셋

#### 모델 학습
- `train_yolo_segmentation.py` - YOLO 세그먼테이션 학습

---

### 06_utilities (공통 유틸리티 및 도구)
**핵심 기능**: 프로젝트 관리, 시스템 도구

#### 프로젝트 관리
- `manage_categories.py` - 카테고리 관리
- `material_utils.py` - 재료 유틸리티
- `example_class_mapping.json` - 예시 클래스 매핑

#### 시스템 도구
- `start_api.py` - API 시작
- `start_n8n.py` - N8N 시작
- `cloudflare_tunnel_tracker.py` - Cloudflare 터널 추적
- `update_mcp.py` - MCP 업데이트

#### 대시보드 및 모니터링
- `dashboard_prototype.py` - 대시보드 프로토타입
- `run_dashboard.sh` - 대시보드 실행 스크립트

#### 문서 및 설정
- `prd.txt` - 제품 요구사항 문서 (영문)
- `kr-prd.txt` - 제품 요구사항 문서 (한글)
- `example_prd.txt` - 예시 PRD
- `task-complexity-report.json` - 작업 복잡도 보고서

---

### 99_deprecated_debug (사용하지 않는 파일들)
**핵심 기능**: 디버그, 테스트, 사용하지 않는 코드

#### 디버그 도구
- `debug_few_shot.py` - Few-Shot 디버그 도구
- `debug_model.py` - 모델 디버그 도구
- `check_autodistill.py` - Autodistill 체크

#### 테스트 파일
- `test_few_shot_classifier.py` - Few-Shot 분류기 테스트
- `fsl_test.py` - FSL 테스트

---

## 📋 파일 이동 계획

### Phase 1: 01_data_preparation
```bash
mv main_launcher.py scripts/01_data_preparation/
mv autodistill_runner.py scripts/01_data_preparation/
mv advanced_preprocessor.py scripts/01_data_preparation/
mv custom_helpers.py scripts/01_data_preparation/
mv data_utils.py scripts/01_data_preparation/
mv preprocess_utils.py scripts/01_data_preparation/
mv mask_utils.py scripts/01_data_preparation/
mv metadata_utils.py scripts/01_data_preparation/
mv mask_converter.py scripts/01_data_preparation/
mv data_converter.py scripts/01_data_preparation/
mv show_mask_info.py scripts/01_data_preparation/
```

### Phase 2: 02_preprocessing
```bash
mv restructure_support_set.py scripts/02_preprocessing/
mv support_set_manager.py scripts/02_preprocessing/
mv run_support_set_manager.sh scripts/02_preprocessing/
mv autodistill_dataset_resizer.py scripts/02_preprocessing/
mv high_resolution_converter.py scripts/02_preprocessing/
mv refine_dataset.py scripts/02_preprocessing/
```

### Phase 3: 03_classification
```bash
mv classifier_cosine.py scripts/03_classification/
mv classifier_cosine_experiment.py scripts/03_classification/
mv classifier_vlm.py scripts/03_classification/
mv run_shot_threshold_experiments.py scripts/03_classification/
mv run_few_shot_platform.py scripts/03_classification/
mv few_shot_webapp.py scripts/03_classification/
mv main_webapp.py scripts/03_classification/
mv analyze_experiment_metrics.py scripts/03_classification/
mv run_model_comparison.py scripts/03_classification/
mv run_classifier_comparison.sh scripts/03_classification/
mv convert_few_shot_results.py scripts/03_classification/
mv start_classification.py scripts/03_classification/
mv run_full_analysis.sh scripts/03_classification/
```

### Phase 4: 04_ground_truth
```bash
mv ground_truth_labeler.py scripts/04_ground_truth/
mv folder_based_labeler.py scripts/04_ground_truth/
mv run_ground_truth_labeler.sh scripts/04_ground_truth/
mv evaluate_ground_truth.py scripts/04_ground_truth/
mv run_ground_truth_evaluator.sh scripts/04_ground_truth/
mv organize_classification_results.py scripts/04_ground_truth/
mv run_organize_results.sh scripts/04_ground_truth/
mv analyze_autodistill_accuracy.py scripts/04_ground_truth/
```

### Phase 5: 05_yolo_training
```bash
mv create_yolo_segmentation_dataset.py scripts/05_yolo_training/
mv create_yolo_from_ground_truth_fixed.py scripts/05_yolo_training/
mv create_yolo_from_ground_truth.py scripts/05_yolo_training/
mv create_yolo_dataset_corrected.py scripts/05_yolo_training/
mv train_yolo_segmentation.py scripts/05_yolo_training/
```

### Phase 6: 06_utilities
```bash
mv manage_categories.py scripts/06_utilities/
mv material_utils.py scripts/06_utilities/
mv example_class_mapping.json scripts/06_utilities/
mv start_api.py scripts/06_utilities/
mv start_n8n.py scripts/06_utilities/
mv cloudflare_tunnel_tracker.py scripts/06_utilities/
mv update_mcp.py scripts/06_utilities/
mv dashboard_prototype.py scripts/06_utilities/
mv run_dashboard.sh scripts/06_utilities/
mv prd.txt scripts/06_utilities/
mv kr-prd.txt scripts/06_utilities/
mv example_prd.txt scripts/06_utilities/
mv task-complexity-report.json scripts/06_utilities/
```

### Phase 7: 99_deprecated_debug
```bash
mv debug_few_shot.py scripts/99_deprecated_debug/
mv debug_model.py scripts/99_deprecated_debug/
mv check_autodistill.py scripts/99_deprecated_debug/
mv test_few_shot_classifier.py scripts/99_deprecated_debug/
mv fsl_test.py scripts/99_deprecated_debug/
```

---

## 🗑️ 사용하지 않는 파일들 (정리 대상)

### 디버그/테스트 파일들
- `debug_few_shot.py` - Few-Shot 디버그 (개발 완료 후 불필요)
- `debug_model.py` - 모델 디버그 (개발 완료 후 불필요)  
- `test_few_shot_classifier.py` - 분류기 테스트 (단위 테스트용)
- `fsl_test.py` - FSL 테스트 (단위 테스트용)
- `check_autodistill.py` - Autodistill 체크 (간단한 검증용)

### 중복/버전 관리 파일들
- `create_yolo_from_ground_truth.py` - 원본 버전 (fixed 버전 사용 권장)
- `create_yolo_dataset_corrected.py` - 수정된 버전 (최신 버전과 중복 가능성)

### 프로토타입/실험용 파일들
- `dashboard_prototype.py` - 대시보드 프로토타입 (완성된 웹앱이 있으면 불필요)

### 문서 파일들 (scripts에서 docs로 이동 권장)
- `prd.txt` - 문서 폴더로 이동
- `kr-prd.txt` - 문서 폴더로 이동  
- `example_prd.txt` - 문서 폴더로 이동
- `task-complexity-report.json` - 보고서 폴더로 이동

---

## ✅ 정리 후 기대 효과

1. **명확한 워크플로우**: Phase별로 구분되어 사용자가 단계별로 이해하기 쉬움
2. **유지보수성 향상**: 관련 기능별로 그룹화되어 코드 관리 용이
3. **중복 제거**: 사용하지 않는 파일들을 별도 폴더로 분리
4. **문서화 개선**: 각 Phase별 README 파일 추가 가능
5. **새 사용자 친화적**: 워크플로우 가이드와 연계된 폴더 구조 