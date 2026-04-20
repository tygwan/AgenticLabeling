# 사용하지 않는 파일들 목록

## 🗑️ 삭제 권장 파일들

### 디버그/테스트 파일들 (99_deprecated_debug로 이동)

#### 1. 디버그 도구들
- **`debug_few_shot.py`** (12KB, 289 lines)
  - 용도: Few-Shot 분류기 디버깅
  - 상태: 개발 완료 후 불필요
  - 권장 조치: 아카이브 또는 삭제

- **`debug_model.py`** (6.4KB, 158 lines)  
  - 용도: 모델 디버깅 및 검증
  - 상태: 개발 완료 후 불필요
  - 권장 조치: 아카이브 또는 삭제

- **`check_autodistill.py`** (1.3KB)
  - 용도: Autodistill 설치 확인
  - 상태: 간단한 검증용, 한번만 사용
  - 권장 조치: 삭제

#### 2. 테스트 파일들
- **`test_few_shot_classifier.py`** (12KB, 292 lines)
  - 용도: Few-Shot 분류기 단위 테스트
  - 상태: 테스트 완료 후 불필요
  - 권장 조치: 테스트 디렉토리로 이동 또는 삭제

- **`fsl_test.py`** (7.3KB, 187 lines)
  - 용도: Few-Shot Learning 테스트
  - 상태: 테스트 완료 후 불필요
  - 권장 조치: 테스트 디렉토리로 이동 또는 삭제

---

## 🔄 중복/버전 관리 파일들

### YOLO 데이터셋 생성 관련
- **`create_yolo_from_ground_truth.py`** (15KB, 396 lines)
  - 용도: Ground Truth 기반 YOLO 데이터셋 생성 (원본)
  - 상태: `create_yolo_from_ground_truth_fixed.py`가 개선된 버전
  - 권장 조치: fixed 버전 사용 권장, 원본은 아카이브

- **`create_yolo_dataset_corrected.py`** (13KB, 350 lines)
  - 용도: 수정된 YOLO 데이터셋 생성
  - 상태: 최신 버전과 중복 가능성 있음
  - 권장 조치: 기능 확인 후 중복 제거

---

## 📊 프로토타입/실험용 파일들

### 대시보드 관련
- **`dashboard_prototype.py`** (32KB, 1001 lines)
  - 용도: 대시보드 프로토타입
  - 상태: 완성된 웹앱이 있으면 불필요
  - 권장 조치: main_webapp.py 사용 시 삭제

### 분석 도구
- **`analyze_autodistill_accuracy.py`** (17KB, 472 lines)
  - 용도: Autodistill 정확도 분석
  - 상태: 한번 분석 후 재사용 빈도 낮음
  - 권장 조치: 필요 시만 보관

---

## 📄 문서 파일들 (scripts에서 docs로 이동 권장)

### PRD 및 문서들
- **`prd.txt`** (37KB)
  - 용도: 제품 요구사항 문서 (영문)
  - 권장 조치: `docs/` 폴더로 이동

- **`kr-prd.txt`** (35KB)
  - 용도: 제품 요구사항 문서 (한글)
  - 권장 조치: `docs/` 폴더로 이동

- **`example_prd.txt`** (1.5KB)
  - 용도: PRD 예시
  - 권장 조치: `docs/examples/` 폴더로 이동

### 보고서
- **`task-complexity-report.json`** (4.8KB)
  - 용도: 작업 복잡도 분석 보고서
  - 권장 조치: `reports/` 폴더로 이동

---

## 🔧 시스템/유틸리티 (용도 불분명)

### API/서비스 관련
- **`start_api.py`** (3.9KB)
  - 용도: API 서버 시작
  - 상태: 사용 여부 확인 필요
  - 권장 조치: 사용하지 않으면 삭제

- **`start_n8n.py`** (4.7KB)
  - 용도: N8N 워크플로우 시작
  - 상태: 사용 여부 확인 필요
  - 권장 조치: 사용하지 않으면 삭제

- **`cloudflare_tunnel_tracker.py`** (7.5KB)
  - 용도: Cloudflare 터널 추적
  - 상태: 사용 여부 확인 필요
  - 권장 조치: 사용하지 않으면 삭제

- **`update_mcp.py`** (3.0KB)
  - 용도: MCP 업데이트
  - 상태: 사용 여부 확인 필요
  - 권장 조치: 사용하지 않으면 삭제

---

## 🗂️ 기타

### 관리 도구
- **`manage_categories.py`** (4.7KB)
  - 용도: 카테고리 관리
  - 상태: 유틸리티로 보관 권장

- **`material_utils.py`** (5.0KB)
  - 용도: 재료 유틸리티
  - 상태: 사용 여부 확인 필요

### Bash 스크립트들
- **`run_dashboard.sh`** (6.2KB, 202 lines)
  - 용도: 대시보드 실행
  - 상태: dashboard_prototype.py와 연동되어 있다면 함께 정리

---

## 📋 정리 액션 플랜

### 즉시 삭제 권장 (5개)
1. `check_autodistill.py` - 간단한 검증용
2. `debug_few_shot.py` - 개발 완료 후 불필요
3. `debug_model.py` - 개발 완료 후 불필요
4. `test_few_shot_classifier.py` - 테스트 완료 후 불필요
5. `fsl_test.py` - 테스트 완료 후 불필요

### 이동 권장 (4개)
1. `prd.txt` → `docs/`
2. `kr-prd.txt` → `docs/`
3. `example_prd.txt` → `docs/examples/`
4. `task-complexity-report.json` → `reports/`

### 검토 후 처리 (6개)
1. `create_yolo_from_ground_truth.py` - fixed 버전 사용 시 삭제
2. `create_yolo_dataset_corrected.py` - 중복 확인 후 처리
3. `dashboard_prototype.py` - 완성된 웹앱 사용 시 삭제
4. `start_api.py` - 사용 여부 확인
5. `start_n8n.py` - 사용 여부 확인
6. `cloudflare_tunnel_tracker.py` - 사용 여부 확인

### 보관 권장 (3개)
1. `analyze_autodistill_accuracy.py` - 필요 시 사용
2. `manage_categories.py` - 유틸리티로 보관
3. `material_utils.py` - 사용 여부 확인 후 보관

---

## 💡 정리 후 기대 효과

1. **저장 공간 절약**: 약 150KB+ 파일 크기 절약
2. **코드베이스 정리**: 혼란을 줄이고 메인 기능에 집중
3. **유지보수성 향상**: 필요한 파일들만 남겨 관리 효율성 증대
4. **새 사용자 친화적**: 핵심 기능만 남겨 학습 곡선 완화

총 **18개 파일**이 정리 대상이며, 이 중 **9개는 삭제 또는 이동**, **6개는 검토 후 처리**, **3개는 보관** 권장입니다. 