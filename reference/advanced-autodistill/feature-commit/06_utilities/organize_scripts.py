#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts 폴더 정리 스크립트

워크플로우에 따라 scripts 폴더의 파일들을 정리하고 재구성합니다.
"""

import os
import shutil
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Phase별 폴더 구조 정의
FOLDER_STRUCTURE = {
    "01_data_preparation": [
        "main_launcher.py",
        "autodistill_runner.py", 
        "advanced_preprocessor.py",
        "custom_helpers.py",
        "data_utils.py",
        "preprocess_utils.py",
        "mask_utils.py",
        "metadata_utils.py",
        "mask_converter.py",
        "data_converter.py",
        "show_mask_info.py"
    ],
    "02_preprocessing": [
        "restructure_support_set.py",
        "support_set_manager.py",
        "run_support_set_manager.sh",
        "autodistill_dataset_resizer.py",
        "high_resolution_converter.py",
        "refine_dataset.py"
    ],
    "03_classification": [
        "classifier_cosine.py",
        "classifier_cosine_experiment.py",
        "classifier_vlm.py",
        "run_shot_threshold_experiments.py",
        "run_few_shot_platform.py",
        "few_shot_webapp.py",
        "main_webapp.py",
        "analyze_experiment_metrics.py",
        "run_model_comparison.py",
        "run_classifier_comparison.sh",
        "convert_few_shot_results.py",
        "start_classification.py",
        "run_full_analysis.sh"
    ],
    "04_ground_truth": [
        "ground_truth_labeler.py",
        "folder_based_labeler.py",
        "run_ground_truth_labeler.sh",
        "evaluate_ground_truth.py",
        "run_ground_truth_evaluator.sh",
        "organize_classification_results.py",
        "run_organize_results.sh",
        "analyze_autodistill_accuracy.py"
    ],
    "05_yolo_training": [
        "create_yolo_segmentation_dataset.py",
        "create_yolo_from_ground_truth_fixed.py",
        "create_yolo_from_ground_truth.py",
        "create_yolo_dataset_corrected.py",
        "train_yolo_segmentation.py"
    ],
    "06_utilities": [
        "manage_categories.py",
        "material_utils.py",
        "example_class_mapping.json",
        "start_api.py",
        "start_n8n.py",
        "cloudflare_tunnel_tracker.py",
        "update_mcp.py",
        "dashboard_prototype.py",
        "run_dashboard.sh",
        "prd.txt",
        "kr-prd.txt",
        "example_prd.txt",
        "task-complexity-report.json"
    ],
    "99_deprecated_debug": [
        "debug_few_shot.py",
        "debug_model.py",
        "check_autodistill.py",
        "test_few_shot_classifier.py",
        "fsl_test.py"
    ]
}

def create_folder_structure():
    """Phase별 폴더 구조 생성"""
    print("📁 폴더 구조 생성 중...")
    
    for folder_name in FOLDER_STRUCTURE.keys():
        folder_path = SCRIPTS_DIR / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"   ✓ {folder_name}")

def move_files():
    """파일들을 적절한 폴더로 이동"""
    print("\n📦 파일 이동 중...")
    
    moved_files = []
    missing_files = []
    
    for folder_name, files in FOLDER_STRUCTURE.items():
        folder_path = SCRIPTS_DIR / folder_name
        
        print(f"\n{folder_name}:")
        for file_name in files:
            source_path = SCRIPTS_DIR / file_name
            dest_path = folder_path / file_name
            
            if source_path.exists():
                try:
                    shutil.move(str(source_path), str(dest_path))
                    print(f"   ✓ {file_name}")
                    moved_files.append(file_name)
                except Exception as e:
                    print(f"   ❌ {file_name} (오류: {e})")
            else:
                print(f"   ⚠️  {file_name} (파일 없음)")
                missing_files.append(file_name)
    
    return moved_files, missing_files

def create_readme_files():
    """각 폴더에 README 파일 생성"""
    print("\n📄 README 파일 생성 중...")
    
    readme_contents = {
        "01_data_preparation": """# Phase 1: 데이터 준비 및 초기 처리

## 개요
원본 이미지에서 객체를 탐지하고 마스크를 생성하는 초기 데이터 처리 단계입니다.

## 주요 스크립트
- `main_launcher.py`: 메인 파이프라인 런처
- `autodistill_runner.py`: Autodistill + SAM2 실행기
- `advanced_preprocessor.py`: 고급 이미지 전처리

## 실행 방법
```bash
python main_launcher.py --category test_category --plot --preprocess
```
""",
        "02_preprocessing": """# Phase 2: 데이터 전처리 및 구조화

## 개요
Support Set을 구조화하고 데이터를 정제하는 단계입니다.

## 주요 스크립트
- `restructure_support_set.py`: Support Set N-shot별 구조화
- `support_set_manager.py`: Support Set 관리
- `refine_dataset.py`: 데이터셋 정제

## 실행 방법
```bash
python restructure_support_set.py --category test_category --shots 1,5,10,30
```
""",
        "03_classification": """# Phase 3: Few-Shot Learning 및 분류

## 개요
Few-Shot Learning을 통한 분류 및 성능 분석 단계입니다.

## 주요 스크립트
- `run_few_shot_platform.py`: Few-Shot 웹 플랫폼
- `run_shot_threshold_experiments.py`: 배치 실험 수행
- `analyze_experiment_metrics.py`: 결과 분석

## 실행 방법
```bash
# 웹 인터페이스
python run_few_shot_platform.py --webapp

# CLI 배치 실험
python run_shot_threshold_experiments.py --category test_category --models resnet,dino
```
""",
        "04_ground_truth": """# Phase 4: Ground Truth 생성 및 관리

## 개요
분류 결과를 검토하고 Ground Truth를 생성하는 단계입니다.

## 주요 스크립트
- `ground_truth_labeler.py`: 대화형 라벨링 도구
- `evaluate_ground_truth.py`: Ground Truth 품질 평가
- `organize_classification_results.py`: 결과 정리

## 실행 방법
```bash
python ground_truth_labeler.py --category test_category
```
""",
        "05_yolo_training": """# Phase 5: YOLO 학습 및 데이터셋 생성

## 개요
YOLO 세그먼테이션 데이터셋을 생성하고 모델을 학습하는 단계입니다.

## 주요 스크립트
- `create_yolo_segmentation_dataset.py`: YOLO 데이터셋 생성
- `train_yolo_segmentation.py`: YOLO 모델 학습

## 실행 방법
```bash
# 데이터셋 생성
python create_yolo_segmentation_dataset.py --category test_category --output data/test_category/8.yolo-dataset

# 모델 학습
python train_yolo_segmentation.py --data data/test_category/8.yolo-dataset/dataset.yaml --epochs 100 --copy-paste 0.3
```
""",
        "06_utilities": """# Phase 6: 공통 유틸리티 및 도구

## 개요
프로젝트 관리, 시스템 도구, 문서 등을 포함하는 폴더입니다.

## 주요 스크립트
- `manage_categories.py`: 카테고리 관리
- `dashboard_prototype.py`: 대시보드 프로토타입
- `start_api.py`: API 서버 시작

## 설정 파일
- `example_class_mapping.json`: 클래스 매핑 예시
- `prd.txt`: 제품 요구사항 문서
""",
        "99_deprecated_debug": """# Phase 99: 사용하지 않는 파일들

## 개요
디버그, 테스트, 사용하지 않는 코드들을 모아둔 폴더입니다.

## 포함된 파일들
- 디버그 도구들
- 단위 테스트 파일들
- 프로토타입 코드들

⚠️ **주의**: 이 폴더의 파일들은 개발 완료 후 삭제하거나 별도 관리할 수 있습니다.
"""
    }
    
    for folder_name, content in readme_contents.items():
        readme_path = SCRIPTS_DIR / folder_name / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ {folder_name}/README.md")

def generate_summary():
    """정리 결과 요약 생성"""
    summary_content = """# Scripts 폴더 정리 완료 보고서

## 📁 새로운 폴더 구조

```
scripts/
├── 01_data_preparation/     # 데이터 준비 및 초기 처리
├── 02_preprocessing/        # 데이터 전처리 및 구조화  
├── 03_classification/       # Few-Shot Learning 및 분류
├── 04_ground_truth/         # Ground Truth 생성 및 관리
├── 05_yolo_training/        # YOLO 학습 및 데이터셋 생성
├── 06_utilities/            # 공통 유틸리티 및 도구
└── 99_deprecated_debug/     # 사용하지 않는 파일들
```

## 🚀 워크플로우별 실행 가이드

### Phase 1: 데이터 준비
```bash
python scripts/01_data_preparation/main_launcher.py --category test_category --plot --preprocess
```

### Phase 2: 전처리
```bash
python scripts/02_preprocessing/restructure_support_set.py --category test_category --shots 1,5,10,30
```

### Phase 3: 분류
```bash
python scripts/03_classification/run_few_shot_platform.py --webapp
```

### Phase 4: Ground Truth
```bash
python scripts/04_ground_truth/ground_truth_labeler.py --category test_category
```

### Phase 5: YOLO 학습
```bash
python scripts/05_yolo_training/train_yolo_segmentation.py --data data/test_category/8.yolo-dataset/dataset.yaml --epochs 100 --copy-paste 0.3
```

## 📋 정리 완료 항목

✅ 워크플로우에 따른 폴더 구조 생성
✅ 파일들을 적절한 폴더로 이동
✅ 각 폴더별 README.md 생성
✅ 사용하지 않는 파일들 분리

## 🎯 다음 단계

1. 각 Phase별 README.md 내용 보완
2. 상대 경로 import 문제 해결
3. 사용하지 않는 파일들 검토 후 삭제
4. 워크플로우 가이드 업데이트
"""
    
    summary_path = PROJECT_ROOT / "SCRIPTS_REORGANIZATION_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    print(f"   ✓ SCRIPTS_REORGANIZATION_SUMMARY.md")

def main():
    """메인 실행 함수"""
    print("🚀 Scripts 폴더 정리를 시작합니다...\n")
    
    # 1. 폴더 구조 생성
    create_folder_structure()
    
    # 2. 파일 이동
    moved_files, missing_files = move_files()
    
    # 3. README 파일 생성
    create_readme_files()
    
    # 4. 요약 보고서 생성
    generate_summary()
    
    # 5. 결과 출력
    print(f"\n✅ 정리 완료!")
    print(f"   - 이동된 파일: {len(moved_files)}개")
    print(f"   - 누락된 파일: {len(missing_files)}개")
    
    if missing_files:
        print(f"\n⚠️ 누락된 파일들:")
        for file in missing_files:
            print(f"   - {file}")
    
    print(f"\n📖 상세 내용은 SCRIPTS_REORGANIZATION_SUMMARY.md를 참조하세요.")

if __name__ == "__main__":
    main() 