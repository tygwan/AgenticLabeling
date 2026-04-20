#!/usr/bin/env python3
"""
이미지 분류 스크립트

전처리된 이미지를 분류 구조에 맞게 분류합니다.
Task 4에서 4가지 분류 방법을 구현할 때 기반으로 활용할 수 있습니다.
"""

import os
import argparse
import json
from pathlib import Path
import shutil
import sys

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 메타데이터 유틸리티 가져오기
from scripts.metadata_utils import (
    get_all_preprocessing_metadata,
    get_classification_structure,
    create_classification_plan,
    execute_classification_plan
)

def setup_argparse():
    """명령줄 인자 파서 설정"""
    parser = argparse.ArgumentParser(description="이미지 분류 스크립트")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 디렉토리 경로")
    parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름")
    parser.add_argument("--method", type=str, help="실행할 특정 분류 방법 (생략 시 모든 방법 실행)")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일 복사 없이 계획만 생성")
    parser.add_argument("--mapping", type=str, help="클래스 매핑 JSON 파일 경로")
    return parser.parse_args()

def load_class_mapping(mapping_file):
    """클래스 매핑 파일 로드"""
    try:
        with open(mapping_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"클래스 매핑 파일 로드 오류: {e}")
        return None

def print_classification_plan(plan):
    """분류 계획 출력"""
    print("\n=== 분류 계획 ===")
    for method, classes in plan.items():
        total_files = sum(len(files) for files in classes.values())
        print(f"\n[{method}] - 총 {total_files}개 파일")
        
        for class_name, files in classes.items():
            print(f"  - {class_name}: {len(files)}개 파일")
            if len(files) > 0:
                print(f"    예: {os.path.basename(files[0])}")

def main():
    """메인 함수"""
    args = setup_argparse()
    
    # 경로 설정
    data_dir = Path(args.data_dir)
    category = args.category
    preprocessed_dir = data_dir / category / "6.preprocessed"
    
    print(f"데이터 디렉토리: {data_dir}")
    print(f"카테고리: {category}")
    print(f"전처리 디렉토리: {preprocessed_dir}")
    
    # 메타데이터 로드
    print("\n메타데이터 로드 중...")
    metadata = get_all_preprocessing_metadata(str(preprocessed_dir))
    
    if not metadata.get("image_metadata"):
        print("오류: 전처리된 이미지 메타데이터를 찾을 수 없습니다.")
        return
    
    # 분류 구조 확인
    classification_structure = get_classification_structure(str(data_dir), category)
    if not classification_structure:
        print("오류: 분류 구조를 찾을 수 없습니다. 먼저 --prepare-classify 옵션으로 분류 구조를 생성하세요.")
        return
    
    # 분류 방법 설정
    methods = [args.method] if args.method else classification_structure.get("classification_methods", [])
    print(f"\n사용할 분류 방법: {', '.join(methods)}")
    
    # 클래스 매핑 로드 (있는 경우)
    class_mapping = None
    if args.mapping:
        class_mapping = load_class_mapping(args.mapping)
        if not class_mapping:
            print("오류: 클래스 매핑 파일을 로드할 수 없습니다.")
            return
        print(f"클래스 매핑 파일 로드 완료: {args.mapping}")
    
    # 분류 계획 생성
    print("\n분류 계획 생성 중...")
    full_plan = create_classification_plan(metadata, class_mapping)
    
    # 지정된 방법만 사용하도록 필터링
    plan = {method: full_plan.get(method, {}) for method in methods if method in full_plan}
    
    # 분류 계획 출력
    print_classification_plan(plan)
    
    # 드라이 런인 경우 여기서 종료
    if args.dry_run:
        print("\n드라이 런 모드: 실제 파일 복사는 수행되지 않았습니다.")
        return
    
    # 분류 실행
    print("\n분류 실행 중...")
    stats = execute_classification_plan(plan, str(data_dir), category)
    
    # 결과 출력
    print("\n=== 분류 결과 ===")
    for method, count in stats.items():
        print(f"{method}: {count}개 파일 복사됨")
    
    print(f"\n분류 결과는 {data_dir}/{category}/8.refine-dataset/ 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 