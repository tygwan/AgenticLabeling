#!/usr/bin/env python3
"""
메타데이터 관리 유틸리티

이 모듈은 전처리 및 분류 결과의 메타데이터를 관리하는 유틸리티 함수를 제공합니다.
"""

import sys
import os
import json
import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union, Tuple, Any

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_metadata_file(metadata_file: str) -> Dict[str, Any]:
    """
    메타데이터 파일 로드
    
    Args:
        metadata_file: 메타데이터 파일 경로
        
    Returns:
        dict: 메타데이터 정보 또는 오류 시 빈 딕셔너리
    """
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"메타데이터 파일 로드 오류: {e}")
        return {}

def get_all_preprocessing_metadata(preprocessed_dir: str) -> Dict[str, Any]:
    """
    전처리 디렉토리의 모든 메타데이터 수집
    
    Args:
        preprocessed_dir: 전처리 결과 디렉토리
        
    Returns:
        dict: 수집된 메타데이터 정보
    """
    result = {
        "summary": None,
        "image_metadata": []
    }
    
    # 요약 메타데이터 파일 확인
    summary_file = os.path.join(preprocessed_dir, "preprocessing_summary.json")
    if os.path.exists(summary_file):
        result["summary"] = load_metadata_file(summary_file)
    
    # 이미지별 메타데이터 파일 수집
    for file in os.listdir(preprocessed_dir):
        if file.endswith("_metadata.json") and file != "preprocessing_summary.json":
            metadata_path = os.path.join(preprocessed_dir, file)
            metadata = load_metadata_file(metadata_path)
            if metadata:
                result["image_metadata"].append(metadata)
    
    return result

def get_objects_by_class(metadata: Dict[str, Any], class_name: str) -> List[Dict[str, Any]]:
    """
    특정 클래스에 속하는 모든 객체 정보 가져오기
    
    Args:
        metadata: 전체 메타데이터 정보
        class_name: 클래스 이름
        
    Returns:
        list: 해당 클래스의 객체 정보 목록
    """
    objects = []
    
    for image_meta in metadata.get("image_metadata", []):
        for obj in image_meta.get("processed_objects", []):
            if obj.get("class_name") == class_name:
                # 원본 이미지 경로 추가
                obj["source_image"] = image_meta.get("source_image", "")
                objects.append(obj)
    
    return objects

def get_classification_structure(base_dir: str, category_name: str) -> Dict[str, Any]:
    """
    분류 구조 정보 가져오기
    
    Args:
        base_dir: 기본 디렉토리 경로
        category_name: 카테고리 이름
        
    Returns:
        dict: 분류 구조 정보
    """
    refine_dir = Path(base_dir) / category_name / "8.refine-dataset"
    metadata_file = refine_dir / "classification_structure.json"
    
    if os.path.exists(metadata_file):
        return load_metadata_file(str(metadata_file))
    else:
        return {}

def create_classification_plan(
    metadata: Dict[str, Any], 
    class_mapping: Dict[str, Dict[str, str]] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    분류 계획 생성
    
    Args:
        metadata: 전체 메타데이터 정보
        class_mapping: 클래스 매핑 정보 (메소드별 클래스 매핑)
        
    Returns:
        dict: 분류 계획 (메소드 -> 클래스 -> 파일 목록)
    """
    # 기본 분류 계획 구조 생성
    plan = {}
    
    # 모든 클래스 및 파일 목록 수집
    all_classes = set()
    class_files = {}
    
    for image_meta in metadata.get("image_metadata", []):
        for obj in image_meta.get("processed_objects", []):
            class_name = obj.get("class_name", "unknown")
            file_path = obj.get("output_file", "")
            
            if file_path:
                all_classes.add(class_name)
                if class_name not in class_files:
                    class_files[class_name] = []
                class_files[class_name].append(file_path)
    
    # 기본 매핑 생성 (직접 매핑)
    if class_mapping is None:
        class_mapping = {
            "method1": {cls: cls for cls in all_classes},
            "method2": {cls: cls for cls in all_classes},
            "method3": {cls: cls for cls in all_classes},
            "method4": {cls: cls for cls in all_classes}
        }
    
    # 각 메소드별 분류 계획 생성
    for method, mapping in class_mapping.items():
        plan[method] = {}
        
        for src_class, dest_class in mapping.items():
            if src_class in class_files:
                if dest_class not in plan[method]:
                    plan[method][dest_class] = []
                plan[method][dest_class].extend(class_files[src_class])
    
    return plan

def execute_classification_plan(
    plan: Dict[str, Dict[str, List[str]]], 
    base_dir: str, 
    category_name: str
) -> Dict[str, int]:
    """
    분류 계획 실행 (파일 복사)
    
    Args:
        plan: 분류 계획
        base_dir: 기본 디렉토리 경로
        category_name: 카테고리 이름
        
    Returns:
        dict: 분류 결과 통계
    """
    refine_dir = Path(base_dir) / category_name / "8.refine-dataset"
    stats = {}
    
    for method, classes in plan.items():
        method_dir = refine_dir / method
        method_dir.mkdir(exist_ok=True, parents=True)
        
        stats[method] = 0
        
        for class_name, files in classes.items():
            class_dir = method_dir / class_name
            class_dir.mkdir(exist_ok=True, parents=True)
            
            for file in files:
                if os.path.exists(file):
                    dest_file = class_dir / os.path.basename(file)
                    shutil.copy2(file, dest_file)
                    stats[method] += 1
    
    # 분류 실행 결과 메타데이터 저장
    result_metadata = {
        "executed_at": datetime.datetime.now().isoformat(),
        "statistics": stats,
        "plan_summary": {
            method: {
                class_name: len(files) 
                for class_name, files in classes.items()
            } 
            for method, classes in plan.items()
        }
    }
    
    metadata_file = refine_dir / "classification_result.json"
    with open(metadata_file, 'w') as f:
        json.dump(result_metadata, f, indent=2)
    
    return stats

# 테스트 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="메타데이터 관리 유틸리티")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 디렉토리 경로")
    parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름")
    parser.add_argument("--summary", action="store_true", help="전처리 요약 정보 출력")
    parser.add_argument("--stats", action="store_true", help="클래스별 통계 출력")
    parser.add_argument("--class-name", type=str, help="특정 클래스 객체 정보 출력")
    
    args = parser.parse_args()
    
    # 경로 설정
    data_dir = Path(args.data_dir)
    preprocessed_dir = data_dir / args.category / "6.preprocessed"
    
    # 메타데이터 로드
    metadata = get_all_preprocessing_metadata(str(preprocessed_dir))
    
    # 요약 정보 출력
    if args.summary and metadata["summary"]:
        print("\n=== 전처리 요약 정보 ===")
        summary = metadata["summary"]
        print(f"총 이미지 수: {summary.get('total_images', 0)}")
        print(f"총 객체 수: {summary.get('total_objects', 0)}")
        print(f"처리 시간: {summary.get('processed_at', '')}")
        
        if "processing_options" in summary:
            print("\n처리 옵션:")
            for option, value in summary["processing_options"].items():
                print(f"  - {option}: {value}")
    
    # 클래스별 통계 출력
    if args.stats:
        print("\n=== 클래스별 통계 ===")
        class_counts = {}
        
        for image_meta in metadata.get("image_metadata", []):
            for obj in image_meta.get("processed_objects", []):
                class_name = obj.get("class_name", "unknown")
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  - {class_name}: {count}개 객체")
    
    # 특정 클래스 객체 정보 출력
    if args.class_name:
        print(f"\n=== '{args.class_name}' 클래스 객체 정보 ===")
        objects = get_objects_by_class(metadata, args.class_name)
        
        print(f"총 {len(objects)}개 객체 발견")
        for i, obj in enumerate(objects[:5], 1):  # 처음 5개만 출력
            print(f"\n객체 {i}:")
            print(f"  ID: {obj.get('id', '')}")
            print(f"  파일: {obj.get('output_file', '')}")
            print(f"  원본 이미지: {obj.get('source_image', '')}")
            print(f"  경계 상자: {obj.get('original_box', [])}")
            print(f"  출력 크기: {obj.get('output_size', [])}")
        
        if len(objects) > 5:
            print(f"\n...외 {len(objects) - 5}개 객체")
    
    # 분류 구조 확인
    classification_structure = get_classification_structure(str(data_dir), args.category)
    if classification_structure:
        print("\n=== 분류 구조 정보 ===")
        print(f"카테고리: {classification_structure.get('category', '')}")
        print("분류 방법:")
        for method in classification_structure.get("classification_methods", []):
            print(f"  - {method}") 