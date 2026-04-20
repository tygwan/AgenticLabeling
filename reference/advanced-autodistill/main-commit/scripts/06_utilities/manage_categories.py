#!/usr/bin/env python
"""
카테고리 및 폴더 구조 관리 스크립트

이 스크립트는 카테고리 기반 폴더 구조를 생성하고 관리하는 기능을 제공합니다.
"""

import argparse
import os
import sys
from pathlib import Path
import json
import yaml

# 스크립트가 있는 디렉토리를 시스템 경로에 추가
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir.parent))

# 데이터 유틸리티 모듈 가져오기
from scripts.data_utils import (
    create_category_structure,
    get_all_categories,
    get_category_path,
    get_images_path,
    get_dataset_path,
    create_data_yaml,
    DEFAULT_DATA_DIR
)

def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="카테고리 및 폴더 구조 관리")
    
    # 카테고리 관리 명령
    subparsers = parser.add_subparsers(dest="command", help="명령")
    
    # 카테고리 생성 명령
    create_parser = subparsers.add_parser("create", help="새 카테고리 생성")
    create_parser.add_argument("category_name", help="생성할 카테고리 이름")
    create_parser.add_argument("--base-dir", help="기본 디렉토리 경로 (기본값: 프로젝트 루트/data)")
    
    # 카테고리 목록 명령
    list_parser = subparsers.add_parser("list", help="카테고리 목록 표시")
    list_parser.add_argument("--base-dir", help="기본 디렉토리 경로 (기본값: 프로젝트 루트/data)")
    
    # YOLO 데이터셋 구성 파일 생성 명령
    yaml_parser = subparsers.add_parser("create-yaml", help="YOLO data.yaml 파일 생성")
    yaml_parser.add_argument("category_name", help="대상 카테고리 이름")
    yaml_parser.add_argument("--classes", nargs="+", required=True, help="클래스 이름 목록 (공백으로 구분)")
    yaml_parser.add_argument("--base-dir", help="기본 디렉토리 경로 (기본값: 프로젝트 루트/data)")
    
    # 경로 표시 명령
    path_parser = subparsers.add_parser("get-path", help="카테고리 경로 표시")
    path_parser.add_argument("category_name", help="대상 카테고리 이름")
    path_parser.add_argument("--type", choices=["category", "images", "dataset"],
                             default="category", help="경로 유형 (기본값: category)")
    path_parser.add_argument("--subset", default="train",
                             help="이미지 하위 집합 (images 유형인 경우, 기본값: train)")
    path_parser.add_argument("--base-dir", help="기본 디렉토리 경로 (기본값: 프로젝트 루트/data)")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    # 기본 디렉토리 설정
    base_dir = DEFAULT_DATA_DIR
    if args.base_dir:
        base_dir = Path(args.base_dir)
    
    # 아무 명령도 지정되지 않으면 도움말 표시
    if not args.command:
        print("사용법: python manage_categories.py [명령]")
        print("자세한 사용법은 -h 또는 --help 옵션을 사용하세요.")
        return
    
    # 카테고리 생성
    if args.command == "create":
        category_path = create_category_structure(args.category_name, base_dir)
        print(f"카테고리 '{args.category_name}'이(가) 생성되었습니다.")
        print(f"경로: {category_path}")
    
    # 카테고리 목록 표시
    elif args.command == "list":
        categories = get_all_categories(base_dir)
        if not categories:
            print(f"카테고리가 없습니다. (검색 경로: {base_dir})")
        else:
            print(f"카테고리 목록 ({len(categories)}개):")
            for i, category in enumerate(categories, 1):
                category_path = get_category_path(category, base_dir)
                print(f"{i}. {category} - {category_path}")
    
    # YOLO 데이터셋 구성 파일 생성
    elif args.command == "create-yaml":
        yaml_path = create_data_yaml(args.category_name, args.classes, base_dir)
        print(f"YOLO data.yaml 파일이 생성되었습니다:")
        print(f"경로: {yaml_path}")
        print("클래스:")
        for i, cls in enumerate(args.classes):
            print(f"  {i}: {cls}")
    
    # 경로 표시
    elif args.command == "get-path":
        if args.type == "category":
            path = get_category_path(args.category_name, base_dir)
        elif args.type == "images":
            path = get_images_path(args.category_name, args.subset, base_dir)
        elif args.type == "dataset":
            path = get_dataset_path(args.category_name, base_dir)
        
        print(f"{args.type.capitalize()} 경로 ({args.category_name}):")
        print(path)

if __name__ == "__main__":
    main() 