#!/usr/bin/env python
"""
Material 폴더 관리 유틸리티 스크립트

이 스크립트는 Material 폴더에서 데이터 폴더로 이미지를 복사하고 관리하는 기능을 제공합니다.
"""

import argparse
import os
import sys
from pathlib import Path
import json

# 스크립트가 있는 디렉토리를 시스템 경로에 추가
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir.parent))

# 데이터 유틸리티 모듈 가져오기
from scripts.data_utils import (
    copy_material_to_data,
    get_material_categories,
    get_material_category_path,
    convert_autodistill_to_yolo
)

def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Material 폴더 관리 유틸리티")
    
    # 명령 하위 파서
    subparsers = parser.add_subparsers(dest="command", help="명령")
    
    # 카테고리 목록 명령
    list_parser = subparsers.add_parser("list", help="Material 폴더 내 카테고리 목록 표시")
    
    # 이미지 복사 명령
    copy_parser = subparsers.add_parser("copy", help="Material 폴더에서 데이터 폴더로 이미지 복사")
    copy_parser.add_argument("category", help="복사할 카테고리")
    
    # YOLO 어노테이션 변환 명령
    convert_parser = subparsers.add_parser("convert", help="Autodistill 어노테이션을 YOLO 형식으로 변환")
    convert_parser.add_argument("category", help="변환할 카테고리")
    convert_parser.add_argument("annotation_file", help="Autodistill 어노테이션 JSON 파일 경로")
    convert_parser.add_argument("--class-mapping", help="클래스 이름에서 ID로의 매핑 JSON 파일 경로")
    convert_parser.add_argument("--img-width", type=int, default=640, help="이미지 너비 (기본값: 640)")
    convert_parser.add_argument("--img-height", type=int, default=640, help="이미지 높이 (기본값: 640)")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    # 아무 명령도 지정되지 않으면 도움말 표시
    if not args.command:
        print("사용법: python material_utils.py [명령]")
        print("자세한 사용법은 -h 또는 --help 옵션을 사용하세요.")
        return
    
    # Material 카테고리 목록 표시
    if args.command == "list":
        categories = get_material_categories()
        if not categories:
            print("Material 폴더에 카테고리가 없습니다.")
        else:
            print(f"Material 폴더 내 카테고리 목록 ({len(categories)}개):")
            for i, category in enumerate(categories, 1):
                category_path = get_material_category_path(category)
                print(f"{i}. {category} - {category_path}")
    
    # Material 폴더에서 데이터 폴더로 이미지 복사
    elif args.command == "copy":
        count, files = copy_material_to_data(args.category)
        if count == 0:
            print(f"복사된 파일이 없습니다. Material 폴더에 이미지가 있는지 확인하세요.")
        else:
            print(f"{count}개 파일이 복사되었습니다.")
            print("복사된 파일 목록:")
            for i, file_path in enumerate(files[:10], 1):
                print(f"  {i}. {file_path}")
            if len(files) > 10:
                print(f"  ... 외 {len(files) - 10}개 파일")
    
    # Autodistill 어노테이션을 YOLO 형식으로 변환
    elif args.command == "convert":
        # 클래스 매핑 로드
        class_mapping = {}
        if args.class_mapping:
            with open(args.class_mapping, 'r') as f:
                class_mapping = json.load(f)
        else:
            # 명령줄에서 직접 입력
            print("클래스 매핑을 입력하세요. 빈 줄을 입력하면 종료됩니다.")
            print("형식: 클래스이름 클래스ID (예: person 0)")
            while True:
                line = input("> ").strip()
                if not line:
                    break
                parts = line.split()
                if len(parts) != 2:
                    print("형식 오류. 다시 입력하세요. (예: person 0)")
                    continue
                class_name, class_id = parts
                class_mapping[class_name] = int(class_id)
        
        # 어노테이션 변환
        count, files = convert_autodistill_to_yolo(
            args.category,
            class_mapping,
            Path(args.annotation_file),
            args.img_width,
            args.img_height
        )
        
        if count == 0:
            print("변환된 어노테이션이 없습니다.")
        else:
            print(f"{count}개 어노테이션이 YOLO 형식으로 변환되었습니다.")
            print("변환된 파일 목록:")
            for i, file_path in enumerate(files[:10], 1):
                print(f"  {i}. {file_path}")
            if len(files) > 10:
                print(f"  ... 외 {len(files) - 10}개 파일")

if __name__ == "__main__":
    main() 