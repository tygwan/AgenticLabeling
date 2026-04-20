#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support Set 구조 변경 스크립트

기존 구조:
2.support-set/
    class_0/
        shot1/
        shot5/
        shot10/
        shot30/
    class_1/
        ...

새 구조:
2.support-set/
    shot1/
        class0/
        class1/
        class2/
        class3/
    shot5/
        ...
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# data_utils 임포트
try:
    from scripts.data_utils import get_category_path
except ImportError:
    print("Error importing data_utils. Using local implementation.")
    def get_category_path(category_name):
        """Get the path to a category directory."""
        return os.path.join(project_root, "data", category_name)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("restructure_support_set")

def restructure_support_set(category_name, backup=True, force=False):
    """
    Support set 구조를 변경합니다.
    
    Args:
        category_name: 카테고리 이름
        backup: 기존 구조 백업 여부
        force: 기존 구조 강제 변경 여부
    """
    # 카테고리 경로 가져오기
    category_path = get_category_path(category_name)
    support_dir = os.path.join(category_path, "2.support-set")
    
    # 기존 구조 확인
    if not os.path.exists(support_dir):
        logger.error(f"Support set 디렉토리를 찾을 수 없습니다: {support_dir}")
        return False
    
    # 디렉토리 구조 분석
    has_class_dirs = False
    has_shot_subdirs = False
    
    for item in os.listdir(support_dir):
        item_path = os.path.join(support_dir, item)
        if os.path.isdir(item_path) and item.lower().startswith('class_'):
            has_class_dirs = True
            # Shot 서브디렉토리 확인
            for subitem in os.listdir(item_path):
                if os.path.isdir(os.path.join(item_path, subitem)) and subitem.startswith('shot'):
                    has_shot_subdirs = True
                    break
    
    # 새 구조로 이미 변경되었는지 확인
    already_new_structure = False
    for item in os.listdir(support_dir):
        if os.path.isdir(os.path.join(support_dir, item)) and item.startswith('shot'):
            already_new_structure = True
            break
    
    if already_new_structure and not force:
        logger.warning("Support set이 이미 새 구조로 변경되었습니다.")
        return True
    
    if not has_class_dirs or not has_shot_subdirs:
        logger.warning("기존 구조에서 클래스 디렉토리 또는 shot 서브디렉토리를 찾을 수 없습니다.")
        if not force:
            logger.error("구조 변경을 중단합니다. --force 옵션을 사용하여 강제 변경할 수 있습니다.")
            return False
    
    # 백업
    if backup:
        backup_dir = os.path.join(category_path, "2.support-set.backup")
        if os.path.exists(backup_dir):
            logger.warning(f"이미 백업 디렉토리가 존재합니다: {backup_dir}")
            backup_dir = os.path.join(category_path, f"2.support-set.backup.{int(time.time())}")
        
        logger.info(f"Support set 백업 중: {backup_dir}")
        shutil.copytree(support_dir, backup_dir)
    
    # 임시 디렉토리 생성
    temp_dir = os.path.join(category_path, "2.support-set.temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 새 구조 생성
    try:
        # 각 Shot 폴더를 최상위에 생성
        shot_folders = ["shot1", "shot5", "shot10", "shot30"]
        for shot_folder in shot_folders:
            os.makedirs(os.path.join(temp_dir, shot_folder), exist_ok=True)
        
        # 각 클래스 폴더 순회
        for class_dir in os.listdir(support_dir):
            class_dir_path = os.path.join(support_dir, class_dir)
            
            # 클래스 폴더만 처리
            if not os.path.isdir(class_dir_path) or not class_dir.lower().startswith('class_'):
                continue
            
            # 클래스 이름 변환 (class_0 -> class0)
            new_class_name = class_dir.replace('_', '')
            
            # 각 Shot 서브디렉토리 처리
            for shot_folder in os.listdir(class_dir_path):
                shot_folder_path = os.path.join(class_dir_path, shot_folder)
                
                # Shot 폴더만 처리
                if not os.path.isdir(shot_folder_path) or not shot_folder.startswith('shot'):
                    continue
                
                # 새 구조의 대상 디렉토리
                target_dir = os.path.join(temp_dir, shot_folder, new_class_name)
                os.makedirs(target_dir, exist_ok=True)
                
                # 이미지 복사
                for file_name in os.listdir(shot_folder_path):
                    file_path = os.path.join(shot_folder_path, file_name)
                    if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        shutil.copy2(file_path, os.path.join(target_dir, file_name))
                        logger.debug(f"복사됨: {file_path} -> {os.path.join(target_dir, file_name)}")
        
        # 기존 Support set 제거 및 새 구조 적용
        logger.info("새 구조 적용 중...")
        shutil.rmtree(support_dir)
        shutil.move(temp_dir, support_dir)
        
        logger.info("Support set 구조 변경 완료!")
        return True
        
    except Exception as e:
        logger.error(f"구조 변경 중 오류 발생: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # 임시 디렉토리 정리
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return False

def main():
    """명령행 인터페이스"""
    parser = argparse.ArgumentParser(description="Support Set 구조 변경")
    parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름")
    parser.add_argument("--no-backup", action="store_true", help="백업 생성하지 않음")
    parser.add_argument("--force", action="store_true", help="강제 변경")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    # 구조 변경 실행
    success = restructure_support_set(
        category_name=args.category,
        backup=not args.no_backup,
        force=args.force
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    import time
    sys.exit(main()) 