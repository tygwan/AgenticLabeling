#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클래스 매핑과 support set 로드 테스트 스크립트
"""

import os
import sys
import json
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 프로젝트 유틸리티 임포트
try:
    from scripts.data_utils import get_category_path, load_class_mapping
except ImportError:
    logger.error("프로젝트 유틸리티를 임포트할 수 없습니다.")
    sys.exit(1)

def test_class_mapping(category_name):
    """
    클래스 매핑 파일 로드 테스트
    """
    logger.info(f"카테고리 '{category_name}'의 클래스 매핑 테스트 중...")
    
    try:
        # 클래스 매핑 로드
        class_mapping = load_class_mapping(category_name)
        logger.info(f"클래스 매핑 로드됨: {class_mapping}")
        
        # 카테고리 경로 가져오기
        category_path = get_category_path(category_name)
        logger.info(f"카테고리 경로: {category_path}")
        
        # 지원 세트 폴더 확인
        support_dir = os.path.join(category_path, "2.support-set")
        if not os.path.exists(support_dir):
            logger.error(f"지원 세트 디렉토리가 존재하지 않습니다: {support_dir}")
            return
        
        logger.info(f"지원 세트 디렉토리 확인됨: {support_dir}")
        
        # shot 폴더 확인
        shot_folders = [f for f in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, f)) and f.startswith("shot")]
        logger.info(f"발견된 shot 폴더: {shot_folders}")
        
        # 첫번째 shot 폴더 분석
        if shot_folders:
            shot_folder = shot_folders[0]
            shot_path = os.path.join(support_dir, shot_folder)
            
            # 클래스 폴더 확인
            class_folders = [f for f in os.listdir(shot_path) if os.path.isdir(os.path.join(shot_path, f))]
            logger.info(f"Shot 폴더 '{shot_folder}'에서 발견된 클래스 폴더: {class_folders}")
            
            # 클래스 매핑에 있는 클래스가 폴더 구조에 존재하는지 확인
            for class_name in class_mapping.keys():
                if class_name in class_folders:
                    logger.info(f"클래스 '{class_name}'는 폴더 구조에 존재합니다.")
                else:
                    logger.warning(f"클래스 매핑에 있는 클래스 '{class_name}'는 폴더 구조에 존재하지 않습니다.")
            
            # 폴더 구조에 있는 클래스가 클래스 매핑에 존재하는지 확인
            for folder_name in class_folders:
                if folder_name in class_mapping:
                    logger.info(f"폴더 '{folder_name}'는 클래스 매핑에 존재합니다.")
                else:
                    logger.warning(f"폴더 '{folder_name}'는 클래스 매핑에 존재하지 않습니다.")
            
            # 각 클래스 폴더의 이미지 수 확인
            for class_folder in class_folders:
                class_path = os.path.join(shot_path, class_folder)
                image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                logger.info(f"클래스 '{class_folder}'에는 {len(image_files)}개의 이미지가 있습니다.")
                
                # 처음 몇 개 이미지 파일 출력
                if image_files:
                    logger.info(f"  첫 {min(3, len(image_files))}개 이미지: {image_files[:3]}")
    
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 커맨드 라인 인자 처리
    if len(sys.argv) > 1:
        category_name = sys.argv[1]
    else:
        category_name = "test_category"  # 기본값
    
    test_class_mapping(category_name)
    logger.info("테스트 완료!") 