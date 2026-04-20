#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support Set 관리 도구

이 모듈은 Few-Shot Learning을 위한 Support Set을 체계적으로 구성하고 관리하는 기능을 제공합니다.
shot별로 명확한 이미지 세트를 구성하여 실험 결과의 일관성을 보장합니다.
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any
import glob

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("support_set.log")
    ]
)
logger = logging.getLogger(__name__)

# 프로젝트 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import project utilities
try:
    from scripts.data_utils import get_category_path, load_class_mapping
except ImportError:
    logger.warning("data_utils 모듈을 로드할 수 없습니다. 기본 경로 탐색 방식을 사용합니다.")
    def get_category_path(category_name):
        return os.path.join("data", category_name)
    
    def load_class_mapping(category_name):
        mapping_file = os.path.join(get_category_path(category_name), "class_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}

class SupportSetManager:
    """Support Set 관리 클래스"""
    
    def __init__(self, category_name: str):
        """
        초기화 함수
        
        Args:
            category_name: 대상 카테고리 이름
        """
        self.category_name = category_name
        self.category_path = get_category_path(category_name)
        
        # 경로 설정
        self.original_support_dir = os.path.join(self.category_path, "2.support-set")
        self.structured_support_dir = os.path.join(self.category_path, "2.support-set-structured")
        
        # 표준 shot 값 설정
        self.standard_shots = [1, 5, 10, 30]
        
        # 클래스 매핑 로드
        try:
            self.class_mapping = load_class_mapping(category_name)
            logger.info(f"클래스 매핑 로드 완료: {self.class_mapping}")
        except Exception as e:
            logger.warning(f"클래스 매핑 로드 실패: {e}. 기본 클래스명 사용.")
            self.class_mapping = {}
        
        # 로드된 이미지 정보 저장용
        self.support_images = {}
        self.structured_support_info = {}
    
    def load_original_support_set(self) -> Dict[str, List[str]]:
        """
        원본 support set 로드
        
        Returns:
            클래스별 이미지 경로 사전
        """
        if not os.path.exists(self.original_support_dir):
            raise FileNotFoundError(f"Support set 디렉토리를 찾을 수 없습니다: {self.original_support_dir}")
        
        # 클래스별 이미지 경로 사전 생성
        support_images = {}
        
        # 클래스 디렉토리 순회
        for class_name in os.listdir(self.original_support_dir):
            class_dir = os.path.join(self.original_support_dir, class_name)
            if os.path.isdir(class_dir):
                # 이미지 파일 목록 가져오기
                image_files = sorted(glob.glob(os.path.join(class_dir, "*.jpg")) + 
                                    glob.glob(os.path.join(class_dir, "*.jpeg")) + 
                                    glob.glob(os.path.join(class_dir, "*.png")) + 
                                    glob.glob(os.path.join(class_dir, "*.webp")))
                
                support_images[class_name] = image_files
                logger.info(f"클래스 {class_name}에서 {len(image_files)}개 이미지 로드됨")
        
        self.support_images = support_images
        return support_images
    
    def create_structured_support_set(self, force: bool = False) -> Dict[str, Dict[int, List[str]]]:
        """
        구조화된 support set 생성
        
        Args:
            force: 기존 구조화된 support set을 덮어쓸지 여부
            
        Returns:
            shot별, 클래스별 이미지 경로 사전
        """
        # 이미 구조화된 디렉토리가 있는지 확인
        if os.path.exists(self.structured_support_dir) and not force:
            logger.warning(f"구조화된 support set 디렉토리가 이미 존재합니다: {self.structured_support_dir}")
            logger.warning("기존 디렉토리를 덮어쓰려면 force=True로 설정하세요.")
            return self._load_structured_support_info()
        
        # 원본 support set 로드
        if not self.support_images:
            self.load_original_support_set()
        
        # 구조화된 디렉토리 생성
        if os.path.exists(self.structured_support_dir) and force:
            shutil.rmtree(self.structured_support_dir)
        os.makedirs(self.structured_support_dir, exist_ok=True)
        
        # shot별 구조화된 정보 사전
        structured_info = {shot: {} for shot in self.standard_shots}
        
        # 각 shot 값에 대해 디렉토리 생성 및 이미지 복사
        for shot in self.standard_shots:
            shot_dir = os.path.join(self.structured_support_dir, f"{shot}-shot")
            os.makedirs(shot_dir, exist_ok=True)
            
            # 각 클래스에 대해 처리
            for class_name, image_files in self.support_images.items():
                # shot 수만큼 이미지 선택
                selected_images = image_files[:shot] if len(image_files) >= shot else image_files
                
                # shot에 필요한 이미지가 부족한 경우 경고
                if len(selected_images) < shot:
                    logger.warning(f"클래스 {class_name}에 {shot}-shot에 필요한 이미지가 부족합니다. "
                                 f"필요: {shot}, 가능: {len(selected_images)}")
                
                # 클래스별 디렉토리 생성
                class_dir = os.path.join(shot_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # 이미지 복사
                for i, img_path in enumerate(selected_images):
                    img_filename = os.path.basename(img_path)
                    # 파일명에 순서 정보 추가 (선택사항)
                    dest_filename = f"{i+1:02d}_{img_filename}"
                    dest_path = os.path.join(class_dir, dest_filename)
                    shutil.copy2(img_path, dest_path)
                
                # 구조화된 정보에 추가
                structured_info[shot][class_name] = selected_images
            
            logger.info(f"{shot}-shot 디렉토리 생성 완료: {shot_dir}")
        
        # 구조화된 정보 저장
        self._save_structured_support_info(structured_info)
        self.structured_support_info = structured_info
        
        logger.info(f"구조화된 support set 생성 완료: {self.structured_support_dir}")
        return structured_info
    
    def _save_structured_support_info(self, structured_info: Dict[str, Dict[int, List[str]]]):
        """
        구조화된 support set 정보 저장
        
        Args:
            structured_info: 구조화된 정보 사전
        """
        # 경로를 상대 경로로 변환
        serializable_info = {}
        for shot, classes in structured_info.items():
            serializable_info[str(shot)] = {}
            for class_name, paths in classes.items():
                serializable_info[str(shot)][class_name] = [
                    os.path.relpath(p, self.category_path) for p in paths
                ]
        
        # JSON 파일로 저장
        info_file = os.path.join(self.structured_support_dir, "support_set_info.json")
        with open(info_file, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        logger.info(f"구조화된 support set 정보 저장 완료: {info_file}")
    
    def _load_structured_support_info(self) -> Dict[str, Dict[int, List[str]]]:
        """
        구조화된 support set 정보 로드
        
        Returns:
            구조화된 정보 사전
        """
        info_file = os.path.join(self.structured_support_dir, "support_set_info.json")
        if not os.path.exists(info_file):
            logger.warning(f"구조화된 support set 정보 파일을 찾을 수 없습니다: {info_file}")
            return {}
        
        with open(info_file, 'r') as f:
            serialized_info = json.load(f)
        
        # 상대 경로를 절대 경로로 변환
        structured_info = {}
        for shot_str, classes in serialized_info.items():
            shot = int(shot_str)
            structured_info[shot] = {}
            for class_name, rel_paths in classes.items():
                structured_info[shot][class_name] = [
                    os.path.join(self.category_path, p) for p in rel_paths
                ]
        
        self.structured_support_info = structured_info
        logger.info(f"구조화된 support set 정보 로드 완료: {info_file}")
        return structured_info
    
    def get_support_set_for_shot(self, shot: int) -> Dict[str, List[str]]:
        """
        특정 shot 값에 대한 support set 가져오기
        
        Args:
            shot: 원하는 shot 값
            
        Returns:
            클래스별 이미지 경로 사전
        """
        # 구조화된 정보가 없으면 로드
        if not self.structured_support_info:
            if os.path.exists(self.structured_support_dir):
                self._load_structured_support_info()
            else:
                self.create_structured_support_set()
        
        # 요청된 shot이 표준 shot 목록에 없는 경우
        if shot not in self.standard_shots:
            closest_shot = min(self.standard_shots, key=lambda x: abs(x - shot))
            logger.warning(f"요청된 shot 값 {shot}이 표준 목록에 없습니다. 가장 가까운 {closest_shot}을 사용합니다.")
            shot = closest_shot
        
        # shot 정보 반환
        return self.structured_support_info.get(shot, {})
    
    def validate_support_set(self) -> bool:
        """
        Support set 유효성 검증
        
        Returns:
            유효성 여부
        """
        # 원본 support set 로드
        if not self.support_images:
            self.load_original_support_set()
        
        is_valid = True
        
        # 각 클래스에 충분한 이미지가 있는지 검증
        max_shot = max(self.standard_shots)
        for class_name, image_files in self.support_images.items():
            if len(image_files) < max_shot:
                logger.warning(f"클래스 {class_name}에 최대 shot({max_shot})에 필요한 이미지가 부족합니다. "
                             f"필요: {max_shot}, 가능: {len(image_files)}")
                is_valid = False
        
        return is_valid
    
    def report_support_set_status(self) -> Dict[str, Any]:
        """
        Support set 상태 보고서 생성
        
        Returns:
            상태 보고서 사전
        """
        # 원본 support set 로드
        if not self.support_images:
            self.load_original_support_set()
        
        # 보고서 초기화
        report = {
            "category": self.category_name,
            "original_support_dir": self.original_support_dir,
            "structured_support_dir": self.structured_support_dir,
            "class_counts": {},
            "shot_coverage": {},
            "is_valid": True,
            "missing_images": {}
        }
        
        # 클래스별 이미지 수 기록
        for class_name, image_files in self.support_images.items():
            report["class_counts"][class_name] = len(image_files)
        
        # 각 shot 값에 대한 커버리지 계산
        for shot in self.standard_shots:
            covered_classes = []
            missing_classes = []
            
            for class_name, image_files in self.support_images.items():
                if len(image_files) >= shot:
                    covered_classes.append(class_name)
                else:
                    missing_classes.append(class_name)
                    if class_name not in report["missing_images"]:
                        report["missing_images"][class_name] = {}
                    report["missing_images"][class_name][shot] = shot - len(image_files)
            
            report["shot_coverage"][shot] = {
                "covered_classes": covered_classes,
                "missing_classes": missing_classes,
                "coverage_percent": 100 * len(covered_classes) / len(self.support_images) if self.support_images else 0
            }
            
            # 모든 클래스가 커버되지 않으면 유효하지 않음
            if missing_classes:
                report["is_valid"] = False
        
        return report

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="Support Set 관리 도구")
    parser.add_argument("--category", type=str, required=True,
                      help="대상 카테고리 이름")
    parser.add_argument("--create", action="store_true",
                      help="구조화된 support set 생성")
    parser.add_argument("--validate", action="store_true",
                      help="Support set 유효성 검증")
    parser.add_argument("--report", action="store_true",
                      help="Support set 상태 보고서 생성")
    parser.add_argument("--force", action="store_true",
                      help="기존 구조화된 support set 덮어쓰기")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    try:
        # Support Set 관리자 초기화
        manager = SupportSetManager(args.category)
        
        if args.validate:
            # 유효성 검증
            is_valid = manager.validate_support_set()
            if is_valid:
                logger.info("모든 클래스에 충분한 이미지가 있습니다.")
            else:
                logger.warning("일부 클래스에 이미지가 부족합니다. 보고서를 확인하세요.")
        
        if args.report:
            # 상태 보고서 생성
            report = manager.report_support_set_status()
            report_file = os.path.join(manager.category_path, "support_set_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # 보고서 내용 출력
            print("\n===== SUPPORT SET 상태 보고서 =====")
            print(f"카테고리: {report['category']}")
            print(f"유효성: {'유효함' if report['is_valid'] else '유효하지 않음'}")
            print("\n클래스별 이미지 수:")
            for class_name, count in report["class_counts"].items():
                print(f"  - {class_name}: {count}개")
            
            print("\nShot 커버리지:")
            for shot, coverage in report["shot_coverage"].items():
                print(f"  - {shot}-shot: {coverage['coverage_percent']:.1f}% ({len(coverage['covered_classes'])}/{len(report['class_counts'])} 클래스)")
                if coverage["missing_classes"]:
                    print(f"    부족한 클래스: {', '.join(coverage['missing_classes'])}")
            
            print(f"\n보고서가 {report_file}에 저장되었습니다.")
        
        if args.create:
            # 구조화된 support set 생성
            manager.create_structured_support_set(force=args.force)
            print(f"\n구조화된 support set이 {manager.structured_support_dir}에 생성되었습니다.")
            print("\n각 shot 값별로 디렉토리가 생성되었으며, 해당 디렉토리 내에 클래스별 서브디렉토리와 이미지가 복사되었습니다.")
            print("\n실험 실행 시 --support-set-dir 옵션을 사용하여 특정 shot 디렉토리를 지정할 수 있습니다.")
        
        return 0
    
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 