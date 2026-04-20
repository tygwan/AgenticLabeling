#!/usr/bin/env python3
"""
Few-Shot 분류 결과 정리 도구

이 스크립트는 few-shot 분류 결과를 바탕으로 이미지를 클래스별 폴더로 분류합니다.
shot*threshold 조합마다 class_0, class_1, class_2, class_3, unknown 폴더를 생성합니다.
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("organize_results.log")
    ]
)
logger = logging.getLogger(__name__)

# 프로젝트 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="Few-Shot 분류 결과 정리 도구")
    parser.add_argument("--category", type=str, required=True, help="정리할 카테고리 (예: test_category)")
    parser.add_argument("--shot-values", type=str, default="1,5,10,30", 
                        help="정리할 shot 값들 (쉼표로 구분)")
    parser.add_argument("--threshold-values", type=str, default="0.5,0.6,0.7,0.8,0.9",
                        help="정리할 threshold 값들 (쉼표로 구분)")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="원본 이미지 디렉토리 (기본값: data/{category}/5.dataset/val)")
    parser.add_argument("--force", action="store_true", help="기존 분류 폴더가 있을 경우 덮어쓰기")
    return parser.parse_args()

class ResultsOrganizer:
    def __init__(self, category, source_dir=None, force=False):
        """초기화"""
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.category = category
        self.category_dir = self.data_dir / category
        self.force = force
        
        # 원본 이미지 디렉토리 설정
        if source_dir:
            self.source_dir = Path(source_dir)
        else:
            self.source_dir = self.category_dir / "5.dataset" / "val"
        
        # 결과 디렉토리
        self.results_dir = self.category_dir / "7.results"
        
        # 클래스 목록
        self.classes = ["class_0", "class_1", "class_2", "class_3", "unknown"]
    
    def validate_environment(self):
        """환경 검증"""
        if not self.source_dir.exists():
            logger.error(f"원본 이미지 디렉토리가 존재하지 않습니다: {self.source_dir}")
            return False
        
        if not self.results_dir.exists():
            logger.error(f"결과 디렉토리가 존재하지 않습니다: {self.results_dir}")
            return False
        
        logger.info(f"원본 이미지 디렉토리: {self.source_dir}")
        logger.info(f"결과 디렉토리: {self.results_dir}")
        
        return True
    
    def process_experiment(self, shot, threshold):
        """개별 실험 결과 처리"""
        # 실험 식별자
        exp_key = f"shot_{shot}_threshold_{threshold}"
        results_subdir = self.results_dir / f"shot_{shot}" / f"threshold_{threshold}"
        results_file = results_subdir / "results.json"
        
        # 결과 파일 확인
        if not results_file.exists():
            logger.warning(f"{exp_key}에 대한 결과 파일이 존재하지 않습니다: {results_file}")
            return False
        
        try:
            # 결과 로드
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info(f"{exp_key} 결과 로드 완료: {len(results)} 이미지")
            
            # 클래스별 폴더 생성
            for class_name in self.classes:
                class_dir = results_subdir / class_name
                
                # 기존 폴더 처리
                if class_dir.exists() and self.force:
                    shutil.rmtree(class_dir)
                
                # 폴더 생성
                class_dir.mkdir(exist_ok=True, parents=True)
            
            # 이미지를 클래스별 폴더로 복사
            copied_count = 0
            for img_name, img_result in results.items():
                predicted_class = img_result.get("predicted_class", "unknown")
                
                # 원본 이미지 경로
                source_img = self.source_dir / img_name
                
                # 대상 폴더
                if predicted_class.startswith("class_"):
                    target_dir = results_subdir / predicted_class
                else:
                    target_dir = results_subdir / "unknown"
                
                # 이미지 복사
                if source_img.exists():
                    shutil.copy2(source_img, target_dir / img_name)
                    copied_count += 1
                else:
                    logger.warning(f"원본 이미지를 찾을 수 없습니다: {source_img}")
            
            logger.info(f"{exp_key}: {copied_count} 이미지를 클래스별 폴더로 복사했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"{exp_key} 처리 중 오류: {e}")
            return False
    
    def organize_results(self, shot_values, threshold_values):
        """결과 정리 실행"""
        logger.info(f"{len(shot_values)}개의 shot 값과 {len(threshold_values)}개의 threshold 값에 대한 결과 정리 시작")
        
        successful = 0
        failed = 0
        
        # 병렬 처리로 속도 향상
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for shot in shot_values:
                for threshold in threshold_values:
                    futures.append(
                        executor.submit(self.process_experiment, shot, threshold)
                    )
            
            # 결과 수집
            for future in futures:
                if future.result():
                    successful += 1
                else:
                    failed += 1
        
        logger.info(f"결과 정리 완료: {successful}개 성공, {failed}개 실패")
        return successful > 0
    
    def create_summary(self, shot_values, threshold_values):
        """결과 요약 생성"""
        summary = {
            "category": self.category,
            "experiments": {}
        }
        
        for shot in shot_values:
            for threshold in threshold_values:
                exp_key = f"shot_{shot}_threshold_{threshold}"
                results_subdir = self.results_dir / f"shot_{shot}" / f"threshold_{threshold}"
                
                class_counts = {}
                
                # 각 클래스별 이미지 수 계산
                for class_name in self.classes:
                    class_dir = results_subdir / class_name
                    if class_dir.exists():
                        count = len(list(class_dir.glob("*.*")))
                        class_counts[class_name] = count
                
                # 요약에 추가
                if class_counts:
                    summary["experiments"][exp_key] = {
                        "class_counts": class_counts,
                        "total_images": sum(class_counts.values())
                    }
        
        # 요약 저장
        summary_file = self.results_dir / "classification_summary.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"분류 요약이 {summary_file}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"요약 저장 중 오류: {e}")
        
        return True

def main():
    """메인 함수"""
    args = parse_args()
    
    # shot 및 threshold 값 파싱
    shot_values = [int(s.strip()) for s in args.shot_values.split(",")]
    threshold_values = [float(t.strip()) for t in args.threshold_values.split(",")]
    
    logger.info(f"카테고리: {args.category}")
    logger.info(f"정리할 shot 값: {shot_values}")
    logger.info(f"정리할 threshold 값: {threshold_values}")
    
    # 결과 정리기 초기화
    organizer = ResultsOrganizer(
        args.category,
        args.source_dir,
        args.force
    )
    
    # 환경 검증
    if not organizer.validate_environment():
        logger.error("환경 검증 실패")
        return 1
    
    # 결과 정리
    if not organizer.organize_results(shot_values, threshold_values):
        logger.error("결과 정리 실패")
        return 1
    
    # 요약 생성
    organizer.create_summary(shot_values, threshold_values)
    
    logger.info("모든 결과 정리 완료")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 