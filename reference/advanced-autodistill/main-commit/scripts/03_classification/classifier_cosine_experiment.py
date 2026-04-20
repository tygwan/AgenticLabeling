#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning 실험 플랫폼

N-Shot K-Way 분류 실험을 실행하고 분석하는 도구입니다.
다양한 shot 수와 threshold 값 조합으로 실험을 실행하고, 결과를 분석할 수 있습니다.

주요 기능:
- 여러 shot-threshold 조합으로 분류 실험 실행
- 예측 결과 평가 및 시각화
- Annotation 정보 동기화 및 분석
- Ground truth와 예측 결과 비교

사용 예시:
1. 기본 실험 실행:
   python classifier_cosine_experiment.py --category test_category --model resnet --shots 1,5,10,30 --thresholds 0.6,0.7,0.8,0.9

2. 특정 실험 평가:
   python classifier_cosine_experiment.py --evaluate --experiment-id shot_5_threshold_0.70

3. Annotation 동기화:
   python classifier_cosine_experiment.py --sync-annotations --experiment-id shot_5_threshold_0.70 --save-by-class

4. Ground truth 분석:
   python classifier_cosine_experiment.py --analyze-ground-truth

5. 예측 결과와 Ground truth 비교:
   python classifier_cosine_experiment.py --compare-annotations --experiment-id shot_5_threshold_0.70

실행 절차:
1. 실험 실행: 다양한 shot-threshold 조합으로 분류 실험 실행
2. Ground truth 구성: 7.results/ground_truth 디렉토리에 class_0, class_1 등의 폴더 생성하고 정답 이미지 배치
3. Annotation 동기화: 실험 결과에 원본 annotation 정보 연결
4. Ground truth 분석: Ground truth 폴더의 annotation 정보 분석
5. 결과 비교: 예측 결과와 ground truth 비교 및 시각화
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
import argparse
from datetime import datetime
import time

# Import project utilities
from data_utils import get_category_path, load_class_mapping
from classifier_cosine import CosineSimilarityClassifier, FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fsl_experiment")

class FewShotExperiment:
    """Few-Shot Learning 실험 관리자 클래스"""
    
    def __init__(self, category_name: str, model_name: str = "resnet", save_images: bool = False, group_unknown: bool = True):
        """
        초기화 함수
        
        Args:
            category_name: 실험 대상 카테고리 이름
            model_name: 특징 추출에 사용할 모델 이름 (resnet, clip, dino)
            save_images: 분류된 이미지를 클래스별 폴더에 저장할지 여부
            group_unknown: unknown_* 카테고리를 하나의 'Unknown' 클래스로 그룹화할지 여부
        """
        self.category_name = category_name
        self.model_name = model_name.lower()
        self.category_path = get_category_path(category_name)
        self.save_images = save_images  # 이미지 저장 여부 설정
        self.group_unknown = group_unknown  # unknown 그룹화 여부
        
        # 실험 설정 초기화
        self.n_shots = [1, 5, 10, 30]
        self.thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        
        # 경로 설정
        self.support_dir = os.path.join(self.category_path, "2.support-set")
        self.structured_support_dir = os.path.join(self.category_path, "2.support-set-structured")
        self.preprocessed_dir = os.path.join(self.category_path, "6.preprocessed")
        
        # 모델별로 결과 디렉토리 구분
        self.base_results_dir = os.path.join(self.category_path, "000.timetest")
        self.results_dir = os.path.join(self.base_results_dir, self.model_name)
        
        # 클래스 매핑 로드
        try:
            self.class_mapping = load_class_mapping(category_name)
            logger.info(f"클래스 매핑 로드 완료: {self.class_mapping}")
        except Exception as e:
            logger.warning(f"클래스 매핑 로드 실패: {e}. 기본 클래스명 사용.")
            self.class_mapping = {}
        
        # 결과 데이터 저장 경로 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 결과 데이터 저장 변수
        self.experiment_results = {}
        
        logger.info(f"Few-Shot Learning 실험 초기화 완료: 카테고리={category_name}, 모델={model_name}, 이미지 저장={save_images}")
        logger.info(f"결과 저장 경로: {self.results_dir}")
    
    def run_all_experiments(self, input_dir: Optional[str] = None):
        """
        모든 N-shot 및 threshold 조합으로 실험 실행
        
        Args:
            input_dir: 분류할 이미지가 있는 디렉토리 (기본값: preprocessed_dir)
        """
        if input_dir is None:
            input_dir = self.preprocessed_dir
        
        logger.info(f"총 {len(self.n_shots) * len(self.thresholds)}개 실험 조합 실행 시작")
        logger.info(f"모델: {self.model_name}")
        
        # 전체 실험 시간 측정을 위한 변수
        total_execution_time = 0
        experiment_count = 0
        
        # 구조화된 support set 확인
        use_structured = os.path.exists(self.structured_support_dir)
        if use_structured:
            logger.info(f"구조화된 support set을 사용합니다: {self.structured_support_dir}")
        else:
            logger.info(f"원본 support set을 사용합니다: {self.support_dir}")
        
        # 각 N-shot 설정에 대해 실험
        for n_shot in self.n_shots:
            logger.info(f"{n_shot}-shot 실험 시작")
            
            # 각 threshold 값에 대해 실험
            for threshold in self.thresholds:
                logger.info(f"  Threshold {threshold:.2f} 실험 실행 중...")
                
                # 실험 시간 측정 시작
                start_time = time.time()
                
                # 실험 ID 생성
                experiment_id = f"shot_{n_shot}_threshold_{threshold:.2f}"
                
                # 결과 저장 디렉토리 생성
                experiment_dir = os.path.join(self.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
                os.makedirs(experiment_dir, exist_ok=True)
                
                # 클래스별 이미지 저장 디렉토리 생성
                for class_name in ["Class_0", "Class_1", "Class_2", "Class_3", "Unknown"]:
                    class_dir = os.path.join(experiment_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                
                # 분류기 초기화
                classifier = CosineSimilarityClassifier(
                    model_name=self.model_name,
                    k_shot=n_shot,
                    similarity_threshold=threshold
                )
                
                # support set 로드
                if use_structured:
                    # 구조화된 support set 사용
                    shot_dir = os.path.join(self.structured_support_dir, f"shot{n_shot}")
                    is_valid = os.path.exists(shot_dir)
                    if is_valid:
                        classifier.load_support_set(self.category_name, shot_dir)
                    else:
                        logger.error(f"Shot 디렉토리를 찾을 수 없음: {shot_dir}")
                        continue
                else:
                    # 원본 support set에서 shot 폴더 찾기
                    shot_folder = f"shot{n_shot}"
                    shot_dir = os.path.join(self.support_dir, shot_folder)
                    is_valid = os.path.exists(shot_dir)
                    if is_valid:
                        classifier.load_support_set(self.category_name, shot_dir)
                    else:
                        logger.error(f"Shot 디렉토리를 찾을 수 없음: {shot_dir}")
                        continue
                
                # support set 특징 추출
                classifier.extract_support_features()
                
                # 분류할 이미지 목록 가져오기
                image_files = self._get_image_files(input_dir)
                logger.info(f"  총 {len(image_files)}개 이미지 분류 중...")
                
                # 분류 실행
                results = classifier.classify_batch(image_files)
                
                # 실험 시간 측정 종료
                end_time = time.time()
                execution_time = end_time - start_time
                total_execution_time += execution_time
                experiment_count += 1
                
                # 결과를 predictions.csv로 저장
                self._save_predictions_csv(results, experiment_dir)
                
                # 이미지 저장 설정이 활성화된 경우에만 이미지 저장
                if self.save_images:
                    logger.info("이미지 저장 설정이 활성화되어 분류된 이미지를 클래스별 폴더에 저장합니다")
                    self._save_classified_images(results, experiment_dir)
                else:
                    logger.info("이미지 저장 설정이 비활성화되어 이미지 파일을 복사하지 않습니다")
                
                # 결과 기록
                self.experiment_results[experiment_id] = {
                    "model": self.model_name,  # 모델 정보 추가
                    "n_shot": n_shot,
                    "threshold": threshold,
                    "results_dir": experiment_dir,
                    "total_images": len(results),
                    "classified_count": sum(1 for r in results if r["class"] is not None and r["class"] != "Unknown"),
                    "unknown_count": sum(1 for r in results if r["class"] == "Unknown"),
                    "support_set_valid": is_valid,
                    "execution_time": execution_time
                }
                
                logger.info(f"실험 {experiment_id} ({self.model_name}) 완료: "  # 모델 정보 포함
                           f"총 {len(results)}개 이미지 중 "
                           f"{self.experiment_results[experiment_id]['classified_count']}개 분류됨, "
                           f"{self.experiment_results[experiment_id]['unknown_count']}개 Unknown "
                           f"(소요 시간: {execution_time:.2f}초)")
        
        avg_execution_time = total_execution_time / experiment_count if experiment_count > 0 else 0
        logger.info(f"모든 {self.model_name} 모델 실험 완료. 총 소요 시간: {total_execution_time:.2f}초, 평균 처리 시간: {avg_execution_time:.2f}초")
        
        # 실험 요약 저장
        self._save_experiment_summary(total_execution_time, avg_execution_time)
        
        # 결과 비교 분석 생성
        self._create_comparative_analysis()

        # 새로운 종합 보고서 및 시각화 생성
        logger.info("종합 결과 보고서 및 시각화를 생성합니다...")
        try:
            self.generate_comprehensive_report()
            logger.info("종합 결과 보고서 및 시각화 생성을 완료했습니다.")
        except Exception as e:
            logger.error(f"종합 결과 보고서 생성 중 오류 발생: {e}", exc_info=True)
    
    def _get_image_files(self, input_dir: str) -> List[str]:
        """
        이미지 파일 목록 가져오기
        
        Args:
            input_dir: 이미지 파일이 있는 디렉토리
            
        Returns:
            이미지 파일 경로 목록
        """
        image_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"{input_dir}에서 {len(image_files)}개 이미지 파일 찾음")
        return image_files
    
    def _save_classified_images(self, results: List[Dict[str, Any]], output_dir: str):
        """
        분류된 이미지를 클래스별 폴더에 저장
        
        Args:
            results: 분류 결과 목록
            output_dir: 결과 저장 기본 디렉토리
        """
        import shutil
        
        # 클래스별 폴더 생성
        class_dirs = {}
        unique_classes = set()
        
        # 예측된 클래스 추출 및 정규화
        for result in results:
            class_name = result["class"]
            
            # 클래스 이름 정규화 (Class_0 형식으로 통일)
            if class_name.lower().startswith('class_'):
                class_num = class_name.split('_')[1]
                normalized_class_name = f"Class_{class_num}"
                # 결과 객체의 클래스 이름도 업데이트
                result["class"] = normalized_class_name
            else:
                normalized_class_name = class_name
                
            if normalized_class_name not in unique_classes:
                unique_classes.add(normalized_class_name)
        
        # Unknown 클래스 추가
        if "Unknown" not in unique_classes:
            unique_classes.add("Unknown")
        
        # 클래스별 폴더 생성
        for class_name in unique_classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[class_name] = class_dir
        
        # 이미지를 각 클래스 폴더로 복사
        for result in results:
            image_path = result["image"]
            predicted_class = result["class"]  # 이미 정규화된 클래스 이름
            
            # 파일 이름에 원본 클래스 정보 추가
            original_class = result.get("original_class", "unknown_origin")
            # 원본 클래스도 정규화
            if original_class and original_class.lower().startswith('class_'):
                class_num = original_class.split('_')[1]
                original_class = f"Class_{class_num}"
                result["original_class"] = original_class
                
            file_name = os.path.basename(image_path)
            
            # 이미지 파일 복사
            if predicted_class in class_dirs:
                dest_path = os.path.join(class_dirs[predicted_class], file_name)
                try:
                    shutil.copy2(image_path, dest_path)
                except Exception as e:
                    logger.error(f"이미지 복사 실패: {image_path} -> {dest_path}, 오류: {e}")
        
        logger.info(f"분류된 이미지를 {len(class_dirs)}개 클래스 폴더에 저장 완료")

    def _save_predictions_csv(self, results: List[Dict[str, Any]], output_dir: str):
        """
        분류 결과를 CSV 파일로 저장
        
        Args:
            results: 분류 결과 목록
            output_dir: 결과 저장 디렉토리
        """
        # 결과 데이터 준비
        data = []
        for result in results:
            image_path = result["image"]
            image_filename = os.path.basename(image_path)
            
            # 예측 클래스 정규화
            predicted_class = result["class"] if result["class"] is not None else "Unknown"
            if predicted_class.lower().startswith('class_'):
                class_num = predicted_class.split('_')[1]
                predicted_class = f"Class_{class_num}"
            
            # 원본 클래스 정규화
            original_class = result.get("original_class", "Unknown")
            if original_class.lower().startswith('class_'):
                class_num = original_class.split('_')[1]
                original_class = f"Class_{class_num}"
                
            similarity = result.get("similarity", 0.0)
            
            # 모든 클래스에 대한 스코어 추출 및 클래스 이름 정규화
            class_scores = {}
            for cls, score in result.get("all_scores", {}).items():
                # 스코어 키도 정규화
                if cls.lower().startswith('class_'):
                    class_num = cls.split('_')[1]
                    normalized_cls = f"Class_{class_num}"
                    class_scores[f"score_{normalized_cls}"] = score
                else:
                    class_scores[f"score_{cls}"] = score
            
            row = {
                "image_path": image_path,
                "image_filename": image_filename,
                "original_class": original_class,
                "predicted_class": predicted_class,
                "classification": f"{original_class} / {predicted_class}",  # OGclass / predicted class 형식
                "similarity": similarity,  # confidence를 similarity로 변경
                **class_scores
            }
            data.append(row)
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(data)
        output_file = os.path.join(output_dir, "predictions.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"예측 결과 저장 완료: {output_file}")
    
    def _save_experiment_summary(self, total_execution_time: float, avg_execution_time: float):
        """실험 요약 정보 저장"""
        # 모델별 요약 파일 생성
        summary_file = os.path.join(self.results_dir, f"experiment_summary_{self.model_name}.json")
        
        summary_data = {
            "model_name": self.model_name,
            "total_execution_time_seconds": total_execution_time,
            "average_execution_time_per_experiment_seconds": avg_execution_time,
            "experiments": self.experiment_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"{self.model_name} 모델 실험 요약 저장 완료: {summary_file}")
        
        # 전체 실험 요약 파일도 업데이트 (모든 모델 결과 통합)
        all_summary_file = os.path.join(self.base_results_dir, "all_experiment_summary.json")
        
        all_experiments = {}
        if os.path.exists(all_summary_file):
            try:
                with open(all_summary_file, 'r') as f:
                    all_experiments = json.load(f)
            except Exception as e:
                logger.warning(f"기존 전체 요약 파일 로드 실패: {e}")
        
        # 현재 모델 결과 추가 (모델 이름으로 그룹화)
        all_experiments[self.model_name] = summary_data
        
        with open(all_summary_file, 'w') as f:
            json.dump(all_experiments, f, indent=2)
        
        logger.info(f"전체 실험 요약 업데이트 완료: {all_summary_file}")

    def _create_comparative_analysis(self):
        """
        다양한 shot 및 threshold 조합의 실험 결과에 대한 비교 분석을 생성
        
        여러 실험 설정에 대한 결과를 비교하여 성능 차이를 시각화하고 최적의 구성을 식별
        """
        if not self.experiment_results:
            logger.warning("비교 분석을 위한 실험 결과가 없습니다.")
            return
        
        # 비교 분석 결과 저장 디렉토리
        analysis_dir = os.path.join(self.results_dir, "comparative_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 분석 데이터 준비
        shot_thresholds = {}  # shot별 threshold-성능 데이터
        for exp_id, exp_data in self.experiment_results.items():
            n_shot = exp_data["n_shot"]
            threshold = exp_data["threshold"]
            classified_count = exp_data["classified_count"]
            unknown_count = exp_data["unknown_count"]
            total = classified_count + unknown_count
            
            if n_shot not in shot_thresholds:
                shot_thresholds[n_shot] = {
                    "thresholds": [],
                    "classified_ratio": [],
                    "unknown_ratio": []
                }
            
            shot_thresholds[n_shot]["thresholds"].append(threshold)
            shot_thresholds[n_shot]["classified_ratio"].append(classified_count / total if total > 0 else 0)
            shot_thresholds[n_shot]["unknown_ratio"].append(unknown_count / total if total > 0 else 0)
        
        # 분석 결과 저장
        analysis_data = {
            "model": self.model_name,
            "shot_threshold_analysis": shot_thresholds,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 분석 결과 JSON 저장
        analysis_file = os.path.join(analysis_dir, f"comparative_analysis_{self.model_name}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"비교 분석 결과 저장 완료: {analysis_file}")
        
        try:
            # Matplotlib로 시각화 (선택적)
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 각 shot에 대한 threshold vs 분류율 그래프 생성
            plt.figure(figsize=(12, 8))
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            markers = ['o', 's', '^', 'D', 'v', '*', 'x']
            
            for i, (shot, data) in enumerate(shot_thresholds.items()):
                color_idx = i % len(colors)
                marker_idx = i % len(markers)
                
                # 데이터를 threshold에 따라 정렬
                sorted_indices = np.argsort(data["thresholds"])
                thresholds = [data["thresholds"][idx] for idx in sorted_indices]
                classified = [data["classified_ratio"][idx] for idx in sorted_indices]
                
                plt.plot(thresholds, classified, 
                         marker=markers[marker_idx], 
                         color=colors[color_idx], 
                         label=f"{shot}-shot")
            
            plt.xlabel('Threshold')
            plt.ylabel('Classified Ratio')
            plt.title(f'{self.model_name} Model: Threshold vs Classification Rate by Shot Count')
            plt.legend()
            plt.grid(True)
            
            # 그래프 이미지 저장
            plot_file = os.path.join(analysis_dir, f"threshold_vs_classification_{self.model_name}.png")
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"비교 분석 그래프 저장 완료: {plot_file}")
        except Exception as e:
            logger.warning(f"비교 분석 그래프 생성 실패: {e}")
        
        return analysis_data

    def _extract_original_image_name(self, filename: str) -> str:
        """
        파일 이름에서 원본 이미지 이름 추출
        전처리된 이미지 파일 이름에서 원본 이미지 파일 이름을 추출합니다.
        
        Args:
            filename: 이미지 파일 이름
            
        Returns:
            원본 이미지 파일 이름
        """
        # 파일 확장자 제거
        base_name = os.path.splitext(filename)[0]
        
        # 전처리 접미사 제거 (예: _preprocessed, _cropped 등)
        for suffix in ['_preprocessed', '_cropped', '_masked', '_enhanced', '_augmented', '_filtered']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # 파일 확장자 복원
        ext = os.path.splitext(filename)[1]
        original_name = base_name + ext
        
        return original_name
    
    def _load_box_data(self, image_name: str) -> Dict:
        """
        이미지의 box 정보 로드
        
        Args:
            image_name: 이미지 파일 이름
            
        Returns:
            box 데이터 또는 빈 딕셔너리
        """
        # box 디렉토리 경로
        box_dir = os.path.join(self.category_path, "3.box")
        if not os.path.exists(box_dir):
            return {}
        
        # 이미지 이름에서 확장자 제거
        base_name = os.path.splitext(image_name)[0]
        
        # box 정보 파일 경로
        box_file = os.path.join(box_dir, f"{base_name}.json")
        if not os.path.exists(box_file):
            return {}
        
        # box 정보 로드
        try:
            with open(box_file, 'r') as f:
                box_data = json.load(f)
            return box_data
        except Exception as e:
            logger.warning(f"Box 정보 로드 실패: {box_file}, 오류: {e}")
            return {}
    
    def _load_mask_data(self, image_name: str) -> Dict:
        """
        이미지의 mask/polygon 정보 로드
        
        Args:
            image_name: 이미지 파일 이름
            
        Returns:
            mask 데이터 또는 빈 딕셔너리
        """
        # mask 디렉토리 경로
        mask_dir = os.path.join(self.category_path, "4.mask")
        if not os.path.exists(mask_dir):
            return {}
        
        # 이미지 이름에서 확장자 제거
        base_name = os.path.splitext(image_name)[0]
        
        # mask 정보 파일 경로 (JSON 또는 다른 형식)
        mask_file = os.path.join(mask_dir, f"{base_name}.json")
        if not os.path.exists(mask_file):
            # JSON이 없으면 mask 이미지 파인
            mask_image = os.path.join(mask_dir, f"{base_name}_mask.png")
            if os.path.exists(mask_image):
                return {"mask_image": mask_image}
            return {}
        
        # mask 정보 로드
        try:
            with open(mask_file, 'r') as f:
                mask_data = json.load(f)
            return mask_data
        except Exception as e:
            logger.warning(f"Mask 정보 로드 실패: {mask_file}, 오류: {e}")
            return {}

    def sync_annotations(self, experiment_id: str, save_by_class: bool = True):
        """
        특정 실험 결과에 원본 annotation 정보 동기화
        
        Args:
            experiment_id: 동기화할 실험 ID (예: "shot_5_threshold_0.75")
            save_by_class: 클래스별로 annotation 정보를 저장할지 여부
        """
        # 실험 ID에서 shot과 threshold 추출
        parts = experiment_id.split('_')
        n_shot = int(parts[1])
        threshold = float(parts[3])
        
        # 실험 결과 디렉토리 경로
        experiment_dir = os.path.join(self.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
        predictions_file = os.path.join(experiment_dir, "predictions.csv")
        
        if not os.path.exists(predictions_file):
            logger.error(f"예측 결과 파일이 없습니다: {predictions_file}")
            return
        
        # 예측 결과 로드
        predictions_df = pd.read_csv(predictions_file)
        
        # annotation 정보 동기화
        annotations = []
        
        # 클래스별 annotation 모음 (클래스별 저장 시 사용)
        class_annotations = {}
        
        for _, row in tqdm(predictions_df.iterrows(), total=len(predictions_df), desc="Annotation 동기화"):
            image_path = row["image_path"]
            image_filename = row["image_filename"]
            predicted_class = row["predicted_class"]
            original_class = row.get("original_class", "Unknown")
            
            # 원본 이미지 파일명 추출 (전처리 접미사 제거)
            original_image_name = self._extract_original_image_name(image_filename)
            
            # 원본 box 정보 로드
            box_data = self._load_box_data(original_image_name)
            
            # 원본 mask/polygon 정보 로드
            mask_data = self._load_mask_data(original_image_name)
            
            # annotation 정보 생성
            annotation = {
                "image_filename": image_filename,
                "original_image_name": original_image_name,
                "predicted_class": predicted_class,
                "original_class": original_class,
                "similarity": row.get("similarity", 0.0),
                "box_data": box_data,
                "mask_data": mask_data,
                "image_path": image_path
            }
            annotations.append(annotation)
            
            # 클래스별 annotation 모음
            if save_by_class:
                if predicted_class not in class_annotations:
                    class_annotations[predicted_class] = []
                class_annotations[predicted_class].append(annotation)
        
        # 전체 annotation 정보 저장
        output_file = os.path.join(experiment_dir, "annotated_predictions.json")
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Annotation 정보 동기화 완료: {output_file}")
        
        # 클래스별 annotation 정보 저장
        if save_by_class:
            annotations_dir = os.path.join(experiment_dir, "annotations_by_class")
            os.makedirs(annotations_dir, exist_ok=True)
            
            for class_name, class_anns in class_annotations.items():
                class_file = os.path.join(annotations_dir, f"{class_name.replace(' ', '_')}_annotations.json")
                with open(class_file, 'w') as f:
                    json.dump(class_anns, f, indent=2, cls=NumpyEncoder)
                logger.info(f"클래스 '{class_name}'의 annotation 정보 저장 완료: {class_file}")
        
        # annotation 요약 정보 저장
        summary = {
            "total_annotations": len(annotations),
            "annotations_by_class": {cls: len(anns) for cls, anns in class_annotations.items()},
            "experiment_id": experiment_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        summary_file = os.path.join(experiment_dir, "annotation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Annotation 요약 정보 저장 완료: {summary_file}")
        
        return annotations

    def analyze_ground_truth(self, ground_truth_dir: str = None, save_annotations: bool = True):
        """
        Ground truth 디렉토리의 annotation 정보 분석
        
        Args:
            ground_truth_dir: Ground truth 디렉토리 경로 (기본값: <category>/7.results/ground_truth)
            save_annotations: 분석된 annotation 정보를 저장할지 여부
            
        Returns:
            Ground truth annotation 분석 결과
        """
        # Ground truth 디렉토리 확인
        if ground_truth_dir is None:
            ground_truth_dir = os.path.join(self.base_results_dir, "ground_truth")
        
        if not os.path.exists(ground_truth_dir):
            logger.error(f"Ground truth 디렉토리를 찾을 수 없음: {ground_truth_dir}")
            return None
        
        logger.info(f"Ground truth 디렉토리 분석: {ground_truth_dir}")
        
        # 클래스별 이미지 및 annotation 정보
        class_annotations = {}
        total_images = 0
        
        # 각 클래스 폴더 처리
        for class_dir in os.listdir(ground_truth_dir):
            class_path = os.path.join(ground_truth_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
                
            # 클래스 이름 정규화
            if class_dir.lower().startswith('class_'):
                # class_0, class_1 등의 형식인 경우 Class_X 형식으로 통일
                class_num = class_dir.split('_')[1]
                normalized_class_name = f"Class_{class_num}"
            elif class_dir.lower().startswith('unknown_') and self.group_unknown:
                # unknown_egifence, unknown_human, unknown_road, unknown_none 등은 
                # 모두 하나의 "Unknown" 클래스로 통합 (그룹화 옵션이 활성화된 경우)
                normalized_class_name = "Unknown"
                logger.info(f"Unknown 하위 카테고리 '{class_dir}'를 'Unknown' 클래스로 통합합니다.")
            else:
                # 기타 다른 형식의 폴더는 그대로 사용
                normalized_class_name = class_dir
            
            # 이 클래스의 annotation 모음
            if normalized_class_name not in class_annotations:
                class_annotations[normalized_class_name] = []
            
            # 이 클래스의 모든 이미지 파일 처리
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # 원본 이미지 파일명 추출
                    original_image_name = self._extract_original_image_name(file_name)
                    
                    # 원본 box 정보 로드
                    box_data = self._load_box_data(original_image_name)
                    
                    # 원본 mask/polygon 정보 로드
                    mask_data = self._load_mask_data(original_image_name)
                    
                    # annotation 정보 생성
                    annotation = {
                        "image_filename": file_name,
                        "original_image_name": original_image_name,
                        "ground_truth_class": normalized_class_name,
                        "original_folder": class_dir,  # 원본 폴더 이름도 저장
                        "box_data": box_data,
                        "mask_data": mask_data,
                        "image_path": os.path.join(class_path, file_name)
                    }
                    class_annotations[normalized_class_name].append(annotation)
                    total_images += 1
            
            logger.info(f"클래스 '{normalized_class_name}' ({class_dir}): {len(class_annotations[normalized_class_name])}개 이미지 처리됨")
        
        # 분석 결과
        analysis_result = {
            "total_images": total_images,
            "classes": list(class_annotations.keys()),
            "images_per_class": {cls: len(anns) for cls, anns in class_annotations.items()},
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Ground truth annotation 정보 저장
        if save_annotations:
            annotations_dir = os.path.join(ground_truth_dir, "annotations")
            os.makedirs(annotations_dir, exist_ok=True)
            
            # 전체 annotation 목록
            all_annotations = []
            for cls_anns in class_annotations.values():
                all_annotations.extend(cls_anns)
            
            # 전체 annotation 저장
            all_file = os.path.join(annotations_dir, "all_annotations.json")
            with open(all_file, 'w') as f:
                json.dump(all_annotations, f, indent=2, cls=NumpyEncoder)
            
            # 클래스별 annotation 저장
            for class_name, class_anns in class_annotations.items():
                class_file = os.path.join(annotations_dir, f"{class_name.replace(' ', '_')}_annotations.json")
                with open(class_file, 'w') as f:
                    json.dump(class_anns, f, indent=2, cls=NumpyEncoder)
            
            # 분석 요약 저장
            summary_file = os.path.join(annotations_dir, "ground_truth_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            logger.info(f"Ground truth annotation 정보 저장 완료: {annotations_dir}")
        
        return analysis_result
    
    def compare_annotations(self, experiment_id: str, ground_truth_dir: str = None):
        """
        예측 결과와 ground truth의 annotation 정보 비교
        
        Args:
            experiment_id: 비교할 실험 ID
            ground_truth_dir: Ground truth 디렉토리 경로
            
        Returns:
            비교 결과 딕셔너리
        """
        # 실험 ID에서 shot과 threshold 추출
        parts = experiment_id.split('_')
        n_shot = int(parts[1])
        threshold = float(parts[3])
        
        # 실험 결과 디렉토리 경로
        experiment_dir = os.path.join(self.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
        predictions_file = os.path.join(experiment_dir, "predictions.csv")
        annotations_file = os.path.join(experiment_dir, "annotated_predictions.json")
        
        if not os.path.exists(predictions_file):
            logger.error(f"예측 결과 파일이 없습니다: {predictions_file}")
            return None
        
        if not os.path.exists(annotations_file):
            # annotation 정보 동기화 시도
            logger.info(f"Annotation 정보 파일이 없어 동기화를 시도합니다.")
            annotations = self.sync_annotations(experiment_id)
            if not annotations:
                logger.error("Annotation 동기화 실패")
                return None
        else:
            # 기존 annotation 정보 로드
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
        
        # Ground truth 디렉토리 확인 - 모델과 상관없이 공통 ground truth 사용
        if ground_truth_dir is None:
            ground_truth_dir = os.path.join(self.base_results_dir, "ground_truth")
        
        if not os.path.exists(ground_truth_dir):
            logger.error(f"Ground truth 디렉토리를 찾을 수 없음: {ground_truth_dir}")
            return None
        
        # Ground truth 매핑 생성
        ground_truth_mapping = self._create_ground_truth_mapping(ground_truth_dir)
        
        if not ground_truth_mapping:
            logger.error("Ground truth 매핑을 생성할 수 없습니다.")
            return None
        
        # 파일명 기준으로 예측 결과와 ground truth 비교
        comparison_results = []
        match_count = 0
        mismatch_count = 0
        missing_gt_count = 0
        
        # 클래스별 통계 저장 변수
        class_stats = {}
        
        for annotation in annotations:
            image_filename = annotation["image_filename"]
            original_image_name = annotation.get("original_image_name", image_filename)
            predicted_class = annotation["predicted_class"]
            confidence = annotation["similarity"]
            
            # Ground truth 클래스 찾기
            gt_info = ground_truth_mapping.get(original_image_name) or ground_truth_mapping.get(image_filename)
            
            if gt_info:
                ground_truth_class = gt_info["ground_truth_class"]
                
                # 두 클래스가 모두 Unknown인 경우 매치로 처리
                if predicted_class.lower() == "unknown" and ground_truth_class.lower() == "unknown":
                    match = True
                # 일반적인 클래스 매치 확인 (대소문자 무시)
                else:
                    # 정규화: 양쪽 다 소문자로 변환하고 비교
                    pred_normalized = predicted_class.lower()
                    gt_normalized = ground_truth_class.lower()
                    
                    # class_0와 Class_0 같은 형식을 동일하게 처리
                    if pred_normalized.startswith("class_") and gt_normalized.startswith("class_"):
                        pred_num = predicted_class.split("_")[1]
                        gt_num = ground_truth_class.split("_")[1]
                        match = pred_num == gt_num
                    else:
                        match = pred_normalized == gt_normalized
                
                if match:
                    match_count += 1
                else:
                    mismatch_count += 1
                
                # 클래스별 통계 업데이트
                if ground_truth_class not in class_stats:
                    class_stats[ground_truth_class] = {
                        "total": 0,
                        "correct": 0,
                        "incorrect": 0,
                        "predicted_as": {}
                    }
                
                class_stats[ground_truth_class]["total"] += 1
                if match:
                    class_stats[ground_truth_class]["correct"] += 1
                else:
                    class_stats[ground_truth_class]["incorrect"] += 1
                    # 어떤 클래스로 잘못 예측되었는지 추적
                    if predicted_class not in class_stats[ground_truth_class]["predicted_as"]:
                        class_stats[ground_truth_class]["predicted_as"][predicted_class] = 0
                    class_stats[ground_truth_class]["predicted_as"][predicted_class] += 1
                
                comparison_results.append({
                    "image_filename": image_filename,
                    "original_image_name": original_image_name,
                    "predicted_class": predicted_class,
                    "ground_truth_class": ground_truth_class,
                    "confidence": confidence,
                    "match": match,
                    "image_path": annotation.get("image_path", ""),
                    "ground_truth_path": gt_info.get("image_path", "")
                })
            else:
                missing_gt_count += 1
                
                comparison_results.append({
                    "image_filename": image_filename,
                    "original_image_name": original_image_name,
                    "predicted_class": predicted_class,
                    "ground_truth_class": "Unknown",
                    "confidence": confidence,
                    "match": False,
                    "image_path": annotation.get("image_path", ""),
                    "ground_truth_path": ""
                })
        
        # 각 클래스별 정확도 계산
        for cls, stats in class_stats.items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
            else:
                stats["accuracy"] = 0.0
        
        # 결과 요약
        total_predictions = len(annotations)
        accuracy = match_count / (match_count + mismatch_count) if (match_count + mismatch_count) > 0 else 0
        
        summary = {
            "experiment_id": experiment_id,
            "model": self.model_name,  # 모델 정보 추가
            "total_predictions": total_predictions,
            "match_count": match_count,
            "mismatch_count": mismatch_count,
            "missing_ground_truth_count": missing_gt_count,
            "accuracy": accuracy,
            "class_stats": class_stats,  # 클래스별 통계 추가
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 결과 저장
        comparison_dir = os.path.join(experiment_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 상세 비교 결과 저장
        comparison_file = os.path.join(comparison_dir, "comparison_results.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, cls=NumpyEncoder)
        
        # 요약 저장
        summary_file = os.path.join(comparison_dir, "comparison_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 모델별 요약 결과를 통합 폴더에도 저장
        model_summary_dir = os.path.join(self.base_results_dir, "model_comparison")
        os.makedirs(model_summary_dir, exist_ok=True)
        model_summary_file = os.path.join(model_summary_dir, f"{self.model_name}_{experiment_id}_summary.json")
        with open(model_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 클래스별 성능 로깅
        logger.info(f"비교 결과 저장 완료:")
        logger.info(f"  - 상세 결과: {comparison_file}")
        logger.info(f"  - 요약 정보: {summary_file}")
        logger.info(f"  - 모델 요약: {model_summary_file}")
        logger.info(f"전체 정확도: {accuracy:.4f} ({match_count}/{match_count + mismatch_count})")
        
        # 클래스별 정확도 출력
        logger.info("클래스별 성능:")
        for cls, stats in class_stats.items():
            logger.info(f"  - {cls}: 정확도 {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
            if stats["incorrect"] > 0 and stats["predicted_as"]:
                misclassifications = sorted(stats["predicted_as"].items(), key=lambda x: x[1], reverse=True)
                misclass_str = ", ".join([f"{cls}: {count}" for cls, count in misclassifications[:3]])
                logger.info(f"    주요 오분류: {misclass_str}")
        
        return {
            "summary": summary,
            "detailed_results": comparison_results,
            "class_comparison": class_stats  # 클래스별 비교 결과 추가
        }

    def _create_ground_truth_mapping(self, ground_truth_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Ground truth 디렉토리에서 파일명 -> 클래스 매핑 생성
        
        Args:
            ground_truth_dir: Ground truth 디렉토리 경로
            
        Returns:
            파일명 -> {ground_truth_class, image_path} 매핑 사전
        """
        mapping = {}
        
        # Ground truth 디렉토리 내의 각 클래스 폴더 처리
        for class_dir in os.listdir(ground_truth_dir):
            class_path = os.path.join(ground_truth_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
                
            # 클래스 이름 정규화
            if class_dir.lower().startswith('class_'):
                # class_0, class_1 등의 형식인 경우 Class_X 형식으로 통일
                class_num = class_dir.split('_')[1]
                normalized_class_name = f"Class_{class_num}"
            elif class_dir.lower().startswith('unknown_') and self.group_unknown:
                # unknown_egifence, unknown_human, unknown_road, unknown_none 등은 
                # 모두 하나의 "Unknown" 클래스로 통합 (그룹화 옵션이 활성화된 경우)
                normalized_class_name = "Unknown"
                logger.info(f"Unknown 하위 카테고리 '{class_dir}'를 'Unknown' 클래스로 통합합니다.")
            else:
                # 기타 다른 형식의 폴더는 그대로 사용
                normalized_class_name = class_dir
            
            # 이 클래스의 모든 이미지 파일 처리
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = os.path.join(class_path, file_name)
                    mapping[file_name] = {
                        "ground_truth_class": normalized_class_name,
                        "image_path": img_path
                    }
        
        # 클래스별 이미지 수 로깅
        class_counts = {}
        for img_info in mapping.values():
            cls = img_info["ground_truth_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            
        for cls, count in class_counts.items():
            logger.info(f"클래스 '{cls}': {count}개 이미지")
            
        logger.info(f"Ground truth 매핑 생성 완료: {len(mapping)}개 이미지, {len(class_counts)}개 클래스")
        return mapping

    def generate_comprehensive_report(self):
        """
        모든 실험 결과를 취합하여 종합 보고서(CSV) 및 시각화 자료를 생성합니다.
        """
        logger.info("종합 보고서 생성을 시작합니다.")
        report_data = []
        
        # 모든 shot/threshold 조합을 순회
        for n_shot in self.n_shots:
            for threshold in self.thresholds:
                experiment_id = f"shot_{n_shot}_threshold_{threshold:.2f}"
                eval_file = os.path.join(self.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}", "evaluation.json")

                if not os.path.exists(eval_file):
                    logger.warning(f"평가 파일 없음: {eval_file}. 이 실험은 건너뜁니다.")
                    continue
                
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                summary = eval_data.get("summary", {})
                class_stats = eval_data.get("class_statistics", {})
                
                y_true = eval_data.get("y_true", [])
                y_pred = eval_data.get("y_pred", [])
                
                if not y_true or not y_pred:
                    logger.warning(f"평가 데이터에 y_true 또는 y_pred가 없습니다: {eval_file}")
                    continue

                # 전체 성능 지표 계산
                overall_accuracy = accuracy_score(y_true, y_pred)
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # 데이터 행 생성
                row = {
                    "shot": n_shot,
                    "threshold": threshold,
                    "overall_accuracy": overall_accuracy,
                    "balanced_accuracy": balanced_acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "classified_ratio": summary.get("classified_ratio", 0),
                    "unknown_ratio": summary.get("unknown_ratio", 0),
                    "total_predictions": summary.get("total_predictions", 0),
                    "correct_predictions": summary.get("correct_predictions", 0),
                    "incorrect_predictions": summary.get("incorrect_predictions", 0),
                }
                
                # 클래스별 성능 지표 추가
                all_classes = sorted(list(class_stats.keys()))
                for cls in all_classes:
                    stats = class_stats[cls]
                    row[f"{cls}_accuracy"] = stats.get("accuracy", 0)
                    row[f"{cls}_precision"] = stats.get("precision", 0)
                    row[f"{cls}_recall"] = stats.get("recall", 0)
                    row[f"{cls}_f1_score"] = stats.get("f1_score", 0)
                    row[f"{cls}_total"] = stats.get("total", 0)
                
                report_data.append(row)

        if not report_data:
            logger.error("처리할 평가 데이터가 없습니다. 보고서 생성을 중단합니다.")
            return

        # 데이터프레임 생성 및 저장
        df = pd.DataFrame(report_data)
        report_path = os.path.join(self.results_dir, "comprehensive_report.csv")
        df.to_csv(report_path, index=False, float_format='%.4f')
        logger.info(f"종합 보고서가 CSV 파일로 저장되었습니다: {report_path}")

        # 시각화 생성
        self.generate_visualizations(df)

    def generate_visualizations(self, df: pd.DataFrame):
        """
        종합 보고서 데이터프레임을 기반으로 다양한 시각화 자료를 생성합니다.
        
        Args:
            df: 종합 보고서 데이터프레임
        """
        vis_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        logger.info(f"시각화 자료를 다음 디렉토리에 저장합니다: {vis_dir}")

        # 1. 성능 히트맵 (Accuracy, F1-Score)
        metrics_to_plot = ["overall_accuracy", "f1_weighted", "balanced_accuracy", "classified_ratio"]
        for metric in metrics_to_plot:
            try:
                pivot_df = df.pivot(index="shot", columns="threshold", values=metric)
                plt.figure(figsize=(14, 8))
                sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
                plt.title(f"{self.model_name.upper()}: {metric.replace('_', ' ').title()} by Shot and Threshold")
                plt.xlabel("Similarity Threshold")
                plt.ylabel("N-Shot")
                save_path = os.path.join(vis_dir, f"heatmap_{metric}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"성능 히트맵 저장 완료: {save_path}")
            except Exception as e:
                logger.error(f"히트맵 생성 중 오류 ({metric}): {e}")

        # 2. Threshold에 따른 성능 변화 그래프
        plt.figure(figsize=(15, 10))
        metrics = ["overall_accuracy", "f1_weighted", "precision_weighted", "recall_weighted"] # weighted로 변경
        
        # precision/recall weighted 계산 (데이터가 없을 수 있으므로 확인)
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            from sklearn.metrics import precision_score, recall_score
            df['precision_weighted'] = df.apply(lambda row: precision_score(row['y_true'], row['y_pred'], average='weighted', zero_division=0), axis=1)
            df['recall_weighted'] = df.apply(lambda row: recall_score(row['y_true'], row['y_pred'], average='weighted', zero_division=0), axis=1)
        else:
             # 임시로 f1으로 대체하거나, 계산할 수 없음을 알림
            if 'f1_weighted' in df.columns:
                df['precision_weighted'] = df['f1_weighted']
                df['recall_weighted'] = df['f1_weighted']
            else:
                logger.warning("Precision/Recall을 계산할 데이터가 없어 f1으로 대체합니다.")


        for i, metric in enumerate(metrics, 1):
            if metric not in df.columns: continue
            plt.subplot(2, 2, i)
            sns.lineplot(data=df, x="threshold", y=metric, hue="shot", marker="o", palette="tab10")
            plt.title(f"{metric.replace('_', ' ').title()} vs. Threshold")
            plt.xlabel("Similarity Threshold")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title="N-Shot")
        
        plt.suptitle(f"{self.model_name.upper()}: Performance Metrics vs. Threshold", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(vis_dir, "lineplot_performance_metrics.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"성능 변화 그래프 저장 완료: {save_path}")

        # 3. Unknown 비율 변화 그래프
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="threshold", y="unknown_ratio", hue="shot", marker="o", palette="tab10")
        plt.title(f"{self.model_name.upper()}: Unknown Classification Ratio vs. Threshold")
        plt.xlabel("Similarity Threshold")
        plt.ylabel("Ratio of Unknown Classifications")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="N-Shot")
        save_path = os.path.join(vis_dir, "lineplot_unknown_ratio.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Unknown 비율 그래프 저장 완료: {save_path}")

        # 4. 최고 성능 실험의 Confusion Matrix
        if 'f1_weighted' in df.columns:
            best_experiment = df.loc[df['f1_weighted'].idxmax()]
            best_shot = int(best_experiment['shot'])
            best_threshold = best_experiment['threshold']
            
            logger.info(f"최고 성능(F1-Weighted) 실험: shot={best_shot}, threshold={best_threshold:.2f}")

            eval_file = os.path.join(self.results_dir, f"shot_{best_shot}", f"threshold_{best_threshold:.2f}", "evaluation.json")
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                y_true = eval_data.get("y_true")
                y_pred = eval_data.get("y_pred")
                labels = sorted(list(set(y_true) | set(y_pred)))
                
                if y_true and y_pred:
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
                    plt.title(f"{self.model_name.upper()}: Confusion Matrix (Best - Shot: {best_shot}, Threshold: {best_threshold:.2f})")
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    save_path = os.path.join(vis_dir, "confusion_matrix_best_experiment.png")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"최고 성능 실험의 혼동 행렬 저장 완료: {save_path}")

class NumpyEncoder(json.JSONEncoder):
    """NumPy 배열을 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Few-Shot Learning Experiment Platform",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Execution examples:
1. Run all experiments for a category:
   python %(prog)s --category test_category --model resnet

2. Evaluate a specific experiment:
   python %(prog)s --evaluate --experiment-id shot_5_threshold_0.70 --category test_category --model resnet

3. Sync annotations for an experiment:
   python %(prog)s --sync-annotations --experiment-id shot_5_threshold_0.70 --category test_category --model resnet

4. Analyze ground truth:
   python %(prog)s --analyze-ground-truth --category test_category --model resnet
   
5. Compare predictions with ground truth:
   python %(prog)s --compare-annotations --experiment-id shot_5_threshold_0.70 --category test_category --model resnet

6. Generate comprehensive report for existing results:
   python %(prog)s --report-only --category test_category --model resnet
"""
    )
    
    # 기본 인수
    parser.add_argument("--category", type=str, required=True, help="Category name for the experiment")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "clip", "dino"], help="Model name for feature extraction")
    
    # 실행 제어 인수
    parser.add_argument("--run-all", action="store_true", help="Run all experiments (default action if no other flag is provided)")
    parser.add_argument("--report-only", action="store_true", help="Generate comprehensive report from existing results without running experiments")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a specific experiment")
    parser.add_argument("--sync-annotations", action="store_true", help="Sync annotations for a specific experiment")
    parser.add_argument("--analyze-ground-truth", action="store_true", help="Analyze ground truth data")
    parser.add_argument("--compare-annotations", action="store_true", help="Compare annotations with ground truth")
    
    # 실험 설정 인수
    parser.add_argument("--shots", type=str, help="Specific shot count(s) to run experiments")
    parser.add_argument("--thresholds", type=str, help="Specific threshold(s) to run experiments")
    parser.add_argument("--save-images", action="store_true", help="Save classified images to class folders")
    parser.add_argument("--group-unknown", action="store_true", default=True, help="Group all 'unknown_*' categories into a single 'Unknown' class during analysis")
    parser.add_argument("--no-group-unknown", action="store_false", dest="group_unknown", help="Do not group 'unknown_*' categories into a single 'Unknown' class")
    
    # 실험 결과 관리 인수
    parser.add_argument("--experiment-id", help="Specific experiment ID to evaluate or sync annotations")
    parser.add_argument("--output-dir", help="Output directory for experiment results")
    parser.add_argument("--input-dir", help="Input directory for images")
    parser.add_argument("--ground-truth-dir", help="Ground truth directory")
    
    # 기타 인수
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-annotations", action="store_true", help="Save analyzed ground truth annotation information")
    
    return parser.parse_args()

def main():
    """Main function to run the experiment."""
    args = parse_args()

    # 로거 레벨 설정
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 실험 객체 초기화
    experiment = FewShotExperiment(
        category_name=args.category,
        model_name=args.model,
        save_images=args.save_images,
        group_unknown=args.group_unknown
    )

    # shot/threshold 값 설정
    if args.shots:
        experiment.n_shots = [int(s.strip()) for s in args.shots.split(',')]
    if args.thresholds:
        experiment.thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    # 인수에 따른 작업 분기
    if args.evaluate:
        if not args.experiment_id:
            logger.error("--evaluate requires --experiment-id")
            return
        experiment.evaluate_experiment(args.experiment_id)
        
    elif args.sync_annotations:
        if not args.experiment_id:
            logger.error("--sync-annotations requires --experiment-id")
            return
        experiment.sync_annotations(args.experiment_id, args.save_by_class)
        
    elif args.analyze_ground_truth:
        experiment.analyze_ground_truth(args.ground_truth_dir, args.save_annotations)
        
    elif args.compare_annotations:
        if not args.experiment_id:
            logger.error("--compare-annotations requires --experiment-id")
            return
        experiment.compare_annotations(args.experiment_id, args.ground_truth_dir)

    elif args.report_only:
        logger.info(f"[{args.model}] 모델의 기존 결과로 종합 보고서 생성을 시작합니다.")
        experiment.generate_comprehensive_report()
        logger.info(f"[{args.model}] 모델의 종합 보고서 생성이 완료되었습니다.")
        
    else:
        # --run-all 또는 다른 플래그가 없을 경우 기본적으로 전체 실험 실행
        logger.info(f"[{args.model}] 모델의 전체 실험을 시작합니다.")
        experiment.run_all_experiments(args.input_dir)
        logger.info(f"[{args.model}] 모델의 전체 실험이 완료되었습니다.")


if __name__ == "__main__":
    main() 