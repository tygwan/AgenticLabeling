#!/usr/bin/env python3
"""
폴더 기반 Ground Truth 평가 도구

이 스크립트는 수동으로 분류된 Ground Truth 폴더에서 정보를 읽고,
다양한 shot*threshold 조합의 성능을 평가합니다.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import seaborn as sns

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ground_truth_eval.log")
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
    parser = argparse.ArgumentParser(description="폴더 기반 Ground Truth 평가 도구")
    parser.add_argument("--category", type=str, required=True, help="평가할 카테고리 (예: test_category)")
    parser.add_argument("--ground-truth-dir", type=str, default=None, 
                        help="Ground Truth 폴더 경로 (기본값: data/{category}/7.results/ground_truth)")
    parser.add_argument("--output", type=str, default=None,
                        help="평가 결과 저장 경로 (기본값: data/{category}/7.results/evaluation_report.json)")
    parser.add_argument("--visualize", action="store_true", help="결과 시각화 활성화")
    parser.add_argument("--shot-values", type=str, default="1,5,10,30", 
                        help="평가할 shot 값들 (쉼표로 구분)")
    parser.add_argument("--threshold-values", type=str, default="0.5,0.6,0.7,0.8,0.9",
                        help="평가할 threshold 값들 (쉼표로 구분)")
    return parser.parse_args()

class GroundTruthEvaluator:
    def __init__(self, category, ground_truth_dir=None, output_path=None):
        """초기화"""
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.category = category
        self.category_dir = self.data_dir / category
        
        # Ground Truth 디렉토리 설정
        if ground_truth_dir:
            self.ground_truth_dir = Path(ground_truth_dir)
        else:
            self.ground_truth_dir = self.category_dir / "7.results" / "ground_truth"
        
        # 출력 경로 설정
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.category_dir / "7.results" / "evaluation_report.json"
        
        # 결과 저장소
        self.ground_truth = {}
        self.experiment_results = {}
        self.evaluation_results = {}
    
    def load_folder_ground_truth(self):
        """폴더 기반 Ground Truth 로드"""
        if not os.path.exists(self.ground_truth_dir):
            logger.error(f"Ground Truth 디렉토리가 존재하지 않습니다: {self.ground_truth_dir}")
            return False
        
        logger.info(f"Ground Truth 폴더에서 데이터 로드 중: {self.ground_truth_dir}")
        
        # Ground Truth 매핑 초기화
        self.ground_truth = {}
        
        # 클래스 폴더 순회
        for class_dir in self.ground_truth_dir.glob("class_*"):
            if class_dir.is_dir():
                class_name = class_dir.name
                logger.info(f"클래스 '{class_name}'에서 이미지 로드 중...")
                
                # 이미지 파일 처리
                for img_file in class_dir.glob("*.*"):
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img_name = img_file.name
                        self.ground_truth[img_name] = class_name
        
        logger.info(f"총 {len(self.ground_truth)} 개의 Ground Truth 이미지가 로드되었습니다.")
        
        # Ground Truth를 표준 JSON 형식으로 저장
        standard_gt_path = self.category_dir / "7.results" / "ground_truth.json"
        try:
            with open(standard_gt_path, 'w') as f:
                json.dump(self.ground_truth, f, indent=2)
            logger.info(f"Ground Truth를 {standard_gt_path}에 저장했습니다.")
        except Exception as e:
            logger.error(f"Ground Truth 저장 중 오류: {e}")
        
        return len(self.ground_truth) > 0
    
    def find_experiment_results(self, shot_values, threshold_values):
        """주어진 shot 및 threshold 값에 대한 실험 결과 찾기"""
        results_dir = self.category_dir / "7.results"
        
        for shot in shot_values:
            for threshold in threshold_values:
                # shot_N 및 threshold_X.Y 형식의 폴더 찾기
                pattern = f"shot_{shot}"
                shot_dir = results_dir / pattern
                
                if shot_dir.exists() and shot_dir.is_dir():
                    # 결과 파일 찾기
                    results_file = shot_dir / f"threshold_{threshold}" / "results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            exp_key = f"shot_{shot}_threshold_{threshold}"
                            self.experiment_results[exp_key] = results
                            logger.info(f"실험 결과 로드됨: {exp_key} ({len(results)} 이미지)")
                        except Exception as e:
                            logger.error(f"{results_file} 로드 중 오류: {e}")
        
        logger.info(f"총 {len(self.experiment_results)} 개의 실험 결과가 로드되었습니다.")
        return len(self.experiment_results) > 0
    
    def evaluate_experiments(self):
        """모든 실험 결과 평가"""
        if not self.ground_truth:
            logger.error("Ground Truth 데이터가 없습니다.")
            return False
        
        if not self.experiment_results:
            logger.error("실험 결과 데이터가 없습니다.")
            return False
        
        logger.info("실험 결과 평가 중...")
        
        for exp_key, results in self.experiment_results.items():
            # 공통 이미지만 평가
            common_images = set(self.ground_truth.keys()) & set(results.keys())
            
            if not common_images:
                logger.warning(f"{exp_key}: 공통 이미지가 없습니다.")
                continue
            
            # 예측 및 실제 레이블 추출
            y_true = []
            y_pred = []
            
            for img_name in common_images:
                true_class = self.ground_truth[img_name]
                pred_class = results[img_name].get("predicted_class", "unknown")
                y_true.append(true_class)
                y_pred.append(pred_class)
            
            # 평가 지표 계산
            accuracy = accuracy_score(y_true, y_pred)
            class_report = classification_report(y_true, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_true, y_pred, labels=list(set(y_true + y_pred)))
            
            # 결과 저장
            self.evaluation_results[exp_key] = {
                "accuracy": accuracy,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "labels": list(set(y_true + y_pred)),
                "num_evaluated_images": len(common_images)
            }
            
            logger.info(f"{exp_key}: 정확도 {accuracy:.4f}, 평가된 이미지 {len(common_images)} 개")
        
        # 결과 저장
        try:
            with open(self.output_path, 'w') as f:
                # confusion_matrix는 numpy array이므로 JSON으로 변환 필요
                json_results = {k: v for k, v in self.evaluation_results.items()}
                json.dump(json_results, f, indent=2)
            logger.info(f"평가 결과를 {self.output_path}에 저장했습니다.")
        except Exception as e:
            logger.error(f"평가 결과 저장 중 오류: {e}")
        
        return True
    
    def visualize_results(self):
        """평가 결과 시각화"""
        if not self.evaluation_results:
            logger.error("시각화할 평가 결과가 없습니다.")
            return False
        
        logger.info("평가 결과 시각화 중...")
        
        # 정확도 비교 그래프
        plt.figure(figsize=(12, 6))
        
        # shot별로 그룹화하여 시각화
        shot_groups = {}
        for exp_key, results in self.evaluation_results.items():
            # shot_N_threshold_X.Y 형식에서 shot과 threshold 추출
            parts = exp_key.split('_')
            shot = parts[1]
            threshold = parts[3]
            
            if shot not in shot_groups:
                shot_groups[shot] = []
            
            shot_groups[shot].append((threshold, results['accuracy']))
        
        for shot, values in shot_groups.items():
            thresholds, accuracies = zip(*sorted(values, key=lambda x: float(x[0])))
            plt.plot(thresholds, accuracies, marker='o', label=f"Shot {shot}")
        
        plt.title('Shot 및 Threshold별 정확도')
        plt.xlabel('Threshold')
        plt.ylabel('정확도')
        plt.grid(True)
        plt.legend()
        
        # 그래프 저장
        plt.tight_layout()
        output_img = self.category_dir / "7.results" / "accuracy_comparison.png"
        plt.savefig(output_img)
        logger.info(f"정확도 비교 그래프를 {output_img}에 저장했습니다.")
        
        # 최적의 조합 찾기
        best_exp = max(self.evaluation_results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"최적의 조합: {best_exp[0]}, 정확도: {best_exp[1]['accuracy']:.4f}")
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(10, 8))
        conf_matrix = np.array(best_exp[1]['confusion_matrix'])
        labels = best_exp[1]['labels']
        
        df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(f'혼동 행렬 - {best_exp[0]}')
        plt.ylabel('실제 클래스')
        plt.xlabel('예측 클래스')
        
        # 혼동 행렬 저장
        confusion_img = self.category_dir / "7.results" / "best_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(confusion_img)
        logger.info(f"최적 조합의 혼동 행렬을 {confusion_img}에 저장했습니다.")
        
        return True
    
    def generate_report(self):
        """평가 보고서 생성"""
        if not self.evaluation_results:
            logger.error("보고서를 생성할 평가 결과가 없습니다.")
            return False
        
        logger.info("평가 보고서 생성 중...")
        
        # 보고서 데이터 준비
        report_data = {
            "summary": {
                "total_ground_truth_images": len(self.ground_truth),
                "total_experiments_evaluated": len(self.evaluation_results),
                "best_experiment": None,
                "best_accuracy": 0.0
            },
            "experiment_details": {}
        }
        
        # 최적 실험 찾기 및 실험별 요약 생성
        for exp_key, results in self.evaluation_results.items():
            accuracy = results['accuracy']
            
            # 최적 실험 업데이트
            if accuracy > report_data["summary"]["best_accuracy"]:
                report_data["summary"]["best_accuracy"] = accuracy
                report_data["summary"]["best_experiment"] = exp_key
            
            # 실험 요약 추가
            report_data["experiment_details"][exp_key] = {
                "accuracy": accuracy,
                "num_evaluated_images": results["num_evaluated_images"],
                "class_metrics": {}
            }
            
            # 클래스별 메트릭스 추가
            for class_name, metrics in results["classification_report"].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_data["experiment_details"][exp_key]["class_metrics"][class_name] = {
                        "precision": metrics['precision'],
                        "recall": metrics['recall'],
                        "f1-score": metrics['f1-score'],
                        "support": metrics['support']
                    }
        
        # 보고서 저장
        report_path = self.category_dir / "7.results" / "evaluation_summary.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"평가 보고서를 {report_path}에 저장했습니다.")
        except Exception as e:
            logger.error(f"평가 보고서 저장 중 오류: {e}")
        
        # 텍스트 보고서 생성
        report_text = f"""
=== Ground Truth 평가 보고서 ===

카테고리: {self.category}
총 Ground Truth 이미지: {report_data['summary']['total_ground_truth_images']}
평가된 실험 수: {report_data['summary']['total_experiments_evaluated']}

최적 실험: {report_data['summary']['best_experiment']}
최적 정확도: {report_data['summary']['best_accuracy']:.4f}

== 실험별 정확도 ==
"""
        
        # 정확도별로 정렬된 실험 목록
        sorted_exps = sorted(
            report_data["experiment_details"].items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )
        
        for exp_key, details in sorted_exps:
            report_text += f"{exp_key}: {details['accuracy']:.4f} ({details['num_evaluated_images']} 이미지)\n"
        
        # 최적 실험의 클래스별 성능
        best_exp = report_data["summary"]["best_experiment"]
        if best_exp:
            report_text += f"\n== 최적 실험 ({best_exp})의 클래스별 성능 ==\n"
            
            for class_name, metrics in report_data["experiment_details"][best_exp]["class_metrics"].items():
                report_text += f"{class_name}:\n"
                report_text += f"  Precision: {metrics['precision']:.4f}\n"
                report_text += f"  Recall: {metrics['recall']:.4f}\n"
                report_text += f"  F1-Score: {metrics['f1-score']:.4f}\n"
                report_text += f"  Support: {metrics['support']} 이미지\n\n"
        
        # 텍스트 보고서 저장
        text_report_path = self.category_dir / "7.results" / "evaluation_summary.txt"
        try:
            with open(text_report_path, 'w') as f:
                f.write(report_text)
            logger.info(f"텍스트 보고서를 {text_report_path}에 저장했습니다.")
        except Exception as e:
            logger.error(f"텍스트 보고서 저장 중 오류: {e}")
        
        return True

def main():
    """메인 함수"""
    args = parse_args()
    
    # shot 및 threshold 값 파싱
    shot_values = [int(s.strip()) for s in args.shot_values.split(",")]
    threshold_values = [float(t.strip()) for t in args.threshold_values.split(",")]
    
    logger.info(f"카테고리: {args.category}")
    logger.info(f"평가할 shot 값: {shot_values}")
    logger.info(f"평가할 threshold 값: {threshold_values}")
    
    # Ground Truth 평가기 초기화
    evaluator = GroundTruthEvaluator(
        args.category,
        args.ground_truth_dir,
        args.output
    )
    
    # Ground Truth 로드
    if not evaluator.load_folder_ground_truth():
        logger.error("Ground Truth 로드 실패")
        return 1
    
    # 실험 결과 찾기
    if not evaluator.find_experiment_results(shot_values, threshold_values):
        logger.error("실험 결과 로드 실패")
        return 1
    
    # 실험 평가
    if not evaluator.evaluate_experiments():
        logger.error("실험 평가 실패")
        return 1
    
    # 보고서 생성
    evaluator.generate_report()
    
    # 결과 시각화
    if args.visualize:
        evaluator.visualize_results()
    
    logger.info("평가 완료")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 