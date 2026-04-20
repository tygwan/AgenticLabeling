#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning 실험 결과 종합 분석 도구

이 스크립트는 여러 shot×threshold 조합에 대한 실험 결과를 분석하여
다음과 같은 평가 지표를 계산하고 CSV 파일로 출력합니다:
- 정확도(Accuracy)
- 균형 정확도(Balanced Accuracy)
- Macro F1-score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix (TP, TN, FP, FN)

사용법:
    python analyze_experiment_metrics.py --category test_category 
                                       --models resnet,dino
                                       --output results_metrics.csv
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import math
from typing import Dict, List, Tuple, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("experiment_metrics")

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="Few-Shot Learning 실험 결과 종합 분석 도구")
    parser.add_argument("--category", type=str, default="test_category",
                      help="데이터셋 카테고리 이름 (기본값: test_category)")
    parser.add_argument("--models", type=str, default="resnet,dino",
                      help="분석할 모델 (쉼표로 구분, 기본값: 'resnet,dino')")
    parser.add_argument("--output", type=str, default="experiment_metrics.csv",
                      help="출력 CSV 파일 경로 (기본값: 'experiment_metrics.csv')")
    parser.add_argument("--include-confusion", action="store_true",
                      help="각 클래스별 confusion matrix 정보도 CSV에 포함")
    return parser.parse_args()

def get_category_path(category: str) -> Path:
    """카테고리 디렉토리 경로 반환"""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_dir = script_dir.parent
    return project_dir / "data" / category

def load_experiment_results(category: str, models: List[str]) -> Dict:
    """모든 실험 결과 로드"""
    results = {}
    category_path = get_category_path(category)
    
    for model in models:
        model_dir = category_path / "7.results" / model
        if not model_dir.exists():
            logger.warning(f"{model} 모델의 결과 디렉토리가 존재하지 않습니다: {model_dir}")
            continue
        
        results[model] = {}
        
        # 모든 shot 디렉토리 탐색
        for shot_dir in model_dir.glob("shot_*"):
            if not shot_dir.is_dir():
                continue
            
            shot = int(shot_dir.name.split("_")[1])
            
            # 모든 threshold 디렉토리 탐색
            for threshold_dir in shot_dir.glob("threshold_*"):
                if not threshold_dir.is_dir():
                    continue
                
                threshold = float(threshold_dir.name.split("_")[1])
                
                # 비교 결과 파일 찾기
                comparison_file = threshold_dir / "comparison" / "comparison_summary.json"
                if not comparison_file.exists():
                    logger.warning(f"비교 결과 파일이 없습니다: {comparison_file}")
                    continue
                
                # 결과 로드
                with open(comparison_file, 'r') as f:
                    comparison_data = json.load(f)
                
                # 실험 식별자
                exp_id = f"shot_{shot}_threshold_{threshold:.2f}"
                results[model][exp_id] = comparison_data
    
    return results

def calculate_metrics(class_stats: Dict) -> Dict:
    """클래스별 통계를 바탕으로 여러 평가 지표 계산"""
    metrics = {}
    
    # 클래스 목록 (Unknown 제외)
    classes = [cls for cls in class_stats.keys() if cls.lower() != "unknown"]
    
    # 전체 클래스 수 (Unknown 포함)
    n_classes = len(class_stats)
    
    # 클래스별 TP, FP, FN, TN 계산
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    
    class_metrics = {}
    
    for cls in class_stats:
        # 현재 클래스에 대한 TP, FP, FN, TN 계산
        tp = class_stats[cls]["correct"]
        fn = class_stats[cls]["incorrect"]
        
        # 다른 클래스로 예측된 현재 클래스의 샘플 수 (FP)
        fp = 0
        for other_cls in class_stats:
            if other_cls == cls:
                continue
            
            pred_as = class_stats[other_cls].get("predicted_as", {})
            if cls in pred_as:
                fp += pred_as[cls]
        
        # 다른 클래스로 올바르게 예측된 샘플 수 (TN)
        tn = 0
        for other_cls in class_stats:
            if other_cls == cls:
                continue
            
            # 다른 클래스가 다른 클래스로 올바르게 예측된 경우 (현재 클래스로 예측되지 않은 경우)
            other_correct = class_stats[other_cls]["correct"]
            other_pred_as = class_stats[other_cls].get("predicted_as", {})
            other_incorrect_not_current = sum(other_pred_as.get(c, 0) for c in other_pred_as if c != cls)
            tn += other_correct + other_incorrect_not_current
        
        # 정밀도(Precision), 재현율(Recall), F1-score 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 클래스별 메트릭 저장
        class_metrics[cls] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
        
        # 전체 합계 업데이트
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn
    
    # 일반 정확도 (데이터셋에 제공된 accuracy)
    metrics["Accuracy"] = sum(class_stats[cls]["correct"] for cls in class_stats) / \
                        sum(class_stats[cls]["total"] for cls in class_stats)
    
    # 균형 정확도 (Balanced Accuracy)
    class_accuracies = []
    for cls in class_stats:
        class_accuracies.append(class_stats[cls]["accuracy"])
    
    metrics["Balanced_Accuracy"] = sum(class_accuracies) / len(class_accuracies)
    
    # Macro F1-score
    metrics["Macro_F1"] = sum(class_metrics[cls]["F1"] for cls in class_metrics) / len(class_metrics)
    
    # Matthews Correlation Coefficient (MCC)
    # Multi-class MCC는 복잡하므로 binary 케이스에 대한 계산을 수행 (Unknown vs. 나머지)
    if "Unknown" in class_stats:
        tp = class_stats["Unknown"]["correct"]
        fn = class_stats["Unknown"]["incorrect"]
        fp = sum(class_stats[cls]["predicted_as"].get("Unknown", 0) for cls in classes)
        tn = sum(class_stats[cls]["correct"] for cls in classes) + \
             sum(sum(v for k, v in class_stats[cls]["predicted_as"].items() if k != "Unknown") 
                 for cls in classes if "predicted_as" in class_stats[cls])
        
        numerator = (tp * tn) - (fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1
        metrics["MCC"] = numerator / denominator
    else:
        metrics["MCC"] = 0
    
    # Confusion Matrix 데이터
    metrics["TP_total"] = tp_total
    metrics["FP_total"] = fp_total
    metrics["FN_total"] = fn_total
    metrics["TN_total"] = tn_total
    
    # 클래스별 메트릭 추가
    metrics["Class_Metrics"] = class_metrics
    
    return metrics

def generate_metrics_table(results: Dict, include_confusion: bool = False) -> pd.DataFrame:
    """모든 실험 결과의 메트릭을 표 형태로 생성"""
    data = []
    
    for model in results:
        for exp_id, exp_data in results[model].items():
            # 기본 정보 추출
            shot = int(exp_id.split("_")[1])
            threshold = float(exp_id.split("_")[3])
            accuracy = exp_data["accuracy"]
            
            # 추가 메트릭 계산
            metrics = calculate_metrics(exp_data["class_stats"])
            
            # 데이터 행 생성
            row = {
                "Model": model,
                "Shot": shot,
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                "Macro_F1": metrics["Macro_F1"],
                "MCC": metrics["MCC"],
                "TP": metrics["TP_total"],
                "FP": metrics["FP_total"],
                "FN": metrics["FN_total"],
                "TN": metrics["TN_total"]
            }
            
            # 클래스별 Confusion Matrix 정보 추가 (선택 사항)
            if include_confusion:
                for cls, cls_metrics in metrics["Class_Metrics"].items():
                    row[f"{cls}_TP"] = cls_metrics["TP"]
                    row[f"{cls}_FP"] = cls_metrics["FP"]
                    row[f"{cls}_FN"] = cls_metrics["FN"]
                    row[f"{cls}_TN"] = cls_metrics["TN"]
                    row[f"{cls}_F1"] = cls_metrics["F1"]
            
            data.append(row)
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 정렬
    df = df.sort_values(by=["Model", "Shot", "Threshold"])
    
    return df

def main():
    """메인 함수"""
    args = parse_args()
    
    # 모델 목록
    models = [m.strip() for m in args.models.split(",")]
    
    logger.info(f"카테고리: {args.category}")
    logger.info(f"분석할 모델: {models}")
    
    # 모든 실험 결과 로드
    results = load_experiment_results(args.category, models)
    
    if not results:
        logger.error("분석할 실험 결과가 없습니다.")
        return 1
    
    # 메트릭 계산 및 표 생성
    df = generate_metrics_table(results, args.include_confusion)
    
    # CSV 저장
    df.to_csv(args.output, index=False)
    logger.info(f"메트릭이 {args.output}에 저장되었습니다.")
    
    # 최고 성능 조합 찾기
    best_acc = df.loc[df["Accuracy"].idxmax()]
    best_balanced = df.loc[df["Balanced_Accuracy"].idxmax()]
    best_f1 = df.loc[df["Macro_F1"].idxmax()]
    
    logger.info("\n=== 최고 성능 조합 ===")
    logger.info(f"최고 정확도: {best_acc['Model']}, shot_{best_acc['Shot']}, threshold_{best_acc['Threshold']:.2f} (Acc: {best_acc['Accuracy']:.4f})")
    logger.info(f"최고 균형 정확도: {best_balanced['Model']}, shot_{best_balanced['Shot']}, threshold_{best_balanced['Threshold']:.2f} (Balanced Acc: {best_balanced['Balanced_Accuracy']:.4f})")
    logger.info(f"최고 Macro F1: {best_f1['Model']}, shot_{best_f1['Shot']}, threshold_{best_f1['Threshold']:.2f} (F1: {best_f1['Macro_F1']:.4f})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 