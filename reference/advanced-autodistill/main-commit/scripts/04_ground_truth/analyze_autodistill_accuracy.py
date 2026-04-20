#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autodistill vs Ground Truth 분류 정확도 분석
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import argparse
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_frame_info_from_filename(filename):
    """
    파일명에서 프레임 정보 추출
    예: 'G_1_2_frame_0073_obj2_cls3_unknown_class_3.png' -> 'G_1_2_frame_0073'
    """
    # _obj가 나오기 전까지의 부분을 추출
    match = re.match(r'(.+?)_obj\d+', filename)
    if match:
        return match.group(1)
    return None

def extract_object_index_from_filename(filename):
    """
    파일명에서 객체 인덱스 추출
    예: 'G_1_2_frame_0073_obj2_cls3_unknown_class_3.png' -> 2
    """
    match = re.search(r'_obj(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def get_ground_truth_mapping(gt_dir):
    """
    Ground Truth 디렉토리에서 파일 매핑 생성
    Returns: {(frame_name, obj_index): gt_class}
    """
    logger.info("Ground Truth 매핑 생성 중...")
    
    gt_mapping = {}
    class_folders = {
        'Class_0': 'Class_0',
        'Class_1': 'Class_1', 
        'Class_2': 'Class_2',
        'Class_3': 'Class_3',
        'unknown_egifence': 'unknown_egifence',
        'unknown_human': 'unknown_human',
        'unknown_road': 'unknown_road',
        'unknown_none': 'unknown_none'
    }
    
    gt_stats = Counter()
    
    for folder_name, gt_class in class_folders.items():
        folder_path = os.path.join(gt_dir, folder_name)
        if not os.path.exists(folder_path):
            logger.warning(f"Ground Truth 폴더가 존재하지 않습니다: {folder_path}")
            continue
            
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            frame_name = extract_frame_info_from_filename(filename)
            obj_index = extract_object_index_from_filename(filename)
            
            if frame_name and obj_index is not None:
                key = (frame_name, obj_index)
                gt_mapping[key] = gt_class
                gt_stats[gt_class] += 1
    
    logger.info("Ground Truth 통계:")
    for gt_class, count in sorted(gt_stats.items(), key=lambda x: (isinstance(x[0], str), x[0])):
        logger.info(f"  {gt_class}: {count}개")
    
    return gt_mapping

def get_autodistill_mapping(autodistill_dir):
    """
    Autodistill 디렉토리에서 파일 매핑 생성
    Returns: {(frame_name, obj_index): autodistill_class}
    """
    logger.info("Autodistill 매핑 생성 중...")
    
    autodistill_mapping = {}
    autodistill_stats = Counter()
    
    for class_id in [0, 1, 2, 3]:
        folder_path = os.path.join(autodistill_dir, f'Class_{class_id}')
        class_label = f'Class_{class_id}'  # 문자열로 변환
        
        if not os.path.exists(folder_path):
            logger.warning(f"Autodistill 폴더가 존재하지 않습니다: {folder_path}")
            continue
            
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            frame_name = extract_frame_info_from_filename(filename)
            obj_index = extract_object_index_from_filename(filename)
            
            if frame_name and obj_index is not None:
                key = (frame_name, obj_index)
                autodistill_mapping[key] = class_label
                autodistill_stats[class_label] += 1
    
    logger.info("Autodistill 통계:")
    for class_id, count in sorted(autodistill_stats.items()):
        logger.info(f"  {class_id}: {count}개")
    
    return autodistill_mapping

def create_confusion_matrix_data(autodistill_mapping, gt_mapping):
    """
    Confusion matrix 데이터 생성
    """
    logger.info("Confusion matrix 데이터 생성 중...")
    
    # 공통 키 찾기
    common_keys = set(autodistill_mapping.keys()) & set(gt_mapping.keys())
    logger.info(f"매칭된 이미지 개수: {len(common_keys)}개")
    
    if len(common_keys) == 0:
        logger.error("매칭된 이미지가 없습니다!")
        return None, None, None
    
    # 예측값과 실제값 리스트 생성
    y_pred = []
    y_true = []
    detailed_data = []
    
    for key in common_keys:
        autodistill_class = autodistill_mapping[key]
        gt_class = gt_mapping[key]
        
        y_pred.append(autodistill_class)
        y_true.append(gt_class)
        
        detailed_data.append({
            'frame_name': key[0],
            'object_index': key[1],
            'autodistill_prediction': autodistill_class,
            'ground_truth': gt_class,
            'correct': autodistill_class == gt_class
        })
    
    return y_pred, y_true, detailed_data

def calculate_metrics(y_true, y_pred, labels):
    """
    정확도 메트릭 계산
    """
    logger.info("정확도 메트릭 계산 중...")
    
    # 전체 정확도
    accuracy = accuracy_score(y_true, y_pred)
    
    # 클래스별 메트릭
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # 매크로 평균
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    
    # 가중 평균
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Confusion matrix 시각화
    """
    logger.info("Confusion matrix 시각화 생성 중...")
    
    # Confusion matrix 계산
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 정규화된 confusion matrix도 계산
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 2개의 subplot 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 원본 confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Ground Truth', fontsize=12)
    ax1.set_ylabel('Autodistill Prediction', fontsize=12)
    
    # 정규화된 confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Ground Truth', fontsize=12)
    ax2.set_ylabel('Autodistill Prediction', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix 저장됨: {output_path}")
    
    return cm, cm_normalized

def save_detailed_results(detailed_data, metrics, cm, cm_normalized, labels, output_dir):
    """
    상세 결과를 CSV 파일로 저장
    """
    logger.info("상세 결과 CSV 저장 중...")
    
    # 1. 상세 예측 결과
    detailed_df = pd.DataFrame(detailed_data)
    detailed_csv_path = os.path.join(output_dir, 'detailed_predictions.csv')
    detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    
    # 2. Confusion matrix
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path, encoding='utf-8-sig')
    
    # 3. 정규화된 confusion matrix
    cm_norm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    cm_norm_csv_path = os.path.join(output_dir, 'confusion_matrix_normalized.csv')
    cm_norm_df.to_csv(cm_norm_csv_path, encoding='utf-8-sig')
    
    # 4. 전체 메트릭
    metrics_data = {
        'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1',
                   'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
        'Value': [metrics['accuracy'], metrics['macro_precision'], metrics['macro_recall'],
                  metrics['macro_f1'], metrics['weighted_precision'], metrics['weighted_recall'],
                  metrics['weighted_f1']]
    }
    
    # 클래스별 메트릭 추가
    for i, label in enumerate(labels):
        metrics_data['Metric'].extend([
            f'{label}_Precision', f'{label}_Recall', f'{label}_F1', f'{label}_Support'
        ])
        metrics_data['Value'].extend([
            metrics['class_metrics']['precision'][i],
            metrics['class_metrics']['recall'][i],
            metrics['class_metrics']['f1'][i],
            metrics['class_metrics']['support'][i]
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(output_dir, 'accuracy_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
    
    # 5. 클래스별 요약 통계
    summary_data = []
    for i, label in enumerate(labels):
        correct = sum(1 for d in detailed_data 
                     if d['autodistill_prediction'] == label and d['correct'])
        total_predicted = sum(1 for d in detailed_data 
                             if d['autodistill_prediction'] == label)
        total_actual = sum(1 for d in detailed_data 
                          if d['ground_truth'] == label)
        
        summary_data.append({
            'Class': label,
            'Precision': metrics['class_metrics']['precision'][i],
            'Recall': metrics['class_metrics']['recall'][i],
            'F1-Score': metrics['class_metrics']['f1'][i],
            'Support': metrics['class_metrics']['support'][i],
            'Correctly_Predicted': correct,
            'Total_Predicted': total_predicted,
            'Total_Actual': total_actual
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'class_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"CSV 파일들이 저장됨: {output_dir}")
    
    return {
        'detailed': detailed_csv_path,
        'confusion_matrix': cm_csv_path,
        'confusion_matrix_norm': cm_norm_csv_path,
        'metrics': metrics_csv_path,
        'summary': summary_csv_path
    }

def print_console_summary(metrics, cm, labels, detailed_data):
    """
    콘솔에 요약 정보 출력
    """
    print("\n" + "="*80)
    print("AUTODISTILL vs GROUND TRUTH 분류 정확도 분석 결과")
    print("="*80)
    
    # 전체 정확도
    print(f"\n📊 전체 정확도: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"📊 총 매칭된 이미지: {len(detailed_data)}개")
    
    # 매크로/가중 평균
    print(f"\n📈 Macro Average:")
    print(f"   - Precision: {metrics['macro_precision']:.4f}")
    print(f"   - Recall: {metrics['macro_recall']:.4f}")
    print(f"   - F1-Score: {metrics['macro_f1']:.4f}")
    
    print(f"\n📈 Weighted Average:")
    print(f"   - Precision: {metrics['weighted_precision']:.4f}")
    print(f"   - Recall: {metrics['weighted_recall']:.4f}")
    print(f"   - F1-Score: {metrics['weighted_f1']:.4f}")
    
    # 클래스별 상세 정보
    print(f"\n📋 클래스별 상세 정보:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    
    for i, label in enumerate(labels):
        precision = metrics['class_metrics']['precision'][i]
        recall = metrics['class_metrics']['recall'][i]
        f1 = metrics['class_metrics']['f1'][i]
        support = int(metrics['class_metrics']['support'][i])
        
        print(f"{str(label):<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    # Confusion Matrix 요약
    print(f"\n📊 Confusion Matrix (실제 → 예측):")
    print("-" * 60)
    
    # 헤더
    header = "실제\\예측".ljust(15)
    for label in labels:
        header += str(label).ljust(12)
    print(header)
    print("-" * 60)
    
    # 각 행
    for i, true_label in enumerate(labels):
        row = str(true_label).ljust(15)
        for j, pred_label in enumerate(labels):
            row += str(cm[i][j]).ljust(12)
        print(row)
    
    # 특별히 잘못 분류된 케이스들
    print(f"\n🚨 주요 오분류 패턴:")
    misclassified = {}
    for data in detailed_data:
        if not data['correct']:
            key = (data['ground_truth'], data['autodistill_prediction'])
            misclassified[key] = misclassified.get(key, 0) + 1
    
    sorted_misclassified = sorted(misclassified.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
    
    for (true_class, pred_class), count in sorted_misclassified:
        print(f"   {true_class} → {pred_class}: {count}개")

def main():
    parser = argparse.ArgumentParser(description='Autodistill vs Ground Truth 정확도 분석')
    parser.add_argument('--category-path', default='data/test_category',
                        help='카테고리 경로 (기본: data/test_category)')
    parser.add_argument('--output-dir', default='analysis_results',
                        help='결과 저장 디렉토리 (기본: analysis_results)')
    parser.add_argument('--verbose', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 경로 설정
    autodistill_dir = os.path.join(args.category_path, '6.preprocessed')
    gt_dir = os.path.join(args.category_path, '7.results', 'ground_truth')
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(args.output_dir, f'analysis_{timestamp}')
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Autodistill 디렉토리: {autodistill_dir}")
    print(f"Ground Truth 디렉토리: {gt_dir}")
    print(f"결과 저장 디렉토리: {output_subdir}")
    
    try:
        # 1. 매핑 생성
        autodistill_mapping = get_autodistill_mapping(autodistill_dir)
        gt_mapping = get_ground_truth_mapping(gt_dir)
        
        # 2. Confusion matrix 데이터 생성
        y_pred, y_true, detailed_data = create_confusion_matrix_data(
            autodistill_mapping, gt_mapping)
        
        if y_pred is None:
            logger.error("분석할 데이터가 없습니다.")
            return
        
        # 3. 라벨 정의 (순서 중요)
        all_labels = sorted(set(y_true + y_pred), key=lambda x: (isinstance(x, str), x))
        
        # 4. 메트릭 계산
        metrics = calculate_metrics(y_true, y_pred, all_labels)
        
        # 5. Confusion matrix 시각화
        cm_png_path = os.path.join(output_subdir, 'confusion_matrix.png')
        cm, cm_normalized = plot_confusion_matrix(y_true, y_pred, all_labels, cm_png_path)
        
        # 6. CSV 파일 저장
        csv_paths = save_detailed_results(detailed_data, metrics, cm, cm_normalized, 
                                        all_labels, output_subdir)
        
        # 7. 콘솔 요약 출력
        print_console_summary(metrics, cm, all_labels, detailed_data)
        
        # 8. 파일 경로 출력
        print(f"\n📁 생성된 파일들:")
        print(f"   - Confusion Matrix (PNG): {cm_png_path}")
        for name, path in csv_paths.items():
            print(f"   - {name.replace('_', ' ').title()} (CSV): {path}")
        
        print(f"\n✅ 분석 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 