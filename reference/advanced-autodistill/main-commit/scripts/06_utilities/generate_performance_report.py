import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)

def process_all_summaries_detailed(results_path):
    """
    모든 comparison_summary.json 파일을 순회하며
    상세한 '예측 흐름' 데이터를 추출합니다.
    """
    print("Processing all summary files for detailed analysis...")
    all_confusion_data = []
    models = ['dino', 'resnet']

    for model in tqdm(models, desc="Processing models"):
        model_path = Path(results_path) / model
        if not model_path.is_dir():
            continue

        for shot_dir in model_path.glob('shot_*'):
            shot = int(shot_dir.name.split('_')[1])
            for threshold_dir in shot_dir.glob('threshold_*'):
                threshold = float(threshold_dir.name.split('_')[1])
                summary_file = threshold_dir / 'comparison' / 'comparison_summary.json'

                if not summary_file.exists():
                    continue

                with open(summary_file, 'r') as f:
                    summary = json.load(f)

                class_stats = summary.get('class_stats', {})
                for true_class, stats in class_stats.items():
                    # 정답 예측 (True Positive)
                    correct_count = stats.get('correct', 0)
                    if correct_count > 0:
                        all_confusion_data.append({
                            'model': model, 'shot': shot, 'threshold': threshold,
                            'true_class': true_class, 'predicted_class': true_class,
                            'count': correct_count
                        })
                    # 오답 예측 (False Negative / Misclassification)
                    for predicted, count in stats.get('predicted_as', {}).items():
                        all_confusion_data.append({
                            'model': model, 'shot': shot, 'threshold': threshold,
                            'true_class': true_class, 'predicted_class': predicted,
                            'count': count
                        })
    
    if not all_confusion_data:
        return pd.DataFrame()

    return pd.DataFrame(all_confusion_data)

def calculate_accuracies_from_detailed_df(df):
    """상세 데이터프레임으로부터 두 시나리오의 정확도를 계산합니다."""
    if df.empty:
        return pd.DataFrame()

    # 각 그룹별로 정확도 계산
    summary_list = []
    for group, group_df in df.groupby(['model', 'shot', 'threshold']):
        model, shot, threshold = group
        
        # 시나리오 B: Unknown 포함
        total_with_unknown = group_df['count'].sum()
        correct_with_unknown = group_df[group_df['true_class'] == group_df['predicted_class']]['count'].sum()
        acc_with_unknown = (correct_with_unknown / total_with_unknown) * 100 if total_with_unknown > 0 else 0

        # 시나리오 A: Known 클래스만
        known_df = group_df[group_df['true_class'].str.startswith('Class_')]
        total_without_unknown = known_df['count'].sum()
        correct_without_unknown = known_df[known_df['true_class'] == known_df['predicted_class']]['count'].sum()
        acc_without_unknown = (correct_without_unknown / total_without_unknown) * 100 if total_without_unknown > 0 else 0
        
        summary_list.append({
            'model': model, 'shot': shot, 'threshold': threshold,
            'accuracy_with_unknown': acc_with_unknown,
            'accuracy_without_unknown': acc_without_unknown,
        })
        
    return pd.DataFrame(summary_list)


def plot_prediction_flow(df, output_path):
    """클래스별 예측 흐름을 시각화합니다."""
    if df.empty:
        print("No detailed data to plot prediction flow.")
        return
    print(f"Generating prediction flow plots to {output_path}...")

    g = sns.relplot(
        data=df,
        x='threshold', y='count',
        hue='predicted_class',
        style='shot',
        row='true_class', col='model',
        kind='line',
        markers=True,
        height=4, aspect=1.8,
        palette='tab10',
        facet_kws={'sharey': False} # y축 공유 안함
    )
    g.fig.suptitle('Detailed Prediction Flow by True Class', fontsize=22, y=1.03)
    g.set_axis_labels('Threshold', 'Prediction Count')
    g.set_titles(row_template="True: {row_name}", col_template="Model: {col_name}")
    g.legend.set_title("Predicted As / Shot")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print("Prediction flow plots saved successfully.")
    plt.close(g.fig)


def plot_prediction_flow_counts(df, output_path):
    """클래스별 예측 수량을 바 차트로 시각화합니다."""
    if df.empty:
        print("No detailed data to plot prediction counts.")
        return
    print(f"Generating prediction count bar charts to {output_path}...")

    # 모델별로 순회하며 별도의 차트 파일 생성
    for model_name, model_df in df.groupby('model'):
        g = sns.catplot(
            data=model_df,
            x='threshold', y='count',
            hue='predicted_class',
            row='true_class', col='shot',
            kind='bar',
            height=4, aspect=1.2,
            palette='tab20b',
            sharey=False,
            margin_titles=True, # facet_kws 대신 직접 파라미터로 전달
            legend=False,
            facet_kws={} # 비워두어 충돌 방지
        )
        g.fig.suptitle(f'Prediction Counts for Model: {model_name}', fontsize=22, y=1.03)
        g.set_axis_labels('Threshold', 'Prediction Count')
        g.set_titles(row_template="True: {row_name}", col_template="Shot: {col_name}")
        g.add_legend(title="Predicted As")
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # 모델별로 파일 이름 지정하여 저장
        model_output_path = Path(output_path).parent / f"{Path(output_path).stem}_{model_name}.png"
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(model_output_path, dpi=300)
        plt.close(g.fig) # 메모리 해제를 위해 명시적으로 닫기
        print(f"Prediction count bar chart for {model_name} saved to {model_output_path}")


def plot_accuracy_trends(df, output_path):
    """두 시나리오에 대한 정확도 추이를 시각화합니다."""
    if df.empty:
        print("No accuracy data to plot.")
        return
    print(f"Generating accuracy trend plots to {output_path}...")

    # 데이터를 long-form으로 변환
    df_long = df.melt(id_vars=['model', 'shot', 'threshold'],
                        value_vars=['accuracy_with_unknown', 'accuracy_without_unknown'],
                        var_name='accuracy_type', value_name='accuracy')
    df_long['accuracy_type'] = df_long['accuracy_type'].map({
        'accuracy_with_unknown': 'Including Unknown',
        'accuracy_without_unknown': 'Known Classes Only'
    })

    g = sns.relplot(
        data=df_long,
        x='threshold', y='accuracy',
        hue='shot', style='model',
        row='accuracy_type', col='model',
        kind='line', markers=True,
        height=5, aspect=1.5,
        palette='bright',
        row_order=['Known Classes Only', 'Including Unknown']
    )
    g.fig.suptitle('Accuracy Trends: Shot vs. Threshold', fontsize=22, y=1.03)
    g.set_axis_labels('Threshold', 'Accuracy (%)')
    g.set_titles(row_template="{row_name}", col_template="Model: {col_name}")
    g.set(ylim=(0, 101))
    g.legend.set_title("Shot")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Accuracy plots saved successfully.")


def plot_confusion_matrices(df, output_path):
    """모델별 집계된 혼동 행렬을 시각화합니다."""
    if df.empty:
        print("No confusion data to plot.")
        return
    print(f"Generating confusion matrices to {output_path}...")

    models = df['model'].unique()
    class_order = sorted(df['true_class'].unique())
    
    fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), 8), sharey=True)
    if len(models) == 1:
        axes = [axes]
        
    fig.suptitle('Aggregated Confusion Matrix (All Shots & Thresholds)', fontsize=22, y=1.02)

    for i, model in enumerate(models):
        ax = axes[i]
        model_df = df[df['model'] == model]
        
        # pivot table을 사용하여 혼동 행렬 생성
        confusion = model_df.pivot_table(index='true_class', columns='predicted_class',
                                         values='count', aggfunc='sum').fillna(0)
        
        # 모든 클래스가 행/열에 나타나도록 reindex
        confusion = confusion.reindex(index=class_order, columns=class_order, fill_value=0)

        sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues', ax=ax, cbar=i==len(models)-1)
        ax.set_title(f'Model: {model.upper()}', fontsize=16)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class' if i == 0 else '', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Confusion matrices saved successfully.")


def calculate_overall_performance(df):
    """상세 데이터프레임에서 전반적인 성능 지표를 계산합니다."""
    if df.empty:
        return pd.DataFrame()

    performance_data = []
    grouped = df.groupby(['model', 'shot', 'threshold'])

    for name, group in tqdm(grouped, desc="Calculating Performance Metrics"):
        model, shot, threshold = name
        
        y_true = []
        y_pred = []
        for _, row in group.iterrows():
            y_true.extend([row['true_class']] * row['count'])
            y_pred.extend([row['predicted_class']] * row['count'])

        # 전체 정확도 (Unknown 포함)
        accuracy_all = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true) if y_true else 0

        # Known 클래스만 필터링
        known_indices = [i for i, yt in enumerate(y_true) if yt.startswith('Class_')]
        y_true_known = [y_true[i] for i in known_indices]
        y_pred_known = [y_pred[i] for i in known_indices]

        # Known 클래스 정확도
        accuracy_known = sum(1 for yt, yp in zip(y_true_known, y_pred_known) if yt == yp) / len(y_true_known) if y_true_known else 0
        
        # F1-Score (weighted)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 분류율 (Unknown이 아닌 예측 비율)
        classified_rate = sum(1 for yp in y_pred if yp != 'Unknown') / len(y_pred) if y_pred else 0

        performance_data.append({
            'model': model,
            'shot': shot,
            'threshold': threshold,
            'accuracy_all': accuracy_all,
            'accuracy_known': accuracy_known,
            'f1_weighted': f1,
            'classified_rate': classified_rate
        })
    
    return pd.DataFrame(performance_data)


def plot_performance_dashboard(df, output_path):
    """모델별 종합 성능 대시보드를 시각화합니다."""
    if df.empty:
        print("No performance data for dashboard.")
        return
    print(f"Generating performance dashboard to {output_path}...")

    models = df['model'].unique()
    for model in models:
        model_df = df[df['model'] == model]
        shots = sorted(model_df['shot'].unique())
        
        fig, axes = plt.subplots(len(shots), 3, figsize=(20, 5 * len(shots)), sharex=True)
        fig.suptitle(f'{model.upper()} Model Performance Dashboard', fontsize=22, y=1.02)

        for i, shot in enumerate(shots):
            shot_df = model_df[model_df['shot'] == shot]
            
            # 1. Accuracy
            ax1 = axes[i, 0]
            sns.lineplot(data=shot_df, x='threshold', y='accuracy_all', ax=ax1, marker='o', label='Accuracy (All)')
            sns.lineplot(data=shot_df, x='threshold', y='accuracy_known', ax=ax1, marker='o', label='Accuracy (Known)')
            ax1.set_title(f'Shot: {shot} - Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()
            ax1.set_ylim(0, 1.01)

            # 2. F1-Score
            ax2 = axes[i, 1]
            sns.lineplot(data=shot_df, x='threshold', y='f1_weighted', ax=ax2, marker='o', color='g')
            ax2.set_title(f'Shot: {shot} - F1 Score (Weighted)')
            ax2.set_ylabel('F1 Score')
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.set_ylim(0, 1.01)

            # 3. Classification Rate
            ax3 = axes[i, 2]
            sns.lineplot(data=shot_df, x='threshold', y='classified_rate', ax=ax3, marker='o', color='r')
            ax3.set_title(f'Shot: {shot} - Classification Rate')
            ax3.set_ylabel('Rate (Non-Unknown)')
            ax3.grid(True, linestyle='--', alpha=0.6)
            ax3.set_ylim(0, 1.01)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        model_output_path = Path(output_path).parent / f"performance_dashboard_{model}.png"
        plt.savefig(model_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Performance dashboard for {model} saved to {model_output_path}")


def analyze_known_class_performance(df, output_path):
    """
    Unknown을 제외한 Known 클래스(Class_0, 1, 2, 3)에 대한 성능을 분석하고,
    Unknown 예측 수와 함께 결과를 CSV로 저장합니다.
    """
    if df.empty:
        print("No detailed data for known class performance analysis.")
        return
    print(f"Analyzing performance on known classes, saving to {output_path}...")

    # 모든 실험에서 나타난 Known 클래스 목록을 미리 확보하여 CSV 컬럼 순서를 고정
    all_known_classes = sorted(df[df['true_class'].str.startswith('Class_')]['true_class'].unique())

    report_data = []
    grouped = df.groupby(['model', 'shot', 'threshold'])

    for name, group in tqdm(grouped, desc="Analyzing Known Class Performance"):
        model, shot, threshold = name

        # 1. 전체 데이터에서 Unknown 예측 수 계산
        total_unknown_predictions = group[group['predicted_class'] == 'Unknown']['count'].sum()

        # 2. Ground Truth가 Known 클래스인 데이터만 필터링
        known_gt_df = group[group['true_class'].str.startswith('Class_')]
        
        # 기본 row 구조 생성
        row = {
            'model': model, 'shot': shot, 'threshold': threshold,
            'total_unknown_predictions': total_unknown_predictions,
            'known_gt_total_samples': 0,
            'known_gt_correctly_classified': 0,
            'known_gt_predicted_as_unknown': 0,
            'known_gt_accuracy_percent': 0
        }
        # 클래스별 샘플 수 및 정답 수 초기화
        for cls in all_known_classes:
            row[f'known_gt_samples_{cls}'] = 0
            row[f'correctly_classified_{cls}'] = 0
        
        if known_gt_df.empty:
            report_data.append(row)
            continue

        # 3. Known 클래스 대상 분석
        known_gt_total_samples = known_gt_df['count'].sum()
        correct_predictions = known_gt_df[known_gt_df['true_class'] == known_gt_df['predicted_class']]['count'].sum()
        predicted_as_unknown = known_gt_df[known_gt_df['predicted_class'] == 'Unknown']['count'].sum()
        accuracy_known_gt = (correct_predictions / known_gt_total_samples) * 100 if known_gt_total_samples > 0 else 0

        # 계산된 값으로 row 업데이트
        row.update({
            'known_gt_total_samples': known_gt_total_samples,
            'known_gt_correctly_classified': correct_predictions,
            'known_gt_predicted_as_unknown': predicted_as_unknown,
            'known_gt_accuracy_percent': accuracy_known_gt
        })
        
        # 클래스별 GT 샘플 수 계산 및 추가
        class_counts = known_gt_df.groupby('true_class')['count'].sum()
        for cls in all_known_classes:
            row[f'known_gt_samples_{cls}'] = class_counts.get(cls, 0)

        # 각 클래스별 정답 수 계산 및 추가
        correctly_classified_df = known_gt_df[known_gt_df['true_class'] == known_gt_df['predicted_class']]
        correct_counts = correctly_classified_df.groupby('true_class')['count'].sum()
        for cls in all_known_classes:
            row[f'correctly_classified_{cls}'] = correct_counts.get(cls, 0)

        report_data.append(row)
    
    if not report_data:
        print("No data found for known class performance report.")
        return

    # 데이터프레임 생성 및 저장
    report_df = pd.DataFrame(report_data)
    # 컬럼 순서 재정렬 (가독성 향상)
    base_cols = ['model', 'shot', 'threshold', 'total_unknown_predictions', 'known_gt_total_samples', 
                 'known_gt_correctly_classified', 'known_gt_predicted_as_unknown', 'known_gt_accuracy_percent']
    sample_cols = sorted([col for col in report_df.columns if 'known_gt_samples_' in col])
    correct_cols = sorted([col for col in report_df.columns if 'correctly_classified_' in col])
    
    final_cols = base_cols + sample_cols + correct_cols
    report_df = report_df[final_cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Known class performance report saved to {output_path}")


def analyze_known_class_performance_detailed(df, output_path):
    """
    Known 클래스에 대한 상세 성능(클래스별 정답 수 및 정확도 포함)을 분석하고 CSV로 저장합니다.
    """
    if df.empty:
        print("No detailed data for detailed known class performance analysis.")
        return
    print(f"Analyzing detailed performance on known classes, saving to {output_path}...")

    all_known_classes = sorted(df[df['true_class'].str.startswith('Class_')]['true_class'].unique())
    report_data = []
    grouped = df.groupby(['model', 'shot', 'threshold'])

    for name, group in tqdm(grouped, desc="Analyzing Detailed Known Class Performance"):
        model, shot, threshold = name
        total_unknown_predictions = group[group['predicted_class'] == 'Unknown']['count'].sum()
        known_gt_df = group[group['true_class'].str.startswith('Class_')]
        
        row = {
            'model': model, 'shot': shot, 'threshold': threshold,
            'total_unknown_predictions': total_unknown_predictions,
        }
        
        # Initialize all possible columns to ensure consistency
        row.update({
            'known_gt_total_samples': 0, 'known_gt_correctly_classified': 0,
            'known_gt_predicted_as_unknown': 0, 'known_gt_accuracy_percent': 0
        })
        for cls in all_known_classes:
            row[f'known_gt_samples_{cls}'] = 0
            row[f'correctly_classified_{cls}'] = 0
            row[f'{cls}_accuracy_percent'] = 0

        if not known_gt_df.empty:
            known_gt_total_samples = known_gt_df['count'].sum()
            correct_predictions = known_gt_df[known_gt_df['true_class'] == known_gt_df['predicted_class']]['count'].sum()
            predicted_as_unknown = known_gt_df[known_gt_df['predicted_class'] == 'Unknown']['count'].sum()
            accuracy_known_gt = (correct_predictions / known_gt_total_samples) * 100 if known_gt_total_samples > 0 else 0

            row.update({
                'known_gt_total_samples': known_gt_total_samples,
                'known_gt_correctly_classified': correct_predictions,
                'known_gt_predicted_as_unknown': predicted_as_unknown,
                'known_gt_accuracy_percent': accuracy_known_gt
            })
            
            class_counts = known_gt_df.groupby('true_class')['count'].sum()
            correctly_classified_df = known_gt_df[known_gt_df['true_class'] == known_gt_df['predicted_class']]
            correct_counts = correctly_classified_df.groupby('true_class')['count'].sum()

            for cls in all_known_classes:
                gt_count = class_counts.get(cls, 0)
                correct_count = correct_counts.get(cls, 0)
                row[f'known_gt_samples_{cls}'] = gt_count
                row[f'correctly_classified_{cls}'] = correct_count
                row[f'{cls}_accuracy_percent'] = (correct_count / gt_count) * 100 if gt_count > 0 else 0

            # FSL 정확도 계산 (Predicted / Ground Truth)
            predicted_counts = group.groupby('predicted_class')['count'].sum()
            for cls in all_known_classes:
                predicted_count = predicted_counts.get(cls, 0)
                gt_count = class_counts.get(cls, 0)
                row[f'fsl_accuracy_{cls}'] = (predicted_count / gt_count) * 100 if gt_count > 0 else 0
        
        report_data.append(row)

    report_df = pd.DataFrame(report_data)
    base_cols = ['model', 'shot', 'threshold', 'total_unknown_predictions', 'known_gt_total_samples', 
                 'known_gt_correctly_classified', 'known_gt_predicted_as_unknown', 'known_gt_accuracy_percent']
    sample_cols = sorted([col for col in report_df.columns if 'known_gt_samples_' in col])
    correct_cols = sorted([col for col in report_df.columns if 'correctly_classified_' in col])
    accuracy_cols = sorted([col for col in report_df.columns if col.endswith('_accuracy_percent') and col != 'known_gt_accuracy_percent'])
    fsl_accuracy_cols = sorted([col for col in report_df.columns if col.startswith('fsl_accuracy_')])

    final_cols = base_cols + sample_cols + correct_cols + accuracy_cols + fsl_accuracy_cols
    report_df = report_df[final_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Detailed known class performance report saved to {output_path}")


def plot_detailed_fsl_accuracy_trends(df, output_path):
    """(차트 A) Threshold에 따른 클래스별 FSL 정확도 추이를 시각화합니다."""
    if df.empty or not any(col.endswith('_accuracy_percent') for col in df.columns):
        print("No per-class accuracy data to plot.")
        return
    print(f"Generating per-class FSL accuracy trend plots to {output_path}...")

    accuracy_cols = sorted([col for col in df.columns if col.startswith('Class_') and col.endswith('_accuracy_percent')])
    df_long = df.melt(id_vars=['model', 'shot', 'threshold'], value_vars=accuracy_cols,
                      var_name='class_accuracy', value_name='accuracy')
    df_long['class'] = df_long['class_accuracy'].str.replace('_accuracy_percent', '')

    g = sns.relplot(data=df_long, x='threshold', y='accuracy', hue='class', style='shot',
                    col='model', kind='line', markers=True, height=5, aspect=1.5,
                    palette='tab10', facet_kws={'sharey': True})
    g.fig.suptitle('Per-Class FSL Accuracy vs. Threshold', fontsize=22, y=1.03)
    g.set_axis_labels('Threshold', 'Accuracy (%)')
    g.set_titles(col_template="{col_name} Model")
    g.set(ylim=(0, 101))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Per-class FSL accuracy trends saved to {output_path}")

def plot_unknown_vs_class_accuracy(df, output_path):
    """(차트 B) Unknown 예측 수와 클래스별 정확도 관계를 시각화합니다."""
    if df.empty:
        print("No data to plot Unknown vs Class Accuracy relationship.")
        return
    print(f"Generating Unknown vs. Class Accuracy plots to {output_path}...")

    models = df['model'].unique()
    shots = sorted(df['shot'].unique())
    accuracy_cols = sorted([col for col in df.columns if col.startswith('Class_') and col.endswith('_accuracy_percent')])

    for model in models:
        fig, axes = plt.subplots(len(shots), 1, figsize=(15, 7 * len(shots)), sharex=True, squeeze=False)
        fig.suptitle(f'{model.upper()} Model: Unknown Predictions vs. Class Accuracy', fontsize=22, y=1.0)
        
        for i, shot in enumerate(shots):
            ax1 = axes[i, 0]
            shot_df = df[(df['model'] == model) & (df['shot'] == shot)]
            
            # 첫 번째 Y축 (Unknown 수)
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Total Unknown Predictions', color='tab:red')
            sns.lineplot(data=shot_df, x='threshold', y='total_unknown_predictions', ax=ax1, 
                         color='tab:red', marker='s', label='Total Unknowns')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            # 두 번째 Y축 (클래스별 정확도)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Per-Class Accuracy (%)')
            df_long = shot_df.melt(id_vars=['threshold'], value_vars=accuracy_cols,
                                   var_name='class_accuracy', value_name='accuracy')
            df_long['class'] = df_long['class_accuracy'].str.replace('_accuracy_percent', '')
            sns.lineplot(data=df_long, x='threshold', y='accuracy', hue='class', ax=ax2, 
                         marker='o', linestyle='--', palette='coolwarm')
            
            ax1.set_title(f'Shot: {shot}')
            ax1.grid(True, axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        model_output_path = Path(output_path).parent / f"unknown_vs_accuracy_{model}.png"
        plt.savefig(model_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Unknown vs. Class Accuracy plot for {model} saved to {model_output_path}")


def plot_comparative_known_accuracy(df, output_path):
    """(차트 C) 모델별 Known 클래스 전체 정확도를 비교 시각화합니다."""
    if df.empty:
        print("No data for comparative known accuracy plot.")
        return
    print(f"Generating comparative known accuracy plots to {output_path}...")
    
    g = sns.relplot(data=df, x='threshold', y='known_gt_accuracy_percent', hue='model',
                    col='shot', kind='line', markers=True, height=5, aspect=1.2,
                    palette='colorblind', facet_kws={'sharey': True})
    g.fig.suptitle('Comparative Accuracy on Known Classes', fontsize=22, y=1.03)
    g.set_axis_labels('Threshold', 'Accuracy on Known GT (%)')
    g.set_titles(col_template="Shot: {col_name}")
    g.set(ylim=(0, 101))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Comparative known accuracy plot saved to {output_path}")


def perform_correlation_analysis(df, output_path):
    """상관관계 분석을 수행하고 히트맵으로 시각화합니다."""
    if df.empty:
        print("No data for correlation analysis.")
        return
    print(f"Performing correlation analysis and saving to {output_path}...")
    
    models = df['model'].unique()
    
    for model in models:
        model_df = df[df['model'] == model].drop(columns=['model'])
        
        # 분석할 컬럼 선택
        corr_cols = ['shot', 'threshold', 'total_unknown_predictions', 'known_gt_accuracy_percent'] + \
                    [col for col in df.columns if col.startswith('Class_') and col.endswith('_accuracy_percent')]
        
        # 존재하지 않는 컬럼 제거
        corr_cols = [col for col in corr_cols if col in model_df.columns]
        
        corr_matrix = model_df[corr_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title(f'{model.upper()} Model: Correlation Matrix of Performance Metrics', fontsize=16)
        
        model_output_path = Path(output_path).parent / f"correlation_matrix_{model}.png"
        plt.savefig(model_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation matrix for {model} saved to {model_output_path}")


def plot_fsl_accuracy_trends(df, output_path):
    """FSL 정확도(Predicted/GT)의 추이를 시각화합니다."""
    if df.empty or not any(col.startswith('fsl_accuracy_') for col in df.columns):
        print("No FSL accuracy data to plot.")
        return
    print(f"Generating FSL accuracy trend plots to {output_path}...")

    fsl_cols = sorted([col for col in df.columns if col.startswith('fsl_accuracy_')])
    df_long = df.melt(id_vars=['model', 'shot', 'threshold'], value_vars=fsl_cols,
                      var_name='fsl_accuracy_class', value_name='fsl_accuracy')
    df_long['class'] = df_long['fsl_accuracy_class'].str.replace('fsl_accuracy_', '')

    g = sns.relplot(data=df_long, x='threshold', y='fsl_accuracy', hue='class', style='shot',
                    col='model', kind='line', markers=True, height=5, aspect=1.5,
                    palette='viridis', facet_kws={'sharey': False})
    g.fig.suptitle('FSL Accuracy (Predicted/GT) vs. Threshold', fontsize=22, y=1.03)
    g.set_axis_labels('Threshold', 'FSL Accuracy (%)')
    g.set_titles(col_template="{col_name} Model")
    g.map(plt.axhline, y=100, color='red', linestyle='--', linewidth=1) # 100% 기준선 추가
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"FSL accuracy trends saved to {output_path}")


def generate_resnet_known_accuracy_report(df, output_dir):
    """
    ResNet 모델의 Known 클래스 정확도 변화 추이를 분석하고
    결과를 CSV와 그래프로 저장합니다.
    """
    if df.empty or 'model' not in df.columns:
        print("Not enough data to generate ResNet specific report.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating ResNet Known Accuracy Trend Report...")

    # 1. ResNet 데이터 필터링 및 CSV 저장
    resnet_df = df[df['model'] == 'resnet'].copy()
    
    if resnet_df.empty:
        print("No data found for ResNet model.")
        return

    # 필요한 컬럼 선택
    report_cols = ['shot', 'threshold', 'known_gt_accuracy_percent']
    if not all(col in resnet_df.columns for col in report_cols):
        print(f"Required columns ({report_cols}) not found in the dataframe.")
        return
        
    resnet_report_df = resnet_df[report_cols]
    
    csv_path = output_dir / 'resnet_known_accuracy_report.csv'
    resnet_report_df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"ResNet known accuracy data saved to {csv_path}")

    # 2. 그래프 생성
    plt.figure(figsize=(12, 8))
    g = sns.relplot(
        data=resnet_report_df,
        x='threshold',
        y='known_gt_accuracy_percent',
        hue='shot',
        kind='line',
        marker='o',
        height=6,
        aspect=1.5,
        palette='crest'
    )
    g.fig.suptitle('ResNet: Known Class Accuracy vs. Threshold', fontsize=18, y=1.03)
    g.set_axis_labels('Threshold', 'Accuracy on Known GT (%)')
    g.set(ylim=(0, 101))
    g.legend.set_title("Shot")
    
    plot_path = output_dir / 'resnet_known_accuracy_trend.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"ResNet known accuracy trend plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a comprehensive and detailed performance report.")
    parser.add_argument('--results-dir', default='data/test_category/7.results')
    parser.add_argument('--output-dir', default='data/test_category/7.results/analysis_results')
    args = parser.parse_args()

    detailed_df = process_all_summaries_detailed(args.results_dir)
    
    if not detailed_df.empty:
        # 1. 상세 데이터 CSV 저장
        output_csv = Path(args.output_dir) / 'detailed_prediction_flow.csv'
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(output_csv, index=False)
        print(f"\nDetailed prediction flow data saved to {output_csv}")

        # 2. 정확도 계산 및 차트 생성
        accuracy_df = calculate_accuracies_from_detailed_df(detailed_df)
        if not accuracy_df.empty:
            plot_accuracy_trends(accuracy_df, Path(args.output_dir) / 'accuracy_trends.png')
        
        # 3. 새로운 예측 흐름 차트 생성
        plot_prediction_flow(detailed_df, Path(args.output_dir) / 'prediction_flow_by_class.png')

        # 4. (신규) 예측 수량 바 차트 생성
        plot_prediction_flow_counts(detailed_df, Path(args.output_dir) / 'prediction_flow_counts_by_class.png')

        # 5. (신규) 종합 성능 대시보드 생성
        performance_df = calculate_overall_performance(detailed_df)
        if not performance_df.empty:
            plot_performance_dashboard(performance_df, Path(args.output_dir) / 'performance_dashboard.png')

        # 6. (신규) Known 클래스 성능 분석 및 CSV 저장
        known_class_report_csv = Path(args.output_dir) / 'known_class_performance_report.csv'
        analyze_known_class_performance(detailed_df, known_class_report_csv)

        # 7. (신규) 상세 Known 클래스 성능 분석
        detailed_known_report_csv = Path(args.output_dir) / 'known_class_performance_report_detailed.csv'
        analyze_known_class_performance_detailed(detailed_df, detailed_known_report_csv)

        # 상세 리포트가 생성되었는지 확인 후 후속 분석 진행
        if detailed_known_report_csv.exists():
            detailed_report_df = pd.read_csv(detailed_known_report_csv)
            
            # 8. (신규) 클래스별 FSL 정확도 추이 차트 생성
            plot_detailed_fsl_accuracy_trends(detailed_report_df, Path(args.output_dir) / 'trends_fsl_accuracy_per_class.png')
            
            # 9. (신규) Unknown 예측과 정확도 관계 차트 생성
            plot_unknown_vs_class_accuracy(detailed_report_df, Path(args.output_dir) / 'relationship_unknown_vs_accuracy.png')

            # 10. (신규) 모델별 Known 클래스 정확도 비교 차트 생성
            plot_comparative_known_accuracy(detailed_report_df, Path(args.output_dir) / 'compare_known_gt_accuracy.png')

            # 11. (신규) 상관관계 분석
            perform_correlation_analysis(detailed_report_df, Path(args.output_dir) / 'correlation_analysis.png')
            
            # 12. (신규) FSL 정확도 추이 차트 생성
            plot_fsl_accuracy_trends(detailed_report_df, Path(args.output_dir) / 'trends_fsl_accuracy.png')

            # 13. (요청사항) ResNet Known 정확도 리포트 생성
            generate_resnet_known_accuracy_report(detailed_report_df, args.output_dir)
    else:
        print("Analysis finished with no data found.")

if __name__ == '__main__':
    main() 