#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning 웹 애플리케이션

이 모듈은 Gradio를 사용하여 Few-Shot Learning 실험 결과를 시각화하고,
Ground Truth 레이블링, 평가 결과 시각화 등의 기능을 제공하는 웹 애플리케이션을 구현합니다.

주요 기능:
1. 실험 결과 브라우징 및 시각화
2. Ground Truth 레이블링 인터페이스
3. 평가 지표 시각화
4. Annotation 정보 동기화
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime

# Import project utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_utils import get_category_path, load_class_mapping
from scripts.classifier_cosine_experiment import FewShotExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fsl_webapp")


class FewShotWebApp:
    """Few-Shot Learning 웹 애플리케이션 클래스"""
    
    def __init__(self):
        """초기화 함수"""
        # 상태 변수 초기화
        self.category_name = None
        self.experiment = None
        self.current_experiment_id = None
        self.ground_truth_data = {}
        self.available_categories = self._get_available_categories()
        
        # UI 구성
        self.app = self._build_ui()
    
    def _get_available_categories(self) -> List[str]:
        """사용 가능한 카테고리 목록 가져오기"""
        categories = []
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                category_path = os.path.join(data_dir, item)
                if os.path.isdir(category_path):
                    support_dir = os.path.join(category_path, "2.support-set")
                    if os.path.exists(support_dir):
                        categories.append(item)
        
        return categories
    
    def _build_ui(self) -> gr.Blocks:
        """Gradio UI 구성"""
        with gr.Blocks(title="Few-Shot Learning Experiment & Evaluation Platform") as app:
            gr.Markdown("# Few-Shot Learning Experiment & Evaluation Platform (FSL-EEP)")
            
            with gr.Tabs():
                # 탭 1: 실험 설정 및 실행
                with gr.TabItem("실험 설정 및 실행"):
                    self._build_experiment_tab()
                
                # 탭 2: 결과 브라우징 및 시각화
                with gr.TabItem("결과 브라우징"):
                    self._build_results_tab()
                
                # 탭 3: Ground Truth 레이블링
                with gr.TabItem("Ground Truth 레이블링"):
                    self._build_gt_labeling_tab()
                
                # 탭 4: 평가 및 지표
                with gr.TabItem("평가 및 지표"):
                    self._build_evaluation_tab()
                
                # 탭 5: Annotation 동기화
                with gr.TabItem("Annotation 동기화"):
                    self._build_annotation_tab()
            
        return app
    
    def _build_experiment_tab(self):
        """실험 설정 및 실행 탭 구성"""
        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=self.available_categories,
                    label="카테고리 선택",
                    info="실험을 수행할 카테고리를 선택하세요."
                )
                
                model_radio = gr.Radio(
                    choices=["resnet", "clip", "dino"],
                    value="resnet",
                    label="모델 선택",
                    info="특징 추출에 사용할 모델을 선택하세요."
                )
                
                custom_input_dir = gr.Textbox(
                    label="입력 디렉토리 (선택사항)",
                    info="분류할 이미지가 있는 디렉토리 경로. 비워두면 카테고리의 preprocessed 디렉토리를 사용합니다."
                )
                
                n_shots_checkbox = gr.CheckboxGroup(
                    choices=["1", "5", "10", "30"],
                    value=["1", "5", "10", "30"],
                    label="N-Shot 설정",
                    info="실험에 사용할 N-Shot 값을 선택하세요."
                )
                
                thresholds_text = gr.Textbox(
                    value="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
                    label="Threshold 설정",
                    info="실험에 사용할 Threshold 값을 쉼표로 구분하여 입력하세요."
                )
                
                run_button = gr.Button("실험 실행", variant="primary")
            
            with gr.Column(scale=1):
                experiment_status = gr.Textbox(
                    label="실험 상태",
                    interactive=False,
                    lines=15
                )
        
        # 실험 실행 이벤트 핸들러
        def run_experiment(category, model, input_dir, n_shots, thresholds_str):
            if not category:
                return "카테고리를 선택하세요."
            
            status_text = f"실험 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            status_text += f"카테고리: {category}, 모델: {model}\n"
            
            # N-shot 설정 변환
            n_shots_list = [int(n) for n in n_shots]
            status_text += f"N-shots: {n_shots_list}\n"
            
            # Threshold 설정 변환
            try:
                thresholds_list = [float(t.strip()) for t in thresholds_str.split(",")]
                status_text += f"Thresholds: {thresholds_list}\n"
            except ValueError:
                return "Threshold 값 형식이 올바르지 않습니다."
            
            # 입력 디렉토리 설정
            if input_dir.strip():
                status_text += f"입력 디렉토리: {input_dir}\n"
            else:
                input_dir = None
                status_text += "입력 디렉토리: 기본값 (카테고리의 preprocessed 디렉토리)\n"
            
            status_text += "\n실험 실행 중...\n"
            yield status_text
            
            # 모델 타입 확인 (필요한 패키지가 설치되어 있는지)
            if model == "clip":
                try:
                    import clip
                except ImportError:
                    error_msg = "CLIP 패키지가 설치되어 있지 않습니다.\n"
                    error_msg += "설치 방법: pip install 'clip @ git+https://github.com/openai/CLIP.git'\n"
                    error_msg += "few_shot_requirements.txt에 있는 모든 패키지를 설치하세요: pip install -r few_shot_requirements.txt"
                    status_text += f"\n❌ 오류 발생: {error_msg}"
                    return status_text
            
            # 실험 객체 초기화
            try:
                self.category_name = category
                self.experiment = FewShotExperiment(category_name=category, model_name=model)
                
                # 사용자 설정으로 실험 객체 업데이트
                self.experiment.n_shots = n_shots_list
                self.experiment.thresholds = thresholds_list
                
                # 실험 실행
                self.experiment.run_all_experiments(input_dir)
                status_text += "\n✅ 실험 완료!\n"
                status_text += f"결과 경로: {self.experiment.results_dir}\n"
                status_text += f"총 {len(n_shots_list) * len(thresholds_list)}개 실험 조합 실행됨\n"
                status_text += "\n결과 탭에서 실험 결과를 확인할 수 있습니다."
            except ImportError as e:
                status_text += f"\n❌ 패키지 오류: {str(e)}\n"
                status_text += "필요한 패키지가 설치되어 있지 않습니다. few_shot_requirements.txt에 있는 모든 패키지를 설치하세요:\n"
                status_text += "pip install -r few_shot_requirements.txt"
            except Exception as e:
                status_text += f"\n❌ 실험 실행 중 오류 발생: {str(e)}\n"
                import traceback
                status_text += f"상세 오류: {traceback.format_exc()}"
            
            return status_text
        
        run_button.click(
            run_experiment,
            inputs=[category_dropdown, model_radio, custom_input_dir, n_shots_checkbox, thresholds_text],
            outputs=experiment_status
        )
    
    def _build_results_tab(self):
        """결과 브라우징 탭 구성"""
        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=self.available_categories,
                    label="카테고리 선택",
                    info="결과를 확인할 카테고리를 선택하세요."
                )
                
                load_results_button = gr.Button("결과 로드")
                
                experiment_dropdown = gr.Dropdown(
                    choices=[],
                    label="실험 선택",
                    info="확인할 실험을 선택하세요."
                )
                
                metrics_display = gr.JSON(
                    label="실험 요약 정보"
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("예측 결과"):
                        predictions_df = gr.Dataframe(
                            label="예측 결과"
                        )
                    
                    with gr.TabItem("Confusion Matrix"):
                        confusion_matrix_plot = gr.Image(
                            label="Confusion Matrix",
                            type="filepath"
                        )
        
        # 카테고리 선택 시 결과 로드 이벤트 핸들러
        def load_category_results(category):
            if not category:
                return gr.update(choices=[], value=None), {}
            
            # 실험 객체 초기화
            self.category_name = category
            self.experiment = FewShotExperiment(category_name=category)
            
            # 실험 요약 파일 확인
            summary_file = os.path.join(self.experiment.results_dir, "experiment_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    # 실험 ID 목록 생성
                    experiment_ids = list(summary.keys())
                    return gr.update(choices=experiment_ids, value=None), {}
                except Exception as e:
                    return gr.update(choices=[], value=None), {"error": f"실험 요약 로드 실패: {str(e)}"}
            else:
                return gr.update(choices=[], value=None), {"info": f"실험 요약 파일이 없습니다: {summary_file}"}
        
        # 실험 선택 시 결과 로드 이벤트 핸들러
        def load_experiment_results(experiment_id, category):
            if not experiment_id or not category:
                return None, None, {}
            
            # 실험 객체 초기화 (아직 초기화되지 않은 경우)
            if self.experiment is None or self.category_name != category:
                self.category_name = category
                self.experiment = FewShotExperiment(category_name=category)
            
            # 실험 ID에서 shot과 threshold 추출
            parts = experiment_id.split('_')
            n_shot = int(parts[1])
            threshold = float(parts[3])
            
            # 실험 결과 디렉토리 경로
            experiment_dir = os.path.join(self.experiment.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
            predictions_file = os.path.join(experiment_dir, "predictions.csv")
            confusion_matrix_file = os.path.join(experiment_dir, "confusion_matrix.png")
            
            # 예측 결과 로드
            if os.path.exists(predictions_file):
                try:
                    predictions_df = pd.read_csv(predictions_file)
                except Exception as e:
                    predictions_df = pd.DataFrame({"error": [f"예측 결과 로드 실패: {str(e)}"]})
            else:
                predictions_df = pd.DataFrame({"info": [f"예측 결과 파일이 없습니다: {predictions_file}"]})
            
            # Confusion Matrix 로드
            cm_path = None
            if os.path.exists(confusion_matrix_file):
                cm_path = confusion_matrix_file
            
            # 실험 요약 정보
            summary_file = os.path.join(self.experiment.results_dir, "experiment_summary.json")
            metrics = {}
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    if experiment_id in summary:
                        metrics = summary[experiment_id]
                except Exception:
                    pass
            
            # 평가 지표 로드
            metrics_file = os.path.join(experiment_dir, "evaluation_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        evaluation_metrics = json.load(f)
                    metrics["evaluation"] = evaluation_metrics
                except Exception:
                    pass
            
            self.current_experiment_id = experiment_id
            
            return predictions_df, cm_path, metrics
        
        # 이벤트 연결
        load_results_button.click(
            load_category_results,
            inputs=[category_dropdown],
            outputs=[experiment_dropdown, metrics_display]
        )
        
        experiment_dropdown.change(
            load_experiment_results,
            inputs=[experiment_dropdown, category_dropdown],
            outputs=[predictions_df, confusion_matrix_plot, metrics_display]
        )
    
    def _build_gt_labeling_tab(self):
        """Ground Truth 레이블링 탭 구성"""
        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=self.available_categories,
                    label="카테고리 선택",
                    info="Ground Truth를 생성할 카테고리를 선택하세요."
                )
                
                load_category_button = gr.Button("카테고리 로드")
                
                experiment_dropdown = gr.Dropdown(
                    choices=[],
                    label="실험 선택",
                    info="Ground Truth를 생성할 실험을 선택하세요."
                )
                
                load_experiment_button = gr.Button("실험 로드")
                
                class_dropdown = gr.Dropdown(
                    choices=["Unknown"],
                    label="클래스 선택",
                    info="이미지에 할당할 클래스를 선택하세요."
                )
                
                save_gt_button = gr.Button("Ground Truth 저장", variant="primary")
                gt_status = gr.Textbox(label="Ground Truth 저장 상태", interactive=False)
            
            with gr.Column(scale=2):
                with gr.Row():
                    # 이미지 선택 갤러리
                    image_gallery = gr.Gallery(
                        label="이미지 갤러리",
                        columns=4,
                        height=600,
                        object_fit="contain"
                    )
                
                with gr.Row():
                    selected_image = gr.Image(
                        label="선택된 이미지",
                        type="filepath",
                        height=300
                    )
                    
                    image_info = gr.JSON(
                        label="이미지 정보"
                    )
        
        # 카테고리 로드 이벤트 핸들러
        def load_category_gt(category):
            if not category:
                return gr.update(choices=[], value=None)
            
            # 실험 객체 초기화
            self.category_name = category
            self.experiment = FewShotExperiment(category_name=category)
            
            # 실험 요약 파일 확인
            summary_file = os.path.join(self.experiment.results_dir, "experiment_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    # 실험 ID 목록 생성
                    experiment_ids = list(summary.keys())
                    return gr.update(choices=experiment_ids, value=None)
                except Exception as e:
                    return gr.update(choices=[], value=None)
            else:
                return gr.update(choices=[], value=None)
        
        # 실험 로드 이벤트 핸들러
        def load_experiment_gt(experiment_id, category):
            if not experiment_id or not category:
                return [], gr.update(choices=["Unknown"], value=None), {}
            
            # 실험 객체 초기화 (아직 초기화되지 않은 경우)
            if self.experiment is None or self.category_name != category:
                self.category_name = category
                self.experiment = FewShotExperiment(category_name=category)
            
            # 실험 ID에서 shot과 threshold 추출
            parts = experiment_id.split('_')
            n_shot = int(parts[1])
            threshold = float(parts[3])
            
            # 실험 결과 디렉토리 경로
            experiment_dir = os.path.join(self.experiment.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
            predictions_file = os.path.join(experiment_dir, "predictions.csv")
            
            # 예측 결과 로드
            if not os.path.exists(predictions_file):
                return [], gr.update(choices=["Unknown"], value=None), {"error": f"예측 결과 파일이 없습니다: {predictions_file}"}
            
            try:
                predictions_df = pd.read_csv(predictions_file)
            except Exception as e:
                return [], gr.update(choices=["Unknown"], value=None), {"error": f"예측 결과 로드 실패: {str(e)}"}
            
            # 이미지 경로 목록 및 클래스 목록 생성
            image_paths = predictions_df["image_path"].tolist()
            unique_classes = sorted(predictions_df["predicted_class"].unique().tolist())
            if "Unknown" not in unique_classes:
                unique_classes.append("Unknown")
            
            # Ground Truth 파일 확인
            gt_file = os.path.join(experiment_dir, "ground_truth.csv")
            if os.path.exists(gt_file):
                try:
                    gt_df = pd.read_csv(gt_file)
                    self.ground_truth_data = dict(zip(gt_df["image_filename"], gt_df["true_class"]))
                except Exception:
                    self.ground_truth_data = {}
            else:
                self.ground_truth_data = {}
            
            self.current_experiment_id = experiment_id
            
            # 이미지 갤러리 항목 생성
            gallery_items = []
            for path in image_paths:
                if os.path.exists(path):
                    filename = os.path.basename(path)
                    predicted_class = predictions_df.loc[predictions_df["image_path"] == path, "predicted_class"].iloc[0]
                    confidence = predictions_df.loc[predictions_df["image_path"] == path, "confidence"].iloc[0]
                    
                    true_class = self.ground_truth_data.get(filename, None)
                    label = f"예측: {predicted_class} ({confidence:.2f})"
                    if true_class:
                        label += f"\nGT: {true_class}"
                    
                    gallery_items.append((path, label))
            
            return gallery_items, gr.update(choices=unique_classes, value=None), {}
        
        # 이미지 선택 이벤트 핸들러
        def select_image(evt: gr.SelectData, gallery, category, experiment_id):
            if not gallery or not category or not experiment_id:
                return None, {}
            
            selected_index = evt.index
            if selected_index < 0 or selected_index >= len(gallery):
                return None, {}
            
            image_path = gallery[selected_index][0]
            image_filename = os.path.basename(image_path)
            
            # 실험 결과 디렉토리 경로
            parts = experiment_id.split('_')
            n_shot = int(parts[1])
            threshold = float(parts[3])
            experiment_dir = os.path.join(self.experiment.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
            predictions_file = os.path.join(experiment_dir, "predictions.csv")
            
            # 예측 결과 로드
            try:
                predictions_df = pd.read_csv(predictions_file)
                pred_row = predictions_df[predictions_df["image_path"] == image_path].iloc[0]
                
                # 이미지 정보 생성
                image_info = {
                    "filename": image_filename,
                    "path": image_path,
                    "predicted_class": pred_row["predicted_class"],
                    "confidence": pred_row["confidence"],
                    "ground_truth": self.ground_truth_data.get(image_filename, None)
                }
                
                # 클래스별 점수 추가
                for col in pred_row.index:
                    if col.startswith("score_"):
                        class_name = col[6:]
                        image_info[f"score_{class_name}"] = pred_row[col]
                
                return image_path, image_info
            except Exception as e:
                return image_path, {"error": f"이미지 정보 로드 실패: {str(e)}"}
        
        # 클래스 할당 이벤트 핸들러
        def assign_class(selected_image, selected_class):
            if not selected_image or not selected_class:
                return {}
            
            image_filename = os.path.basename(selected_image)
            self.ground_truth_data[image_filename] = selected_class
            
            return {"filename": image_filename, "assigned_class": selected_class}
        
        # Ground Truth 저장 이벤트 핸들러
        def save_ground_truth(category, experiment_id):
            if not category or not experiment_id or not self.ground_truth_data:
                return "Ground Truth 데이터가 없습니다."
            
            try:
                # 실험 객체 초기화 (아직 초기화되지 않은 경우)
                if self.experiment is None or self.category_name != category:
                    self.category_name = category
                    self.experiment = FewShotExperiment(category_name=category)
                
                # Ground Truth 저장
                self.experiment.create_ground_truth(experiment_id, self.ground_truth_data)
                
                return f"Ground Truth 데이터 저장 완료: {len(self.ground_truth_data)}개 이미지"
            except Exception as e:
                return f"Ground Truth 데이터 저장 실패: {str(e)}"
        
        # 이벤트 연결
        load_category_button.click(
            load_category_gt,
            inputs=[category_dropdown],
            outputs=[experiment_dropdown]
        )
        
        load_experiment_button.click(
            load_experiment_gt,
            inputs=[experiment_dropdown, category_dropdown],
            outputs=[image_gallery, class_dropdown, image_info]
        )
        
        image_gallery.select(
            select_image,
            inputs=[image_gallery, category_dropdown, experiment_dropdown],
            outputs=[selected_image, image_info]
        )
        
        class_dropdown.change(
            assign_class,
            inputs=[selected_image, class_dropdown],
            outputs=[image_info]
        )
        
        save_gt_button.click(
            save_ground_truth,
            inputs=[category_dropdown, experiment_dropdown],
            outputs=[gt_status]
        )
    
    def _build_evaluation_tab(self):
        """평가 및 지표 탭 구성"""
        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=self.available_categories,
                    label="카테고리 선택",
                    info="평가할 카테고리를 선택하세요."
                )
                
                load_category_button = gr.Button("카테고리 로드")
                
                experiment_dropdown = gr.Dropdown(
                    choices=[],
                    label="실험 선택",
                    info="평가할 실험을 선택하세요."
                )
                
                evaluate_button = gr.Button("평가 실행", variant="primary")
                
                metrics_json = gr.JSON(
                    label="평가 지표"
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Confusion Matrix"):
                        confusion_matrix_plot = gr.Image(
                            label="Confusion Matrix",
                            type="filepath"
                        )
                    
                    with gr.TabItem("클래스별 지표"):
                        class_metrics_df = gr.Dataframe(
                            label="클래스별 지표"
                        )
                    
                    with gr.TabItem("Binary Confusion Matrices"):
                        binary_cm_gallery = gr.Gallery(
                            label="Binary Confusion Matrices",
                            columns=2,
                            height=600,
                            object_fit="contain"
                        )
        
        # 카테고리 로드 이벤트 핸들러
        def load_category_eval(category):
            if not category:
                return gr.update(choices=[], value=None)
            
            # 실험 객체 초기화
            self.category_name = category
            self.experiment = FewShotExperiment(category_name=category)
            
            # 실험 요약 파일 확인
            summary_file = os.path.join(self.experiment.results_dir, "experiment_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    # 실험 ID 목록 생성
                    experiment_ids = list(summary.keys())
                    return gr.update(choices=experiment_ids, value=None)
                except Exception as e:
                    return gr.update(choices=[], value=None)
            else:
                return gr.update(choices=[], value=None)
        
        # 평가 실행 이벤트 핸들러
        def run_evaluation(category, experiment_id):
            if not category or not experiment_id:
                return {}, None, None, []
            
            # 실험 객체 초기화 (아직 초기화되지 않은 경우)
            if self.experiment is None or self.category_name != category:
                self.category_name = category
                self.experiment = FewShotExperiment(category_name=category)
            
            try:
                # 평가 실행
                metrics = self.experiment.evaluate_experiment(experiment_id)
                if not metrics:
                    return {"error": "평가 실패. Ground Truth 파일이 있는지 확인하세요."}, None, None, []
                
                # 실험 ID에서 shot과 threshold 추출
                parts = experiment_id.split('_')
                n_shot = int(parts[1])
                threshold = float(parts[3])
                
                # 실험 결과 디렉토리 경로
                experiment_dir = os.path.join(self.experiment.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
                
                # Confusion Matrix 파일 경로
                cm_file = os.path.join(experiment_dir, "confusion_matrix.png")
                
                # 클래스별 지표 생성
                class_metrics = []
                for cls in metrics.get("classes", []):
                    class_metric = {
                        "class": cls,
                        "fallout": metrics.get("class_fallouts", {}).get(cls, 0)
                    }
                    
                    # Binary metrics 파일 확인
                    binary_metrics_file = os.path.join(
                        experiment_dir, 
                        "binary_confusion_matrices", 
                        f"binary_metrics_{cls.replace(' ', '_')}.json"
                    )
                    
                    if os.path.exists(binary_metrics_file):
                        try:
                            with open(binary_metrics_file, 'r') as f:
                                binary_metrics = json.load(f)
                            
                            class_metric.update({
                                "precision": binary_metrics.get("precision", 0),
                                "recall": binary_metrics.get("recall", 0),
                                "specificity": binary_metrics.get("specificity", 0),
                                "f1_score": binary_metrics.get("f1_score", 0),
                                "true_positive": binary_metrics.get("true_positive", 0),
                                "false_positive": binary_metrics.get("false_positive", 0),
                                "false_negative": binary_metrics.get("false_negative", 0),
                                "true_negative": binary_metrics.get("true_negative", 0)
                            })
                        except Exception:
                            pass
                    
                    class_metrics.append(class_metric)
                
                class_metrics_df = pd.DataFrame(class_metrics)
                
                # Binary Confusion Matrix 갤러리 항목 생성
                binary_cm_dir = os.path.join(experiment_dir, "binary_confusion_matrices")
                binary_cm_items = []
                
                if os.path.exists(binary_cm_dir):
                    for cls in metrics.get("classes", []):
                        binary_cm_file = os.path.join(
                            binary_cm_dir, 
                            f"binary_cm_{cls.replace(' ', '_')}.png"
                        )
                        
                        if os.path.exists(binary_cm_file):
                            binary_cm_items.append((binary_cm_file, f"Class: {cls}"))
                
                return metrics, cm_file, class_metrics_df, binary_cm_items
            except Exception as e:
                return {"error": f"평가 실행 중 오류 발생: {str(e)}"}, None, None, []
        
        # 이벤트 연결
        load_category_button.click(
            load_category_eval,
            inputs=[category_dropdown],
            outputs=[experiment_dropdown]
        )
        
        evaluate_button.click(
            run_evaluation,
            inputs=[category_dropdown, experiment_dropdown],
            outputs=[metrics_json, confusion_matrix_plot, class_metrics_df, binary_cm_gallery]
        )
    
    def _build_annotation_tab(self):
        """Annotation 동기화 탭 구성"""
        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=self.available_categories,
                    label="카테고리 선택",
                    info="Annotation을 동기화할 카테고리를 선택하세요."
                )
                
                load_category_button = gr.Button("카테고리 로드")
                
                experiment_dropdown = gr.Dropdown(
                    choices=[],
                    label="실험 선택",
                    info="Annotation을 동기화할 실험을 선택하세요."
                )
                
                sync_annotations_button = gr.Button("Annotation 동기화", variant="primary")
                
                sync_status = gr.Textbox(
                    label="동기화 상태",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Annotation 미리보기"):
                        annotation_preview = gr.JSON(
                            label="Annotation 미리보기"
                        )
                    
                    with gr.TabItem("Box 데이터 시각화"):
                        box_viz_image = gr.Image(
                            label="Box 데이터 시각화",
                            type="filepath"
                        )
        
        # 카테고리 로드 이벤트 핸들러
        def load_category_ann(category):
            if not category:
                return gr.update(choices=[], value=None)
            
            # 실험 객체 초기화
            self.category_name = category
            self.experiment = FewShotExperiment(category_name=category)
            
            # 실험 요약 파일 확인
            summary_file = os.path.join(self.experiment.results_dir, "experiment_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    # 실험 ID 목록 생성
                    experiment_ids = list(summary.keys())
                    return gr.update(choices=experiment_ids, value=None)
                except Exception as e:
                    return gr.update(choices=[], value=None)
            else:
                return gr.update(choices=[], value=None)
        
        # Annotation 동기화 이벤트 핸들러
        def sync_annotations(category, experiment_id):
            if not category or not experiment_id:
                return "카테고리와 실험을 선택하세요.", {}
            
            # 실험 객체 초기화 (아직 초기화되지 않은 경우)
            if self.experiment is None or self.category_name != category:
                self.category_name = category
                self.experiment = FewShotExperiment(category_name=category)
            
            try:
                # Annotation 동기화 실행
                self.experiment.sync_annotations(experiment_id)
                
                # 실험 ID에서 shot과 threshold 추출
                parts = experiment_id.split('_')
                n_shot = int(parts[1])
                threshold = float(parts[3])
                
                # 실험 결과 디렉토리 경로
                experiment_dir = os.path.join(self.experiment.results_dir, f"shot_{n_shot}", f"threshold_{threshold:.2f}")
                annotations_file = os.path.join(experiment_dir, "annotated_predictions.json")
                
                # Annotation 파일 확인
                if os.path.exists(annotations_file):
                    try:
                        with open(annotations_file, 'r') as f:
                            annotations = json.load(f)
                        
                        if annotations:
                            # 첫 번째 annotation 항목 미리보기
                            preview = annotations[0]
                            return f"Annotation 동기화 완료: {len(annotations)}개 항목", preview
                    except Exception as e:
                        return f"Annotation 파일 로드 실패: {str(e)}", {}
                
                return "Annotation 동기화 완료되었으나, 결과 파일을 찾을 수 없습니다.", {}
            except Exception as e:
                return f"Annotation 동기화 실패: {str(e)}", {}
        
        # 이벤트 연결
        load_category_button.click(
            load_category_ann,
            inputs=[category_dropdown],
            outputs=[experiment_dropdown]
        )
        
        sync_annotations_button.click(
            sync_annotations,
            inputs=[category_dropdown, experiment_dropdown],
            outputs=[sync_status, annotation_preview]
        )
    
    def launch(self, **kwargs):
        """웹앱 실행"""
        import logging
        logger = logging.getLogger("few_shot_webapp_launch")
        logger.setLevel(logging.DEBUG)
        
        # 기본 포트를 7865로 변경하고 다른 옵션 설정
        default_kwargs = {
            "server_port": 7865,  # 다른 포트 사용
            "share": False,       # 공유 링크 비활성화
            "debug": True,        # 디버그 모드 활성화
            "show_error": True,   # 에러 표시 활성화
            "server_name": "0.0.0.0"  # 모든 IP에서 접속 가능
        }
        
        # 사용자 지정 옵션이 기본값보다 우선
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        
        logger.debug(f"웹앱 실행 옵션: {kwargs}")
        
        try:
            self.app.launch(**kwargs)
        except Exception as e:
            logger.error(f"웹앱 실행 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 포트 충돌 가능성이 있는 경우 다른 포트 시도
            if "address already in use" in str(e).lower() or "socket.error" in str(e).lower():
                alt_port = kwargs.get("server_port", 7865) + 1
                logger.info(f"다른 포트로 재시도: {alt_port}")
                kwargs["server_port"] = alt_port
                try:
                    self.app.launch(**kwargs)
                except Exception as e2:
                    logger.error(f"대체 포트로도 실행 실패: {e2}")
                    logger.error(traceback.format_exc())


def main():
    """웹앱을 직접 실행"""
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("few_shot_webapp")
    
    try:
        logger.debug("웹앱 시작 시도...")
        webapp = FewShotWebApp()
        logger.debug("웹앱 객체 생성 완료")
        logger.debug(f"사용 가능한 카테고리: {webapp.available_categories}")
        webapp.launch(server_port=7860)
    except Exception as e:
        logger.error(f"웹앱 실행 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 