#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning 간소화된 웹앱

그라디오 업데이트 문제를 해결하고 포트 충돌을 방지하는 통합된 스크립트입니다.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
import glob
from pathlib import Path
from datetime import datetime

# 그라디오 환경 변수 설정
os.environ["GRADIO_SERVER_PORT"] = "7865"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fsl_webapp.log")
    ]
)
logger = logging.getLogger("fsl_webapp_simple")

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def main():
    """웹앱 실행 함수"""
    try:
        # 그라디오 임포트
        import gradio as gr
        logger.info(f"Gradio 버전: {gr.__version__}")
        logger.info(f"gr.update 존재 여부: {hasattr(gr, 'update')}")
        
        # FewShotExperiment 클래스 임포트
        try:
            from scripts.classifier_cosine_experiment import FewShotExperiment
            from scripts.data_utils import get_category_path, load_class_mapping
        except ImportError as e:
            logger.error(f"필요한 모듈을 임포트할 수 없습니다: {e}")
            sys.exit(1)
        
        # 사용 가능한 카테고리 목록 가져오기
        def get_available_categories():
            categories = []
            data_dir = os.path.join(PROJECT_ROOT, "data")
            if os.path.exists(data_dir):
                for item in os.listdir(data_dir):
                    category_path = os.path.join(data_dir, item)
                    if os.path.isdir(category_path):
                        support_dir = os.path.join(category_path, "2.support-set")
                        if os.path.exists(support_dir):
                            categories.append(item)
            return categories
        
        available_categories = get_available_categories()
        logger.info(f"사용 가능한 카테고리: {available_categories}")
        
        # 웹앱 클래스 정의
        class SimpleWebApp:
            def __init__(self):
                self.experiment = None
                self.category_name = None
                self.current_experiment_id = None
                self.ground_truth_data = {}
                self.available_categories = available_categories
                self.standard_shots = [1, 5, 10, 30]
            
            def load_category_results(self, category):
                logger.debug(f"load_category_results 호출: {category}")
                if not category:
                    return gr.update(choices=[], value=None), {}, "카테고리를 선택하세요."
                
                try:
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
                            logger.debug(f"실험 ID 목록: {experiment_ids}")
                            return gr.update(choices=experiment_ids, value=None), {}, f"{len(experiment_ids)}개 실험 로드 완료"
                        except Exception as e:
                            logger.error(f"실험 요약 로드 실패: {e}")
                            return gr.update(choices=[], value=None), {"error": f"실험 요약 로드 실패: {str(e)}"}, f"오류: {str(e)}"
                    else:
                        logger.warning(f"실험 요약 파일이 없습니다: {summary_file}")
                        return gr.update(choices=[], value=None), {"info": f"실험 요약 파일이 없습니다"}, "실험 요약 파일이 없습니다."
                except Exception as e:
                    logger.error(f"카테고리 로드 중 오류: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return gr.update(choices=[], value=None), {"error": str(e)}, f"오류: {str(e)}"
            
            def load_experiment_results(self, experiment_id, category):
                logger.debug(f"load_experiment_results 호출: {experiment_id}, {category}")
                if not experiment_id or not category:
                    return None, None, {}, "실험 ID와 카테고리를 선택하세요.", []
                
                try:
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
                    
                    self.current_experiment_id = experiment_id
                    
                    # Support set 이미지 로드
                    support_images = self.load_support_images(category, n_shot)
                    
                    return predictions_df, cm_path, metrics, f"실험 {experiment_id} 결과 로드 완료", support_images
                except Exception as e:
                    logger.error(f"실험 결과 로드 중 오류: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None, None, {"error": str(e)}, f"오류: {str(e)}", []
            
            def load_support_images(self, category, shot):
                """
                지정된 카테고리와 shot에 대한 support 이미지 로드
                
                Args:
                    category: 카테고리 이름
                    shot: shot 값 (1, 5, 10, 30)
                    
                Returns:
                    Gradio Gallery 형식의 이미지 경로와 캡션 튜플 목록
                """
                logger.debug(f"Support set 이미지 로드: {category}, {shot}-shot")
                category_path = get_category_path(category)
                
                # 구조화된 support set 확인
                structured_support_dir = os.path.join(category_path, "2.support-set-structured")
                use_structured = os.path.exists(structured_support_dir)
                
                if use_structured:
                    # 구조화된 support set에서 가져오기
                    shot_dir = os.path.join(structured_support_dir, f"{shot}-shot")
                    
                    if os.path.exists(shot_dir):
                        logger.info(f"구조화된 support set 사용: {shot_dir}")
                        support_images = []
                        
                        for class_name in sorted(os.listdir(shot_dir)):
                            class_dir = os.path.join(shot_dir, class_name)
                            if os.path.isdir(class_dir):
                                image_files = sorted(
                                    glob.glob(os.path.join(class_dir, "*.jpg")) + 
                                    glob.glob(os.path.join(class_dir, "*.jpeg")) + 
                                    glob.glob(os.path.join(class_dir, "*.png")) + 
                                    glob.glob(os.path.join(class_dir, "*.webp"))
                                )
                                
                                # 각 이미지에 대한 캡션 추가
                                for i, img_path in enumerate(image_files):
                                    img_filename = os.path.basename(img_path)
                                    caption = f"클래스: {class_name} | {i+1}/{len(image_files)}"
                                    support_images.append((img_path, caption))
                        
                        return support_images
                    else:
                        logger.warning(f"구조화된 {shot}-shot 디렉토리를 찾을 수 없습니다: {shot_dir}")
                
                # 원본 support set에서 가져오기
                support_dir = os.path.join(category_path, "2.support-set")
                if not os.path.exists(support_dir):
                    logger.error(f"Support set 디렉토리를 찾을 수 없습니다: {support_dir}")
                    return []
                
                logger.info(f"원본 support set 사용: {support_dir}")
                support_images = []
                
                for class_name in sorted(os.listdir(support_dir)):
                    class_dir = os.path.join(support_dir, class_name)
                    if os.path.isdir(class_dir):
                        image_files = sorted(
                            glob.glob(os.path.join(class_dir, "*.jpg")) + 
                            glob.glob(os.path.join(class_dir, "*.jpeg")) + 
                            glob.glob(os.path.join(class_dir, "*.png")) + 
                            glob.glob(os.path.join(class_dir, "*.webp"))
                        )
                        
                        # shot 수에 맞게 이미지 선택
                        selected_images = image_files[:shot] if len(image_files) >= shot else image_files
                        
                        # 각 이미지에 대한 캡션 추가
                        for i, img_path in enumerate(selected_images):
                            img_filename = os.path.basename(img_path)
                            caption = f"클래스: {class_name} | {i+1}/{len(selected_images)}"
                            support_images.append((img_path, caption))
                
                return support_images
        
        # 웹앱 인스턴스 생성
        webapp = SimpleWebApp()
        
        # UI 구성
        with gr.Blocks(title="Few-Shot Learning 통합 앱") as demo:
            gr.Markdown("# Few-Shot Learning 통합 앱")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 상태 표시
                    status_text = gr.Textbox(
                        label="상태 메시지",
                        interactive=False
                    )
                    
                    # 카테고리 선택
                    category_dropdown = gr.Dropdown(
                        choices=available_categories,
                        label="카테고리 선택",
                        info="결과를 확인할 카테고리를 선택하세요."
                    )
                    
                    load_results_button = gr.Button("결과 로드")
                    
                    # 실험 선택
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
                        with gr.TabItem("분류 결과"):
                            # 예측 결과
                            predictions_df = gr.Dataframe(
                                label="예측 결과"
                            )
                            
                            # Confusion Matrix
                            confusion_matrix_plot = gr.Image(
                                label="Confusion Matrix",
                                type="filepath"
                            )
                        
                        with gr.TabItem("Support Set 이미지"):
                            gr.Markdown("### 현재 실험에 사용된 Support Set 이미지")
                            gr.Markdown("선택한 N-shot 설정에 따라 분류에 사용된 실제 Support Set 이미지를 표시합니다.")
                            
                            # Support Set 갤러리
                            support_gallery = gr.Gallery(
                                label="Support Set 이미지",
                                columns=4,
                                height=600,
                                object_fit="contain",
                                show_label=True
                            )
            
            # 이벤트 연결
            load_results_button.click(
                webapp.load_category_results,
                inputs=[category_dropdown],
                outputs=[experiment_dropdown, metrics_display, status_text]
            )
            
            experiment_dropdown.change(
                webapp.load_experiment_results,
                inputs=[experiment_dropdown, category_dropdown],
                outputs=[predictions_df, confusion_matrix_plot, metrics_display, status_text, support_gallery]
            )
        
        # 웹앱 실행
        logger.info("웹앱 실행 중...")
        demo.launch(server_port=7865, server_name="0.0.0.0", share=False)
    
    except Exception as e:
        logger.error(f"웹앱 실행 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 