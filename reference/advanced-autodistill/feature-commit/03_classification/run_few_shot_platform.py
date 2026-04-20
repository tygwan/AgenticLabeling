#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning Experiment & Evaluation Platform 실행 스크립트

이 스크립트는 Few-Shot Learning 실험 및 평가 플랫폼을 실행하는 메인 진입점입니다.
웹 앱 모드 또는 CLI 모드로 실행할 수 있습니다.

사용법:
    - 웹 앱 모드 실행:
        python run_few_shot_platform.py --webapp
        
    - CLI 모드로 실험 실행:
        python run_few_shot_platform.py --cli --category test_category --model resnet
        
    - Support Set 뷰어를 포함한 웹 앱 모드 실행:
        python run_few_shot_platform.py --mywebapp
"""

import os
import sys
import argparse
import traceback
import glob
from pathlib import Path
import json
import logging
import cv2
import numpy as np
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # 로깅 레벨을 DEBUG로 변경
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler("fsl_platform.log")  # 파일 출력 추가
    ]
)
logger = logging.getLogger("fsl_platform")

# 프로젝트 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir) if os.path.dirname(current_dir) else current_dir
sys.path.append(project_dir)

# Import project utilities
try:
    from scripts.data_utils import get_category_path, load_class_mapping
    from scripts.classifier_cosine import CosineSimilarityClassifier
    from scripts.classifier_cosine_experiment import FewShotExperiment
except ImportError as e:
    logger.error(f"필요한 모듈을 임포트할 수 없습니다: {e}")
    logger.error("scripts 디렉토리에 필요한 모듈이 있는지 확인하세요.")
    
    def get_category_path(category_name):
        return os.path.join("data", category_name)
    
    def load_class_mapping(category_name):
        mapping_file = os.path.join(get_category_path(category_name), "class_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}

class FewShotPlatform:
    """Few-Shot Learning Platform 클래스 - Support Set 이미지 표시 기능을 제공합니다."""
    
    def __init__(self):
        """초기화 함수"""
        # Gradio 패키지 가져오기
        try:
            import gradio as gr
            self.gr = gr
        except ImportError:
            logger.error("Gradio 패키지를 설치해야 합니다. pip install gradio")
            sys.exit(1)
            
        # 카테고리 목록 로드
        self.categories = self._get_available_categories()
        # 표준 shot 값
        self.standard_shots = [1, 5, 10, 30]
        # 표준 threshold 값
        self.standard_thresholds = [round(t, 2) for t in np.arange(0.50, 0.95, 0.05)]
        # 분류기 객체
        self.classifier = None
        # 웹 인터페이스 설정
        self.setup_interface()
    
    def _get_available_categories(self):
        """사용 가능한 카테고리 목록 가져오기"""
        try:
            categories = []
            data_dir = os.path.join(project_dir, "data")
            if os.path.exists(data_dir):
                for item in os.listdir(data_dir):
                    full_path = os.path.join(data_dir, item)
                    if os.path.isdir(full_path):
                        # support set 폴더가 있는지 확인
                        support_dir = os.path.join(full_path, "2.support-set")
                        if os.path.exists(support_dir):
                            categories.append(item)
            return sorted(categories)
        except Exception as e:
            logger.error(f"카테고리 목록 로드 중 오류 발생: {e}")
            return []
    
    def load_support_images(self, category, shot, use_structured=True):
        """
        지정된 카테고리와 shot에 대한 support 이미지 로드
        
        Args:
            category: 카테고리 이름
            shot: shot 값 (1, 5, 10, 30)
            use_structured: 구조화된 support set 사용 여부
            
        Returns:
            클래스별 이미지 경로 사전
        """
        category_path = get_category_path(category)
        
        if use_structured:
            # 구조화된 support set에서 가져오기
            structured_dir = os.path.join(category_path, "2.support-set-structured")
            shot_dir = os.path.join(structured_dir, f"{shot}-shot")
            
            if os.path.exists(shot_dir):
                support_images = {}
                for class_name in os.listdir(shot_dir):
                    class_dir = os.path.join(shot_dir, class_name)
                    if os.path.isdir(class_dir):
                        image_files = sorted(
                            glob.glob(os.path.join(class_dir, "*.jpg")) + 
                            glob.glob(os.path.join(class_dir, "*.jpeg")) + 
                            glob.glob(os.path.join(class_dir, "*.png")) + 
                            glob.glob(os.path.join(class_dir, "*.webp"))
                        )
                        support_images[class_name] = image_files
                return support_images
            else:
                logger.warning(f"구조화된 {shot}-shot 디렉토리를 찾을 수 없습니다: {shot_dir}")
                logger.warning("원본 support set에서 가져옵니다.")
        
        # 원본 support set에서 가져오기
        support_dir = os.path.join(category_path, "2.support-set")
        if not os.path.exists(support_dir):
            logger.error(f"Support set 디렉토리를 찾을 수 없습니다: {support_dir}")
            return {}
        
        support_images = {}
        
        # 원본 디렉토리 구조 확인 (class_0, class_1, class_2, class_3 형식)
        class_pattern = re.compile(r'^class_\d+$')
        
        # 디렉토리 목록 가져오기
        try:
            class_dirs = [d for d in os.listdir(support_dir) 
                        if os.path.isdir(os.path.join(support_dir, d)) and class_pattern.match(d)]
            
            # 디렉토리가 없거나 패턴이 일치하지 않는 경우 일반 방식으로 진행
            if not class_dirs:
                logger.warning("class_N 형식의 디렉토리를 찾을 수 없습니다. 일반 방식으로 진행합니다.")
                for class_name in os.listdir(support_dir):
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
                        support_images[class_name] = selected_images
            else:
                # class_N 형식의 디렉토리가 있는 경우
                logger.info(f"class_N 형식의 디렉토리를 찾았습니다: {class_dirs}")
                
                # 클래스 이름으로 정렬 (class_0, class_1, class_2, ...)
                class_dirs.sort(key=lambda x: int(x.split('_')[1]))
                
                for class_name in class_dirs:
                    class_dir = os.path.join(support_dir, class_name)
                    image_files = sorted(
                        glob.glob(os.path.join(class_dir, "*.jpg")) + 
                        glob.glob(os.path.join(class_dir, "*.jpeg")) + 
                        glob.glob(os.path.join(class_dir, "*.png")) + 
                        glob.glob(os.path.join(class_dir, "*.webp"))
                    )
                    
                    # shot 수에 맞게 이미지 선택
                    selected_images = image_files[:shot] if len(image_files) >= shot else image_files
                    
                    # 로그에 선택된 이미지 정보 출력
                    logger.debug(f"클래스 {class_name}에서 {len(selected_images)}/{shot} 이미지 선택됨")
                    
                    # 클래스별로 이미지 저장
                    support_images[class_name] = selected_images
        except Exception as e:
            logger.error(f"Support set 디렉토리 처리 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
        
        return support_images
    
    def format_support_images_for_gallery(self, support_images):
        """
        Support 이미지를 Gradio 갤러리 형식으로 변환
        
        Args:
            support_images: 클래스별 이미지 경로 사전
            
        Returns:
            Gradio 갤러리용 이미지 목록 (경로, 캡션)
        """
        gallery_images = []
        for class_name, images in support_images.items():
            for i, img_path in enumerate(images):
                # 이미지 파일명 추출
                img_filename = os.path.basename(img_path)
                caption = f"클래스: {class_name} | 이미지 {i+1}/{len(images)} | {img_filename}"
                gallery_images.append((img_path, caption))
        
        return gallery_images
    
    def update_gallery(self, category, shot, use_structured):
        """
        갤러리 업데이트 함수
        
        Args:
            category: 카테고리 이름
            shot: shot 값
            use_structured: 구조화된 support set 사용 여부
            
        Returns:
            갤러리 이미지, 통계 텍스트
        """
        if not category:
            return [], "카테고리를 선택하세요."
        
        try:
            shot = int(shot)
            support_images = self.load_support_images(category, shot, use_structured)
            
            if not support_images:
                return [], f"Support set 이미지를 찾을 수 없습니다: {category} ({shot}-shot)"
            
            # 갤러리용 이미지 형식으로 변환
            gallery_images = self.format_support_images_for_gallery(support_images)
            
            # 통계 정보 생성
            stats = [f"카테고리: {category}", f"Shot 수: {shot}"]
            
            if use_structured:
                source_type = "구조화된 Support Set"
                category_path = get_category_path(category)
                structured_dir = os.path.join(category_path, "2.support-set-structured")
                shot_dir = os.path.join(structured_dir, f"{shot}-shot")
                stats.append(f"소스: {source_type} ({shot_dir})")
            else:
                source_type = "원본 Support Set"
                stats.append(f"소스: {source_type}")
            
            stats.append("")
            stats.append("클래스별 이미지 수:")
            for class_name, images in support_images.items():
                status = "✅" if len(images) >= shot else "⚠️"
                stats.append(f"- {class_name}: {len(images)}/{shot} {status}")
            
            return gallery_images, "\n".join(stats)
        
        except Exception as e:
            logger.error(f"갤러리 업데이트 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], f"오류 발생: {str(e)}"
    
    def initialize_classifier(self, category, shot, threshold, model_name, use_structured):
        """
        분류기 초기화 함수
        
        Args:
            category: 카테고리 이름
            shot: shot 값
            threshold: 유사도 임계값
            model_name: 특징 추출 모델 이름
            use_structured: 구조화된 support set 사용 여부
            
        Returns:
            초기화 메시지
        """
        if not category:
            return "카테고리를 선택하세요."
        
        try:
            shot = int(shot)
            threshold = float(threshold)
            
            # 분류기 경로 검증
            scripts_dir = os.path.join(project_dir, "scripts")
            sys.path.append(scripts_dir)
            
            # 분류기 초기화
            logger.info(f"분류기 초기화 중: {model_name}, shot={shot}, threshold={threshold}")
            
            try:
                # 직접 CosineSimilarityClassifier 클래스 가져오기
                self.classifier = CosineSimilarityClassifier(
                    model_name=model_name,
                    k_shot=shot,
                    similarity_threshold=threshold
                )
                
                # Support set 로드
                if use_structured:
                    # 구조화된 support set 사용
                    shot_dir = f"{shot}-shot"
                    self.classifier.load_support_set(category, shot_dir=shot_dir)
                else:
                    # 원본 support set 사용 - class_N 형식 처리 지원
                    try:
                        # 먼저 원본 방식으로 시도
                        self.classifier.load_support_set(category)
                    except Exception as load_err:
                        logger.warning(f"기본 방식으로 support set 로드 실패: {load_err}")
                        logger.info("수동 방식으로 support set 로드를 시도합니다...")
                        
                        # 수동으로 support set 로드 시도
                        category_path = get_category_path(category)
                        support_dir = os.path.join(category_path, "2.support-set")
                        
                        # 이미 로드된 support_images 사용
                        support_images = self.load_support_images(category, shot, use_structured=False)
                        
                        # 수동으로 support set 설정
                        if hasattr(self.classifier, 'set_support_images_directly'):
                            self.classifier.set_support_images_directly(support_images)
                            logger.info("수동 방식으로 support set 로드 성공")
                        else:
                            # 대체 방법: 클래스 변수에 직접 설정
                            self.classifier.support_images = support_images
                            self.classifier.support_dir = support_dir
                            self.classifier.category = category
                            logger.info("대체 방식으로 support set 설정됨")
                
                # 특징 추출
                self.classifier.extract_support_features()
                
                return f"분류기 초기화 완료: {category}, {shot}-shot, threshold={threshold}, model={model_name}"
            
            except Exception as inner_e:
                logger.error(f"분류기 초기화 실패(내부 오류): {inner_e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 대체 경로 시도
                error_msg = str(inner_e)
                if "load_support_set" in error_msg or "extract_support_features" in error_msg:
                    return f"Support set 로드 실패: {error_msg}"
                else:
                    return f"분류기 초기화 실패: {error_msg}"
        
        except Exception as e:
            logger.error(f"분류기 초기화 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.classifier = None
            return f"오류 발생: {str(e)}"
    
    def classify_image(self, image):
        """
        이미지 분류 함수
        
        Args:
            image: 분류할 이미지 (PIL 이미지 또는 파일 경로)
            
        Returns:
            분류 결과 이미지, 결과 텍스트
        """
        if self.classifier is None:
            return image, "분류기가 초기화되지 않았습니다. 분류기를 먼저 초기화하세요."
        
        try:
            # 이미지가 None인 경우 처리
            if image is None:
                return None, "이미지가 없습니다. 이미지를 업로드하세요."
            
            # 이미지가 경로인지 PIL 이미지인지 확인
            image_path = None
            if isinstance(image, str) and os.path.exists(image):
                image_path = image
                pil_image = Image.open(image).convert('RGB')
            else:
                # 임시 파일로 저장
                try:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        image_path = tmp.name
                        image.save(image_path)
                    pil_image = image
                except Exception as e:
                    logger.error(f"이미지 저장 중 오류: {e}")
                    # 오류 발생시 기본 이미지 반환
                    return image, f"이미지 처리 중 오류: {str(e)}"
            
            # 이미지 분류
            try:
                # support set에 이미지가 없는 경우 처리
                if not self.classifier.support_features:
                    logger.error("Support set 특징이 추출되지 않았습니다.")
                    
                    # support_paths에는 있지만 특징이 추출되지 않은 경우
                    if self.classifier.support_paths:
                        logger.info("Support set 특징 추출을 시도합니다...")
                        self.classifier.extract_support_features()
                        
                        # 그래도 없는 경우
                        if not self.classifier.support_features:
                            return image, "Support set 특징을 추출할 수 없습니다. 분류기를 다시 초기화하세요."
                    else:
                        return image, "Support set 이미지가 로드되지 않았습니다. 분류기를 다시 초기화하세요."
                
                # Support set에 class_N 형식이 포함된 경우 처리
                class_pattern = re.compile(r'^class_\d+$')
                has_class_n_format = any(class_pattern.match(cls) for cls in self.classifier.support_features.keys())
                
                # 이미지 분류 수행
                result = self.classifier.classify_image(image_path)
                
                # class_N 형식인 경우 결과에 클래스 ID 추가
                if has_class_n_format and result["class"] is not None and result["class"] != "Unknown":
                    if class_pattern.match(result["class"]):
                        class_id = result["class"].split('_')[1]
                        result["class_id"] = class_id
                        result["display_class"] = f"Class {class_id}"
                    else:
                        result["display_class"] = result["class"]
                else:
                    result["display_class"] = result["class"]
                
            except Exception as e:
                logger.error(f"분류 중 오류: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return image, f"분류 중 오류: {str(e)}"
            
            # 결과 이미지에 분류 결과 표시
            try:
                draw = ImageDraw.Draw(pil_image)
                
                # 글꼴 설정 (기본 글꼴 사용)
                try:
                    # Ubuntu 폰트 경로
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, 24)
                    else:
                        font = ImageFont.load_default()
                except Exception:
                    font = ImageFont.load_default()
                
                # 이미지 상단에 분류 결과 표시
                predicted_class = result.get("display_class", result["class"]) if result["class"] is not None else "Unknown"
                confidence = result["score"] if result["class"] is not None else 0.0
                
                text = f"Class: {predicted_class}"
                if predicted_class != "Unknown":
                    text += f" ({confidence:.2f})"
                
                # 텍스트 배경 그리기
                try:
                    # draw.textsize가 있는지 확인 (PIL 버전에 따라 다름)
                    if hasattr(draw, 'textsize'):
                        text_width, text_height = draw.textsize(text, font=font)
                    else:
                        # PIL 9.0.0 이상
                        try:
                            text_width, text_height = font.getbbox(text)[2:]
                        except:
                            text_width, text_height = 200, 30
                except:
                    text_width, text_height = 200, 30
                
                # 텍스트 그리기
                try:
                    draw.rectangle([(10, 10), (text_width + 20, text_height + 20)], fill=(0, 0, 0, 180))
                    draw.text((15, 15), text, fill=(255, 255, 255), font=font)
                except Exception as e:
                    logger.error(f"텍스트 그리기 오류: {e}")
                    # 오류 발생해도 계속 진행
                
                # 결과 텍스트 생성
                result_text = f"분류 결과: {predicted_class}\n"
                if predicted_class != "Unknown":
                    result_text += f"신뢰도: {confidence:.2f}\n\n"
                
                result_text += "모든 클래스 점수:\n"
                for cls, score in result["all_scores"].items():
                    # class_N 형식인 경우 디스플레이 이름 변환
                    if has_class_n_format and class_pattern.match(cls):
                        display_name = f"Class {cls.split('_')[1]}"
                        result_text += f"- {display_name}: {score:.2f}\n"
                    else:
                        result_text += f"- {cls}: {score:.2f}\n"
                
                # 이미지 저장 및 반환
                result_image_path = tempfile.mktemp(suffix='.jpg')
                pil_image.save(result_image_path)
                
                return result_image_path, result_text
            
            except Exception as e:
                logger.error(f"결과 이미지 생성 중 오류: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 오류 발생시 원본 이미지와 텍스트만 반환
                result_text = f"분류 결과: {result['class'] if result['class'] else 'Unknown'}\n"
                result_text += f"이미지 처리 중 오류가 발생했습니다: {str(e)}"
                return image, result_text
        
        except Exception as e:
            logger.error(f"이미지 분류 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return image, f"오류 발생: {str(e)}"
    
    def setup_interface(self):
        """Gradio 인터페이스 설정"""
        gr = self.gr
        with gr.Blocks(title="Few-Shot Learning Platform") as self.interface:
            gr.Markdown("# Few-Shot Learning Platform\n선택한 카테고리와 shot 수에 따라 Few-Shot 분류를 수행하고 Support Set 이미지를 확인합니다.")
            
            with gr.Tabs():
                # Support Set 뷰어 탭
                with gr.TabItem("Support Set 뷰어"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # 컨트롤 패널
                            category_viewer = gr.Dropdown(
                                choices=self.categories, 
                                label="카테고리 선택",
                                interactive=True
                            )
                            
                            shot_viewer = gr.Dropdown(
                                choices=[str(s) for s in self.standard_shots],
                                value="5",
                                label="Shot 수 선택",
                                interactive=True
                            )
                            
                            use_structured_viewer = gr.Checkbox(
                                label="구조화된 Support Set 사용",
                                value=True,
                                interactive=True
                            )
                            
                            view_btn = gr.Button("이미지 보기", variant="primary")
                            
                            # 통계 정보
                            stats_viewer = gr.Textbox(label="통계 정보", lines=10)
                        
                        with gr.Column(scale=7):
                            # 이미지 갤러리
                            gallery_viewer = gr.Gallery(
                                label="Support Set 이미지",
                                columns=4,
                                height=600,
                                object_fit="contain",
                                show_label=True,
                                allow_preview=True
                            )
                
                # 실시간 분류 탭
                with gr.TabItem("실시간 분류"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # 분류 설정
                            category_classifier = gr.Dropdown(
                                choices=self.categories, 
                                label="카테고리 선택",
                                interactive=True
                            )
                            
                            shot_classifier = gr.Dropdown(
                                choices=[str(s) for s in self.standard_shots],
                                value="5",
                                label="Shot 수 선택",
                                interactive=True
                            )
                            
                            threshold_classifier = gr.Dropdown(
                                choices=[str(t) for t in self.standard_thresholds],
                                value="0.75",
                                label="임계값 선택",
                                interactive=True
                            )
                            
                            model_classifier = gr.Dropdown(
                                choices=["dino", "clip", "resnet"],
                                value="dino",
                                label="모델 선택",
                                interactive=True
                            )
                            
                            use_structured_classifier = gr.Checkbox(
                                label="구조화된 Support Set 사용",
                                value=True,
                                interactive=True
                            )
                            
                            init_btn = gr.Button("분류기 초기화", variant="primary")
                            
                            # 초기화 상태
                            init_status = gr.Textbox(label="초기화 상태", lines=2)
                            
                            # 이미지 업로드
                            image_input = gr.Image(
                                label="분류할 이미지 업로드",
                                type="pil"
                            )
                            
                            classify_btn = gr.Button("이미지 분류", variant="primary")
                            
                            # 분류 결과
                            classification_result = gr.Textbox(label="분류 결과", lines=10)
                        
                        with gr.Column(scale=4):
                            # Support Set 갤러리
                            gallery_classifier = gr.Gallery(
                                label="현재 사용 중인 Support Set 이미지",
                                columns=3,
                                height=400,
                                object_fit="contain",
                                show_label=True
                            )
                            
                            # 분류 결과 이미지
                            result_image = gr.Image(
                                label="분류 결과 이미지",
                                type="filepath",
                                height=400
                            )
                
                # 실험 결과 탭
                with gr.TabItem("실험 결과"):
                    gr.Markdown("### 실험 결과 뷰어\n이 탭은 준비 중입니다. FewShotExperiment 클래스의 결과를 보여줄 예정입니다.")
            
            # 이벤트 연결
            # Support Set 뷰어 탭 이벤트
            view_btn.click(
                fn=self.update_gallery,
                inputs=[category_viewer, shot_viewer, use_structured_viewer],
                outputs=[gallery_viewer, stats_viewer]
            )
            
            # 실시간 분류 탭 이벤트
            init_btn.click(
                fn=self.initialize_classifier,
                inputs=[
                    category_classifier, 
                    shot_classifier, 
                    threshold_classifier, 
                    model_classifier,
                    use_structured_classifier
                ],
                outputs=[init_status]
            )
            
            # 분류기 초기화 후 Support Set 갤러리 업데이트
            init_btn.click(
                fn=self.update_gallery,
                inputs=[category_classifier, shot_classifier, use_structured_classifier],
                outputs=[gallery_classifier, gr.Textbox(visible=False)]
            )
            
            # 이미지 분류 이벤트
            classify_btn.click(
                fn=self.classify_image,
                inputs=[image_input],
                outputs=[result_image, classification_result]
            )
            
            # 카테고리 변경 시 자동 갤러리 업데이트
            category_viewer.change(
                fn=lambda c, s, u: self.update_gallery(c, s, u) if c else ([], ""),
                inputs=[category_viewer, shot_viewer, use_structured_viewer],
                outputs=[gallery_viewer, stats_viewer]
            )
    
    def launch(self, share=False, server_port=7860):
        """인터페이스 실행"""
        # 환경 변수로 포트 설정 - Gradio 내부에서도 참조
        os.environ["GRADIO_SERVER_PORT"] = str(server_port)
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        
        logger.info(f"Gradio 웹 인터페이스를 포트 {server_port}에서 시작합니다...")
        
        try:
            # 이미 실행 중인 서버 확인
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            try:
                # 포트가 이미 사용 중인지 확인
                s.bind(('localhost', server_port))
                s.close()
            except socket.error as e:
                logger.error(f"포트 {server_port}가 이미 사용 중입니다: {e}")
                # 다른 포트 시도 (최대 10번)
                for offset in range(1, 11):
                    new_port = server_port + offset
                    logger.info(f"대체 포트 {new_port}를 시도합니다...")
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.bind(('localhost', new_port))
                        s.close()
                        server_port = new_port
                        os.environ["GRADIO_SERVER_PORT"] = str(server_port)
                        logger.info(f"포트 {server_port}를 사용합니다.")
                        break
                    except socket.error:
                        continue
            
            # Gradio 서버 시작 
            self.interface.launch(
                share=share,
                server_port=server_port,
                prevent_thread_lock=False,
                show_error=True,
                quiet=False
            )
            
        except Exception as e:
            logger.error(f"Gradio 인터페이스 실행 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="Few-Shot Learning Experiment & Evaluation Platform"
    )
    
    # 실행 모드 선택
    mode_group = parser.add_argument_group("실행 모드")
    mode_group.add_argument("--webapp", action="store_true", 
                       help="웹 애플리케이션 모드로 실행")
    mode_group.add_argument("--cli", action="store_true",
                       help="CLI 모드로 실행")
    mode_group.add_argument("--mywebapp", action="store_true",
                       help="Support Set 뷰어를 포함한 웹앱 모드로 실행")
    
    # CLI 모드 옵션
    cli_group = parser.add_argument_group("CLI 모드 옵션")
    cli_group.add_argument("--category", type=str,
                       help="실험 대상 카테고리 이름")
    cli_group.add_argument("--model", type=str, default="resnet",
                       choices=["resnet", "clip", "dino"],
                       help="특징 추출에 사용할 모델")
    cli_group.add_argument("--shots", type=str, default="1,5,10,30",
                       help="N-shots 값 (쉼표로 구분)")
    cli_group.add_argument("--thresholds", type=str, 
                       default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
                       help="임계값 목록 (쉼표로 구분)")
    cli_group.add_argument("--input-dir", type=str,
                       help="분류할 이미지가 있는 디렉토리 (기본값: 카테고리의 preprocessed 디렉토리)")
    cli_group.add_argument("--skip-experiments", action="store_true",
                       help="실험 실행을 건너뛰고 기존 결과만 평가")
    
    # 웹 앱 모드 옵션
    webapp_group = parser.add_argument_group("웹 앱 모드 옵션")
    webapp_group.add_argument("--port", type=int, default=7860,
                          help="웹 앱 실행 포트")
    webapp_group.add_argument("--share", action="store_true",
                          help="Gradio 공유 링크 생성")
    
    args = parser.parse_args()
    
    # CLI 모드에서 필수 인수 확인
    if args.cli and not args.category:
        parser.error("--cli 모드에서는 --category가 필요합니다.")
    
    return args


def run_webapp_mode(port=7860, share=False):
    """웹 애플리케이션 모드 실행"""
    try:
        # 필요한 패키지 확인
        try:
            import gradio
            logger.debug(f"Gradio 버전: {gradio.__version__}")
        except ImportError as e:
            logger.error(f"Gradio 패키지를 로드할 수 없습니다: {e}")
            logger.error("gradio 패키지가 설치되어 있는지 확인하세요: pip install gradio")
            sys.exit(1)

        # Gradio 서버 포트 이슈 확인
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("localhost", port))
            s.close()
            logger.debug(f"포트 {port}가 사용 가능합니다.")
        except socket.error as e:
            logger.warning(f"포트 {port}가 이미 사용 중입니다: {e}")
            # 다른 포트 시도
            alt_port = port + 5
            logger.warning(f"대체 포트 {alt_port}를 시도합니다.")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", alt_port))
                s.close()
                logger.debug(f"대체 포트 {alt_port}가 사용 가능합니다.")
                port = alt_port
            except socket.error as e2:
                logger.warning(f"대체 포트 {alt_port}도 사용 중입니다: {e2}")
                logger.warning(f"다른 포트를 사용하려면 --port 옵션을 사용하세요.")
        
        # 환경 변수로 포트 설정
        os.environ["GRADIO_SERVER_PORT"] = str(port)
        
        # FewShotWebApp 모듈 로드
        try:
            from scripts.few_shot_webapp import FewShotWebApp
            logger.debug("FewShotWebApp 클래스 로드 성공")
        except ImportError as e:
            logger.error(f"FewShotWebApp 모듈을 로드할 수 없습니다: {e}")
            logger.error("스크립트 경로가 올바른지 확인하세요.")
            sys.exit(1)
        
        # 웹앱 초기화 시도
        try:
            logger.info("웹 앱 초기화 중...")
            webapp = FewShotWebApp()
            logger.debug("웹 앱 초기화 성공")
            logger.debug(f"사용 가능한 카테고리: {webapp.available_categories}")
        except Exception as e:
            logger.error(f"웹 앱 초기화 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # 웹앱 실행 시도
        try:
            logger.info(f"웹 앱 실행 중... (포트: {port}, 공유: {share})")
            webapp.launch(server_port=port, share=share, server_name="0.0.0.0")
            logger.info("웹 앱 실행 완료")
        except Exception as e:
            logger.error(f"웹 앱 실행 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            
            # 시스템 매개변수를 사용하여 대체 스크립트 실행
            logger.info("간소화된 웹앱으로 재시도합니다...")
            try:
                logger.info("simple_webapp.py 실행 중...")
                import subprocess
                subprocess.run([sys.executable, "simple_webapp.py"], check=True)
            except Exception as sub_e:
                logger.error(f"대체 웹앱 실행 실패: {sub_e}")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"웹 앱 모드 실행 중 예상치 못한 오류 발생: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_cli_mode(args):
    """CLI 모드 실행"""
    try:
        from scripts.classifier_cosine_experiment import FewShotExperiment
        
        logger.info("CLI 모드로 실행합니다...")
        
        # N-shots 및 threshold 변환
        n_shots = [int(n.strip()) for n in args.shots.split(",")]
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        
        # 실험 객체 초기화
        experiment = FewShotExperiment(
            category_name=args.category,
            model_name=args.model
        )
        
        # 사용자 설정으로 실험 객체 업데이트
        experiment.n_shots = n_shots
        experiment.thresholds = thresholds
        
        logger.info(f"카테고리: {args.category}, 모델: {args.model}")
        logger.info(f"N-shots: {n_shots}")
        logger.info(f"Thresholds: {thresholds}")
        
        # 실험 실행
        if not args.skip_experiments:
            logger.info("실험을 실행합니다...")
            experiment.run_all_experiments(args.input_dir)
            logger.info(f"총 {len(n_shots) * len(thresholds)}개 실험 조합 실행됨")
        else:
            logger.info("실험 실행을 건너뜁니다.")
        
        logger.info(f"결과 경로: {experiment.results_dir}")
        logger.info("CLI 모드 실행 완료")
        
    except ImportError as e:
        logger.error(f"필요한 모듈을 로드할 수 없습니다: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"입력 값이 잘못되었습니다: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_mywebapp_mode(port=7860, share=False):
    """Support Set 뷰어가 포함된 웹앱 모드 실행"""
    try:
        # 필요한 패키지 확인
        try:
            import gradio as gr
            logger.debug(f"Gradio 버전: {gr.__version__}")
        except ImportError as e:
            logger.error(f"Gradio 패키지를 로드할 수 없습니다: {e}")
            logger.error("gradio 패키지가 설치되어 있는지 확인하세요: pip install gradio")
            sys.exit(1)
            
        logger.info("Support Set 뷰어를 포함한 웹앱 모드를 시작합니다...")
        
        # 포트가 사용 가능한지 확인
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('localhost', port))
            s.close()
        except socket.error as e:
            logger.error(f"포트 {port}를 사용할 수 없습니다: {e}")
            new_port = port + 1
            logger.info(f"대체 포트 {new_port}를 사용합니다.")
            port = new_port
        
        platform = FewShotPlatform()
        platform.launch(server_port=port, share=share)
        return 0
        
    except Exception as e:
        logger.error(f"Support Set 뷰어 모드 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """메인 함수"""
    args = parse_args()
    
    if args.webapp:
        run_webapp_mode(port=args.port, share=args.share)
    elif args.cli:
        run_cli_mode(args)
    elif args.mywebapp:
        run_mywebapp_mode(port=args.port, share=args.share)
    else:
        # 실행 모드가 지정되지 않은 경우 도움말 표시
        print("실행 모드를 지정해야 합니다. --webapp, --cli 또는 --mywebapp")
        print("도움말: python run_few_shot_platform.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main() 