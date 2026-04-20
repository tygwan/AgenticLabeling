#!/usr/bin/env python3
"""
폴더 기반 Ground Truth Labeling 시스템

이 모듈은 폴더 탐색기 스타일의 인터페이스를 제공하여 이미지를 쉽게 분류하고
Ground Truth를 설정할 수 있게 해줍니다.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
import gradio as gr
import cv2
from PIL import Image
import numpy as np
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("folder_labeler.log")
    ]
)
logger = logging.getLogger(__name__)

# 프로젝트 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 기본 클래스 목록 - 카테고리 로드 시 업데이트됨
DEFAULT_CLASSES = ["unknown", "fence_person", "sidewalk", "car", "traffic cone"]

class FolderBasedLabeler:
    def __init__(self):
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.current_category = None
        self.current_experiment = None
        self.ground_truth_dir = None
        self.class_dirs = {}
        self.images_data = {}
        self.class_list = DEFAULT_CLASSES.copy()
        self.current_class = None
        self.current_view = "source"  # "source" 또는 "ground_truth"
        
    def get_categories(self):
        """사용 가능한 카테고리 목록 가져오기"""
        categories = []
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    categories.append(item.name)
        return categories
    
    def get_experiments(self, category):
        """카테고리에 대한 사용 가능한 실험 가져오기"""
        self.current_category = category
        experiments = []
        results_dir = self.data_dir / category / "7.results"
        
        if results_dir.exists():
            for item in results_dir.iterdir():
                if item.is_dir():
                    # results.json 파일이 있는지 확인
                    if (item / "results.json").exists():
                        experiments.append(item.name)
        
        return experiments
    
    def prepare_ground_truth_folders(self, category):
        """Ground Truth 폴더 구조 준비"""
        # Ground Truth 기본 디렉토리
        self.ground_truth_dir = self.data_dir / category / "8.ground_truth"
        os.makedirs(self.ground_truth_dir, exist_ok=True)
        
        # 각 클래스별 폴더 생성
        for class_name in self.class_list:
            class_dir = self.ground_truth_dir / class_name
            os.makedirs(class_dir, exist_ok=True)
            self.class_dirs[class_name] = class_dir
        
        # Ground Truth 매핑 파일 초기화 또는 로드
        self.gt_mapping_file = self.ground_truth_dir / "mapping.json"
        if self.gt_mapping_file.exists():
            with open(self.gt_mapping_file, 'r') as f:
                self.images_data = json.load(f)
        else:
            self.images_data = {}
        
        return f"{len(self.class_dirs)} 클래스 폴더가 준비되었습니다."
    
    def load_experiment(self, category, experiment):
        """실험 데이터 로드"""
        self.current_category = category
        self.current_experiment = experiment
        
        # 클래스 목록 초기화
        self.class_list = DEFAULT_CLASSES.copy()
        
        # 결과 파일 로드
        results_path = self.data_dir / category / "7.results" / experiment / "results.json"
        if not results_path.exists():
            return f"오류: {results_path}에서 결과 파일을 찾을 수 없습니다."
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # 고유 클래스 이름 추출
            classes = set(["unknown"])
            for img_data in results.values():
                if "predicted_class" in img_data:
                    classes.add(img_data["predicted_class"])
            
            self.class_list = sorted(list(classes))
            
            # Ground Truth 폴더 준비
            self.prepare_ground_truth_folders(category)
            
            # 이미지 데이터 초기화
            for img_name, img_data in results.items():
                if img_name not in self.images_data:
                    predicted = img_data.get("predicted_class", "unknown")
                    confidence = img_data.get("confidence", 0.0)
                    self.images_data[img_name] = {
                        "original_class": predicted,
                        "ground_truth_class": None,
                        "confidence": confidence,
                        "processed": False
                    }
            
            # 이미 처리된 이미지 확인
            self._update_processed_status()
            
            # 매핑 파일 저장
            self._save_mapping()
            
            return f"{len(results)} 이미지가 로드되었습니다. {len(self.class_list)} 클래스가 발견되었습니다."
        
        except Exception as e:
            logger.error(f"실험 로드 중 오류: {e}")
            return f"실험 로드 중 오류: {str(e)}"
    
    def _update_processed_status(self):
        """처리된 이미지 상태 업데이트"""
        # 각 클래스 폴더에서 이미지 확인
        for class_name, class_dir in self.class_dirs.items():
            if class_dir.exists():
                for img_file in class_dir.glob("*.*"):
                    img_name = img_file.name
                    if img_name in self.images_data:
                        self.images_data[img_name]["ground_truth_class"] = class_name
                        self.images_data[img_name]["processed"] = True
    
    def _save_mapping(self):
        """Ground Truth 매핑 저장"""
        try:
            with open(self.gt_mapping_file, 'w') as f:
                json.dump(self.images_data, f, indent=2)
            logger.info(f"매핑 파일이 저장됨: {self.gt_mapping_file}")
        except Exception as e:
            logger.error(f"매핑 파일 저장 중 오류: {e}")
    
    def export_ground_truth(self):
        """Ground Truth를 표준 형식으로 내보내기"""
        if not self.ground_truth_dir or not self.current_category:
            return "Ground Truth 폴더가 설정되지 않았습니다."
        
        gt_data = {}
        for img_name, img_info in self.images_data.items():
            if img_info.get("processed") and img_info.get("ground_truth_class"):
                gt_data[img_name] = img_info["ground_truth_class"]
        
        # 표준 Ground Truth 파일 저장
        standard_gt_path = self.data_dir / self.current_category / "7.results" / "ground_truth.json"
        try:
            with open(standard_gt_path, 'w') as f:
                json.dump(gt_data, f, indent=2)
            
            return f"{len(gt_data)} 이미지에 대한 Ground Truth가 {standard_gt_path}에 저장되었습니다."
        except Exception as e:
            logger.error(f"Ground Truth 내보내기 중 오류: {e}")
            return f"Ground Truth 내보내기 중 오류: {str(e)}"
    
    def get_image_path(self, img_name):
        """이미지의 전체 경로 가져오기"""
        return self.data_dir / self.current_category / "1.images" / img_name
    
    def get_images_by_class(self, class_name=None, view="source"):
        """클래스별 이미지 가져오기"""
        self.current_class = class_name
        self.current_view = view
        
        if not self.current_category or not self.images_data:
            return []
        
        image_list = []
        
        if view == "source":
            # 원본 이미지에서 해당 클래스 이미지 가져오기
            for img_name, img_info in self.images_data.items():
                if (class_name is None or 
                    (class_name == "unprocessed" and not img_info.get("processed")) or
                    (class_name != "unprocessed" and img_info.get("original_class") == class_name)):
                    
                    img_path = self.get_image_path(img_name)
                    if img_path.exists():
                        is_processed = img_info.get("processed", False)
                        gt_class = img_info.get("ground_truth_class", "")
                        confidence = img_info.get("confidence", 0.0)
                        
                        # 레이블 텍스트 생성
                        status = "✓" if is_processed else ""
                        label = f"{img_name}\n{status} {gt_class if is_processed else ''}\n{confidence:.2f}"
                        
                        image_list.append((str(img_path), label))
        
        elif view == "ground_truth":
            # Ground Truth 폴더에서 이미지 가져오기
            if class_name and class_name in self.class_dirs:
                class_dir = self.class_dirs[class_name]
                for img_file in class_dir.glob("*.*"):
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img_name = img_file.name
                        confidence = self.images_data.get(img_name, {}).get("confidence", 0.0)
                        label = f"{img_name}\n{confidence:.2f}"
                        image_list.append((str(img_file), label))
        
        return image_list
    
    def move_image_to_class(self, img_path, target_class):
        """이미지를 대상 클래스 폴더로 이동"""
        if not img_path or not target_class or target_class not in self.class_dirs:
            return "잘못된 이미지 경로 또는 대상 클래스입니다."
        
        try:
            img_path = Path(img_path)
            img_name = img_path.name
            
            # 다른 클래스 폴더에서 이미지가 있는지 확인하고 제거
            for class_name, class_dir in self.class_dirs.items():
                existing_path = class_dir / img_name
                if existing_path.exists():
                    os.remove(existing_path)
            
            # 원본 이미지를 대상 클래스 폴더로 복사
            target_path = self.class_dirs[target_class] / img_name
            shutil.copy2(self.get_image_path(img_name), target_path)
            
            # 이미지 데이터 업데이트
            if img_name in self.images_data:
                self.images_data[img_name]["ground_truth_class"] = target_class
                self.images_data[img_name]["processed"] = True
            
            # 매핑 파일 저장
            self._save_mapping()
            
            return f"{img_name}이(가) {target_class} 클래스로 지정되었습니다."
        
        except Exception as e:
            logger.error(f"이미지 이동 중 오류: {e}")
            return f"이미지 이동 중 오류: {str(e)}"
    
    def batch_move_images(self, img_paths, target_class):
        """여러 이미지를 대상 클래스 폴더로 이동"""
        if not img_paths or not target_class or target_class not in self.class_dirs:
            return "잘못된 이미지 경로 또는 대상 클래스입니다."
        
        success_count = 0
        for img_path in img_paths:
            try:
                img_path = Path(img_path)
                img_name = img_path.name
                
                # 다른 클래스 폴더에서 이미지가 있는지 확인하고 제거
                for class_name, class_dir in self.class_dirs.items():
                    existing_path = class_dir / img_name
                    if existing_path.exists():
                        os.remove(existing_path)
                
                # 원본 이미지를 대상 클래스 폴더로 복사
                target_path = self.class_dirs[target_class] / img_name
                shutil.copy2(self.get_image_path(img_name), target_path)
                
                # 이미지 데이터 업데이트
                if img_name in self.images_data:
                    self.images_data[img_name]["ground_truth_class"] = target_class
                    self.images_data[img_name]["processed"] = True
                
                success_count += 1
            except Exception as e:
                logger.error(f"{img_name} 이동 중 오류: {e}")
        
        # 매핑 파일 저장
        self._save_mapping()
        
        return f"{success_count}개 이미지가 {target_class} 클래스로 지정되었습니다."
    
    def get_statistics(self):
        """현재 Ground Truth 상태에 대한 통계 가져오기"""
        stats = {
            "total": len(self.images_data),
            "processed": 0,
            "by_class": {class_name: 0 for class_name in self.class_list}
        }
        
        for img_info in self.images_data.values():
            if img_info.get("processed", False):
                stats["processed"] += 1
                gt_class = img_info.get("ground_truth_class")
                if gt_class in stats["by_class"]:
                    stats["by_class"][gt_class] += 1
        
        return stats

def create_ui():
    """Gradio UI 생성"""
    labeler = FolderBasedLabeler()
    
    with gr.Blocks(title="폴더 기반 Ground Truth Labeling 도구") as app:
        gr.Markdown("# 폴더 기반 Ground Truth Labeling 도구")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 카테고리 및 실험 선택
                category_dropdown = gr.Dropdown(
                    choices=labeler.get_categories(),
                    label="카테고리 선택",
                    interactive=True
                )
                experiment_dropdown = gr.Dropdown(
                    choices=[],
                    label="실험 선택",
                    interactive=True
                )
                load_btn = gr.Button("실험 로드")
                load_result = gr.Textbox(label="상태", lines=2)
                
                # 클래스 및 뷰 선택
                gr.Markdown("### 클래스 및 뷰 선택")
                view_radio = gr.Radio(
                    choices=["원본 이미지", "Ground Truth 폴더"],
                    label="뷰 선택",
                    value="원본 이미지"
                )
                class_dropdown = gr.Dropdown(
                    choices=["모든 클래스", "미처리 이미지"] + labeler.class_list,
                    label="클래스 필터",
                    value="모든 클래스",
                    interactive=True
                )
                refresh_btn = gr.Button("이미지 새로고침")
                
                # 배치 작업
                gr.Markdown("### 배치 작업")
                target_class = gr.Dropdown(
                    choices=labeler.class_list,
                    label="선택한 이미지를 다음 클래스로 이동",
                    interactive=True
                )
                apply_batch_btn = gr.Button("선택한 이미지에 적용")
                
                # 통계 및 내보내기
                gr.Markdown("### 통계 및 내보내기")
                stats_text = gr.Textbox(label="통계", lines=6)
                refresh_stats_btn = gr.Button("통계 새로고침")
                export_btn = gr.Button("Ground Truth 내보내기")
                export_result = gr.Textbox(label="내보내기 결과", lines=2)
            
            with gr.Column(scale=3):
                # 메인 이미지 갤러리
                gallery = gr.Gallery(
                    label="이미지",
                    columns=4,
                    rows=5,
                    show_label=True,
                    object_fit="contain",
                    height="600px",
                    allow_preview=True,
                    selected_index=None
                ).style(grid=4, height="600px")
                
                # 선택한 이미지 및 작업
                with gr.Row():
                    with gr.Column(scale=2):
                        selected_image = gr.Image(label="선택한 이미지", type="filepath")
                    with gr.Column(scale=1):
                        image_info = gr.Textbox(label="이미지 정보", lines=3)
                        single_class = gr.Dropdown(
                            choices=labeler.class_list,
                            label="이 이미지를 다음 클래스로 이동",
                            interactive=True
                        )
                        apply_single_btn = gr.Button("적용")
        
        # 이벤트 핸들러
        def update_experiments(category):
            experiments = labeler.get_experiments(category)
            return gr.Dropdown(choices=experiments)
        
        def load_experiment_handler(category, experiment):
            result = labeler.load_experiment(category, experiment)
            
            # 클래스 드롭다운 업데이트
            class_choices = ["모든 클래스", "미처리 이미지"] + labeler.class_list
            
            # 이미지 갤러리 업데이트
            images = labeler.get_images_by_class(None, "source")
            
            return [
                result,
                gr.Dropdown(choices=labeler.class_list),
                gr.Dropdown(choices=labeler.class_list),
                gr.Dropdown(choices=class_choices, value="모든 클래스"),
                images
            ]
        
        def view_change_handler(view_selection, class_selection):
            # 뷰 이름 변환
            view = "source" if view_selection == "원본 이미지" else "ground_truth"
            
            # 클래스 이름 변환
            class_name = None
            if class_selection == "모든 클래스":
                class_name = None
            elif class_selection == "미처리 이미지":
                class_name = "unprocessed"
            else:
                class_name = class_selection
            
            # 이미지 가져오기
            images = labeler.get_images_by_class(class_name, view)
            
            return images
        
        def select_image_handler(evt: gr.SelectData):
            if not labeler.images_data:
                return None, "이미지를 선택하지 않았습니다"
            
            try:
                selected_index = evt.index
                gallery_images = labeler.get_images_by_class(
                    labeler.current_class,
                    labeler.current_view
                )
                
                if selected_index < len(gallery_images):
                    img_path, _ = gallery_images[selected_index]
                    img_name = os.path.basename(img_path)
                    
                    # 이미지 정보 가져오기
                    img_info = labeler.images_data.get(img_name, {})
                    original_class = img_info.get("original_class", "알 수 없음")
                    gt_class = img_info.get("ground_truth_class", "지정되지 않음")
                    processed = img_info.get("processed", False)
                    confidence = img_info.get("confidence", 0.0)
                    
                    info_text = (
                        f"파일명: {img_name}\n"
                        f"원본 클래스: {original_class} (신뢰도: {confidence:.4f})\n"
                        f"Ground Truth: {gt_class} (처리됨: {'예' if processed else '아니오'})"
                    )
                    
                    return img_path, info_text
            
            except Exception as e:
                logger.error(f"이미지 선택 중 오류: {e}")
            
            return None, "이미지 선택 중 오류가 발생했습니다"
        
        def move_single_image(img_path, target_class):
            if not img_path or not target_class:
                return "이미지 또는 대상 클래스가 선택되지 않았습니다", labeler.get_images_by_class(labeler.current_class, labeler.current_view)
            
            result = labeler.move_image_to_class(img_path, target_class)
            
            # 갤러리 업데이트
            updated_images = labeler.get_images_by_class(labeler.current_class, labeler.current_view)
            
            return result, updated_images
        
        def batch_move_images(evt: gr.SelectData, target_class):
            if not target_class:
                return "대상 클래스가 선택되지 않았습니다", labeler.get_images_by_class(labeler.current_class, labeler.current_view)
            
            selected_indices = evt.index
            if not isinstance(selected_indices, list):
                selected_indices = [selected_indices]
            
            gallery_images = labeler.get_images_by_class(labeler.current_class, labeler.current_view)
            selected_paths = [gallery_images[i][0] for i in selected_indices if i < len(gallery_images)]
            
            if not selected_paths:
                return "선택된 이미지가 없습니다", gallery_images
            
            result = labeler.batch_move_images(selected_paths, target_class)
            
            # 갤러리 업데이트
            updated_images = labeler.get_images_by_class(labeler.current_class, labeler.current_view)
            
            return result, updated_images
        
        def update_statistics():
            stats = labeler.get_statistics()
            processed_percent = (stats["processed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            
            text = (
                f"전체 이미지: {stats['total']}\n"
                f"처리된 이미지: {stats['processed']} ({processed_percent:.1f}%)\n\n"
                f"클래스별 분포:\n"
            )
            
            for cls, count in stats["by_class"].items():
                if count > 0:
                    class_percent = (count / stats["processed"] * 100) if stats["processed"] > 0 else 0
                    text += f"- {cls}: {count} ({class_percent:.1f}%)\n"
            
            return text
        
        def export_ground_truth():
            return labeler.export_ground_truth()
        
        # 이벤트 연결
        category_dropdown.change(
            fn=update_experiments,
            inputs=category_dropdown,
            outputs=experiment_dropdown
        )
        
        load_btn.click(
            fn=load_experiment_handler,
            inputs=[category_dropdown, experiment_dropdown],
            outputs=[
                load_result,
                target_class,
                single_class,
                class_dropdown,
                gallery
            ]
        )
        
        view_radio.change(
            fn=view_change_handler,
            inputs=[view_radio, class_dropdown],
            outputs=gallery
        )
        
        class_dropdown.change(
            fn=view_change_handler,
            inputs=[view_radio, class_dropdown],
            outputs=gallery
        )
        
        refresh_btn.click(
            fn=view_change_handler,
            inputs=[view_radio, class_dropdown],
            outputs=gallery
        )
        
        gallery.select(
            fn=select_image_handler,
            outputs=[selected_image, image_info]
        )
        
        apply_single_btn.click(
            fn=move_single_image,
            inputs=[selected_image, single_class],
            outputs=[load_result, gallery]
        )
        
        apply_batch_btn.click(
            fn=lambda *args: batch_move_images(*args),
            inputs=[gallery, target_class],
            outputs=[load_result, gallery]
        )
        
        refresh_stats_btn.click(
            fn=update_statistics,
            outputs=stats_text
        )
        
        export_btn.click(
            fn=export_ground_truth,
            outputs=export_result
        )
    
    return app

def main():
    app = create_ui()
    app.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main() 