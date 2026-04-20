#!/usr/bin/env python
"""
Autodistill Pipeline Runner (Florence-2 + SAM2)

This script implements an automatic image annotation pipeline using Autodistill,
combining Florence-2 for zero-shot classification and GroundedSAM2 for object detection.
"""

import os
import time
import datetime
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm  # 프로그레스 바를 위한 tqdm 추가

# 로깅 레벨 설정
VERBOSE = False  # 상세 로그 출력 여부
DEBUG = False    # 디버그 모드

# 로그 출력 함수
def log(message, level="INFO"):
    """레벨에 따른 로그 출력"""
    if level == "DEBUG" and not DEBUG:
        return
    if level == "VERBOSE" and not VERBOSE:
        return
    print(f"[{level}] {message}")

# Autodistill-related imports
log("필요한 라이브러리 로드 중...", "VERBOSE")
from autodistill_florence_2 import Florence2
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
from autodistill_yolov8 import YOLOv8

# Import project utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_utils import (
    get_images_path,
    get_dataset_path,
    create_data_yaml,
    get_subfolder_path,
    process_classification_results
)

# 이미지 전처리 유틸리티 가져오기
try:
    from scripts.preprocess_utils import (
        batch_preprocess_images,
        preprocess_detections
    )
except ImportError:
    log("Warning: preprocess_utils 모듈을 가져올 수 없습니다.", "WARNING")
    batch_preprocess_images = None
    preprocess_detections = None

# Mask utilities
from scripts.mask_utils import process_mask_file, convert_and_save_mask, save_coords_without_mask


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Autodistill Pipeline Runner (Florence-2 + SAM2)")
    
    parser.add_argument("--category", type=str, default="test_category", 
                        help="카테고리 이름 (기본값: test_category)")
    parser.add_argument("--class-mapping", help="Path to the class mapping JSON file")
    parser.add_argument("--max-images", type=int, default=None, 
                        help="Maximum number of images to process (default: all)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot and save detection results")
    parser.add_argument("--train", action="store_true",
                        help="Train YOLO model after annotation")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for YOLO training (default: 100)")
    parser.add_argument("--device", default="0",
                        help="Device to use for training (default: '0')")
    parser.add_argument("--yolo-model", default="yolov8x-seg.pt",
                        help="YOLO model to use (default: yolov8x-seg.pt)")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="Confidence threshold for classification (default: 0.7)")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip Florence-2 classification and go directly to SAM2 detection")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip SAM2 detection and only perform Florence-2 classification")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess images after detection (crop objects, apply masks)")
    parser.add_argument("--target-size", type=int, nargs=2, default=[640, 640],
                        help="Target size for preprocessed images (width, height)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip cropping objects during preprocessing")
    parser.add_argument("--no-mask", action="store_true",
                        help="Skip applying masks during preprocessing")
    parser.add_argument("--save-npy", action="store_true",
                        help="Save mask data as NPY files")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    global DEBUG, VERBOSE
    DEBUG = args.debug
    VERBOSE = args.verbose
    return args


def load_class_mapping(file_path: Optional[str] = None) -> Dict[str, str]:
    """Load class mapping from file or use default mapping."""
    # 기본 클래스 매핑 (프롬프트 -> 클래스 이름)
    default_mapping = {
        "What blue, fabric-like barrier, explicitly designed to shield or guide pedestrians near construction zones, is visible in this image?": "fence_person",
        "What type of pedestrian pathway, found alongside roads or within construction zones, composed of materials like nonwoven fabric, sand, bricks, or asphalt, is visible in this image?": "sidewalk",
        "What motorized vehicle, designed for passenger or cargo transport, commonly seen in urban or road environments, is visible in this image?": "car",
        "What small, brightly colored cone-shaped object, specifically designed to redirect traffic or highlight construction hazards, is visible in this image?": "traffic cone"
    }
    
    if not file_path or not os.path.exists(file_path):
        log(f"기본 클래스 매핑 사용: {list(default_mapping.values())}")
        return default_mapping
    
    try:
        with open(file_path, 'r') as f:
            mapping = json.load(f)
            log(f"클래스 매핑 로드 완료: {file_path}: {list(mapping.values())}")
            return mapping
    except Exception as e:
        log(f"클래스 매핑 로드 중 오류: {e}", "ERROR")
        log(f"기본 클래스 매핑 사용: {list(default_mapping.values())}", "WARNING")
        return default_mapping


def create_grounded_sam_ontology(class_mapping: Dict[str, str]) -> CaptionOntology:
    """Create a CaptionOntology for GroundedSAM2 based on class mapping."""
    # CaptionOntology는 이미 prompt->class 형식 매핑을 받으므로
    # 바로 매핑을 전달할 수 있음
    return CaptionOntology(class_mapping)


def get_image_paths(directory: str, extensions: Tuple[str, ...] = ('jpg', 'jpeg', 'png')) -> List[str]:
    """Get paths to all images in the directory with the specified extensions."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def run_florence2_classification(
    image_paths: List[str], 
    class_names: List[str],
    results_dir: str,
    confidence_threshold: float = 0.7
) -> Dict[str, Dict[str, float]]:
    """
    Run Florence-2 classification on the images.
    
    Args:
        image_paths: List of image paths to classify
        class_names: List of class names to identify
        results_dir: Directory to save results
        confidence_threshold: Confidence threshold for classification
        
    Returns:
        Dictionary mapping image paths to class confidence scores
    """
    log(f"\n Florence-2 분류 시작: {len(class_names)} 클래스 대상")
    
    # Initialize Florence-2 model with a simple ontology
    ontology = CaptionOntology({f"What {class_name}?" : class_name for class_name in class_names})
    classifier = Florence2(ontology=ontology)
    
    # Prepare the output
    classification_results = {}
    
    # Process each image with progress bar
    for i, image_path in enumerate(tqdm(image_paths, desc="이미지 분류 중")):
        log(f"이미지 분류 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}", "VERBOSE")
        
        try:
            # Run classification
            results = classifier.predict(
                image_path=image_path,
                candidate_labels=class_names
            )
            
            # Store results
            image_results = {}
            for class_name, confidence in results.items():
                image_results[class_name] = float(confidence)
                if confidence >= confidence_threshold:
                    log(f"  - 탐지됨: {class_name} (신뢰도: {confidence:.4f})", "DEBUG")
            
            classification_results[image_path] = image_results
            
        except Exception as e:
            log(f"이미지 분류 중 오류 {image_path}: {e}", "ERROR")
    
    # Save the classification results
    results_file = os.path.join(results_dir, "florence2_classification.json")
    with open(results_file, 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    log(f"분류 결과 저장 완료: {results_file}")
    
    return classification_results


def create_support_set(
    classification_results: Dict[str, Dict[str, float]],
    class_names: List[str],
    category: str,
    confidence_threshold: float = 0.7
) -> Dict[str, List[str]]:
    """
    Create a support set of images for each class based on classification results.
    
    Args:
        classification_results: Dictionary of classification results
        class_names: List of class names
        category: Category name
        confidence_threshold: Confidence threshold for inclusion in support set
        
    Returns:
        Dictionary mapping class names to lists of support images
    """
    log("\n분류 결과를 기반으로 지원 세트 생성 중")
    
    # Process classification results
    class_images = process_classification_results(
        category,
        Path(os.path.join(get_subfolder_path(category, "6.results"), "florence2_classification.json")),
        confidence_threshold
    )
    
    # Get the support set directory
    support_dir = get_subfolder_path(category, "2.support-set")
    if not os.path.exists(support_dir):
        os.makedirs(support_dir)
    
    # Create a subdirectory for each class and copy top images
    for class_name, images in class_images.items():
        class_dir = os.path.join(support_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Sort images by confidence (assuming the order in class_images matches confidence)
        # In a real implementation, you would sort by confidence score
        top_images = images[:5]  # Use top 5 images for each class
        
        log(f"클래스 {class_name}: {len(images)} 이미지 찾음, 상위 {len(top_images)}개 사용")
        
        # Create symbolic links to the top images
        for i, img_path in enumerate(top_images):
            try:
                import shutil
                dest_path = os.path.join(class_dir, f"{i+1}_{os.path.basename(img_path)}")
                shutil.copy2(img_path, dest_path)
                log(f"  - 추가됨: {os.path.basename(img_path)} -> {class_name} 지원 세트", "DEBUG")
            except Exception as e:
                log(f"  - 지원 세트 추가 중 오류: {img_path}: {e}", "ERROR")
    
    return class_images


def run_autodistill_pipeline(args, progress_callback=None):
    """
    Run the complete Autodistill pipeline.
    
    Args:
        args: The arguments for the pipeline
        progress_callback: Optional callback function to report progress.
                          Function signature: (step: str, progress: float) -> None
    
    Returns:
        Dictionary with pipeline results
    """
    start_time = time.time()
    
    # Helper function to report progress
    def report_progress(step, progress):
        if progress_callback:
            try:
                progress_callback(step, progress)
            except Exception as e:
                log(f"프로그레스 콜백 오류: {e}", "ERROR")
    
    # Initialize result dictionary
    result = {
        "images_processed": 0,
        "objects_detected": {},
        "output_paths": {}
    }
    
    # Initial progress
    report_progress("파이프라인 초기화", 0.0)
    
    # Load class mapping
    class_mapping = load_class_mapping(args.class_mapping)
    # 클래스 이름만 추출 (프롬프트 매핑의 값들)
    class_names = list(set(class_mapping.values()))
    
    # Get paths
    images_dir = get_images_path(args.category)
    dataset_dir = get_dataset_path(args.category)
    results_dir = get_subfolder_path(args.category, "6.results")
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create preprocessed directory if needed
    preprocessed_dir = None
    if args.preprocess:
        preprocessed_dir = get_subfolder_path(args.category, "5.preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Get image paths
    image_paths = get_image_paths(str(images_dir))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    if not image_paths:
        log(f"이미지를 찾을 수 없음: {images_dir}", "WARNING")
        report_progress("이미지를 찾을 수 없음", 100.0)
        return result
    
    # Set images processed count
    result["images_processed"] = len(image_paths)
    
    log(f"처리할 이미지 {len(image_paths)}개 발견")
    report_progress("이미지 찾음", 5.0)
    
    # Create YAML file for YOLO training
    yaml_path = create_data_yaml(args.category, class_names)
    result["output_paths"]["data_yaml"] = str(yaml_path)
    
    # Step 1: Run Florence-2 classification (unless skipped)
    classification_results = None
    if not args.skip_classification:
        report_progress("Florence-2 분류 실행", 10.0)
        classification_results = run_florence2_classification(
            image_paths,
            class_names,
            str(results_dir),
            args.confidence
        )
        
        # Save classification results path
        result["output_paths"]["classification_results"] = os.path.join(results_dir, "florence2_classification.json")
        
        report_progress("지원 세트 생성", 35.0)
        # Process classification results (create support set)
        support_set = create_support_set(
            classification_results,
            class_names,
            args.category,
            args.confidence
        )
        result["output_paths"]["support_set"] = get_subfolder_path(args.category, "2.support-set")
    else:
        report_progress("분류 건너뛰기", 35.0)
    
    # Step 2: Run GroundedSAM2 detection (unless skipped)
    if not args.skip_detection:
        report_progress("GroundedSAM2 초기화", 40.0)
        log("\nGroundedSAM2 객체 탐지 실행")
        
        # Create ontology
        ontology = create_grounded_sam_ontology(class_mapping)
        log(f"온톨로지 생성 완료: {len(ontology.prompts())} 프롬프트")
        
        # Initialize GroundedSAM2 model
        base_model = GroundedSAM2(ontology=ontology)
        log("GroundedSAM2 모델 초기화 완료")
        
        # Create detections directory under results
        detections_dir = os.path.join(results_dir, "detections")
        os.makedirs(detections_dir, exist_ok=True)
        
        # Create mask directory
        mask_directory = get_subfolder_path(args.category, "4.mask")
        os.makedirs(mask_directory, exist_ok=True)
        
        # Store all detection results
        all_detections = {}
        
        # Initialize object counter
        objects_detected = {class_name: 0 for class_name in class_names}
        
        # Process each image with progress bar
        report_progress("객체 탐지 실행", 45.0)
        for i, image_path in enumerate(tqdm(image_paths, desc="객체 탐지 중")):
            # Update progress based on image processing
            progress = 45.0 + (i / len(image_paths)) * 25.0
            report_progress(f"이미지 {i+1}/{len(image_paths)}에서 객체 탐지 중", progress)
            
            log(f"이미지 처리 중 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}", "VERBOSE")
            
            try:
                # Run detection
                results = base_model.predict(image_path)
                
                # Save detection results
                detection_file = os.path.join(
                    detections_dir, 
                    f"detection_{os.path.basename(image_path)}.json"
                )
                
                # Extract detections information
                detection_data = {}
                if results is not None and hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                    detection_data = {
                        "boxes": results.xyxy.tolist(),
                        "class_ids": results.class_id.tolist(),
                        "classes": [ontology.classes()[cls_id] for cls_id in results.class_id],
                        "confidence": results.confidence.tolist() if hasattr(results, 'confidence') else None,
                    }
                    
                    # Update object counts
                    for cls_name in detection_data["classes"]:
                        objects_detected[cls_name] = objects_detected.get(cls_name, 0) + 1
                    
                    log(f"  - {len(results.xyxy)}개 객체 탐지됨", "DEBUG")
                    all_detections[image_path] = detection_data
                else:
                    log(f"  - 탐지된 객체 없음", "DEBUG")
                    all_detections[image_path] = {"boxes": [], "class_ids": [], "classes": []}
                
                # Save detection results
                with open(detection_file, 'w') as f:
                    json.dump(detection_data, f, indent=2)
                
                # Process mask information if available
                img_basename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Handle masks or boxes
                handle_detection_results(
                    image_path,
                    results,
                    mask_directory,
                    class_names=ontology.classes(),
                    save_npy=args.save_npy
                )
                
                # Visualize results if requested
                if args.plot and results is not None:
                    try:
                        # Matplotlib 백엔드 문제를 방지하기 위해 파일로 직접 저장
                        img = cv2.imread(image_path)
                        if img is None:
                            log(f"  - 시각화를 위한 이미지 로드 실패: {image_path}", "WARNING")
                            continue
                        
                        # OpenCV로 직접 시각화 처리
                        output_path = os.path.join(
                            detections_dir, 
                            f"vis_{os.path.basename(image_path)}"
                        )
                        
                        # 결과 시각화 (간단한 방식으로)
                        vis_image = img.copy()
                        
                        # 상자 시각화
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                            for i, box in enumerate(results.xyxy):
                                x1, y1, x2, y2 = [int(coord) for coord in box]
                                cls_id = results.class_id[i]
                                conf = results.confidence[i] if hasattr(results, 'confidence') else None
                                
                                # 클래스별 색상 (간단한 해시 방식)
                                color = (hash(str(cls_id)) % 255, 
                                         (hash(str(cls_id)) * 2) % 255, 
                                         (hash(str(cls_id)) * 3) % 255)
                                
                                # 상자 그리기
                                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                                
                                # 텍스트 정보
                                cls_name = ontology.classes()[cls_id]
                                label = f"{cls_name}"
                                if conf is not None:
                                    label += f": {conf:.2f}"
                                
                                # 텍스트 그리기
                                cv2.putText(vis_image, label, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 결과 저장
                        cv2.imwrite(output_path, vis_image)
                        log(f"  - OpenCV로 시각화 저장됨: {output_path}", "DEBUG")
                    except Exception as e:
                        log(f"  - 시각화 오류: {e}", "WARNING")
            
            except Exception as e:
                log(f"이미지 처리 중 오류 {image_path}: {e}", "ERROR")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
        
        # Save object detection counts
        result["objects_detected"] = objects_detected
        
        # Save all detections to a single file
        all_detections_file = os.path.join(results_dir, "all_detections.json")
        with open(all_detections_file, 'w') as f:
            # Replace paths with relative paths for portability
            portable_detections = {}
            for path, data in all_detections.items():
                rel_path = os.path.basename(path)
                portable_detections[rel_path] = data
            
            json.dump(portable_detections, f, indent=2)
        
        result["output_paths"]["detection_results"] = detections_dir
        result["output_paths"]["all_detections"] = all_detections_file
        
        log(f"모든 탐지 결과 저장 완료: {all_detections_file}")
        
        # Generate YOLO annotations
        report_progress("YOLO 주석 생성", 75.0)
        log("\nYOLO 주석 생성 중")
        try:
            base_model.label(
                input_folder=str(images_dir),
                output_folder=str(dataset_dir),
                extension=".png"
            )
            log(f"YOLO 주석 생성 완료: {dataset_dir}")
            result["output_paths"]["yolo_annotations"] = str(dataset_dir)
        except Exception as e:
            log(f"YOLO 주석 생성 중 오류: {e}", "ERROR")
            if DEBUG:
                import traceback
                traceback.print_exc()
    else:
        report_progress("탐지 건너뛰기", 75.0)
    
    # Step 3: Preprocess images (if requested)
    if args.preprocess and preprocessed_dir and batch_preprocess_images is not None:
        report_progress("이미지 전처리 중", 80.0)
        log("\n탐지 결과를 기반으로 이미지 전처리 중")
        
        try:
            # Run batch preprocessing
            target_size = tuple(args.target_size) if args.target_size else (640, 640)
            processed_files = batch_preprocess_images(
                image_paths=image_paths,
                results_dir=os.path.join(results_dir, "detections"),
                output_dir=str(preprocessed_dir),
                target_size=target_size,
                apply_masks=not args.no_mask,
                crop_objects=not args.no_crop
            )
            
            # Save preprocessing results summary
            summary_file = os.path.join(results_dir, "preprocessing_summary.json")
            
            # Make paths relative for better portability
            summary = {}
            for img_path, processed_paths in processed_files.items():
                img_name = os.path.basename(img_path)
                processed_names = [os.path.basename(p) for p in processed_paths]
                summary[img_name] = processed_names
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Count total preprocessed files
            total_files = sum(len(files) for files in processed_files.values())
            log(f"이미지 {len(processed_files)}개를 {total_files}개 출력 파일로 전처리")
            log(f"전처리 요약 저장됨: {summary_file}")
            
            result["output_paths"]["preprocessed_images"] = str(preprocessed_dir)
            result["output_paths"]["preprocessing_summary"] = summary_file
        except Exception as e:
            log(f"이미지 전처리 중 오류: {e}", "ERROR")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    # Step 4: Train YOLO model (if requested)
    if args.train:
        report_progress("YOLO 모델 학습", 90.0)
        log("\nYOLO 모델 학습 중")
        
        try:
            # Load YOLO model
            yolo_model = YOLOv8(args.yolo_model)
            
            # Train on the dataset
            model_path = yolo_model.train(
                data=str(yaml_path),
                epochs=args.epochs,
                device=args.device,
                imgsz=args.target_size[0],
                batch=8,
                name=f"{args.category}_yolo"
            )
            
            log(f"YOLO 모델 학습 완료: {model_path}")
            result["output_paths"]["trained_model"] = str(model_path)
        except Exception as e:
            log(f"YOLO 모델 학습 중 오류: {e}", "ERROR")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    log(f"\n파이프라인 완료: {execution_time:.2f}초 소요")
    
    # Final progress
    report_progress("파이프라인 완료", 100.0)
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    run_autodistill_pipeline(args)


if __name__ == "__main__":
    main()

# 탐지 결과를 처리할 때 mask 변환 기능 추가
def handle_detection_results(image_path, results, mask_directory, class_names=None, save_npy=False):
    """
    탐지 결과를 처리하고 마스크를 저장하는 함수
    
    Args:
        image_path: 원본 이미지 경로
        results: 탐지 결과 객체
        mask_directory: 마스크 저장 디렉토리
        class_names: 클래스 이름 목록
        save_npy: NPY 파일로 마스크 저장 여부 (기본값: False)
    
    Returns:
        dict: 저장된 파일 경로
    """
    # 경로 및 파일 이름 설정
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 마스크가 있는 경우 마스크 기반 처리
    if hasattr(results, 'mask') and results.mask is not None:
        # 마스크 데이터를 NPY로 저장 (선택적)
        if save_npy:
            mask_npy_path = os.path.join(mask_directory, f"{img_basename}_mask.npy")
            np.save(mask_npy_path, results.mask)
            log(f"마스크 NPY 저장 완료: {mask_npy_path}", "DEBUG")
        
        # class_id가 있는 경우 클래스 이름 매핑
        if hasattr(results, 'class_id') and class_names:
            masks_class_names = [class_names[cls_id] for cls_id in results.class_id]
        else:
            masks_class_names = None
        
        # 마스크를 여러 형식으로 변환 및 저장
        return convert_and_save_mask(
            results.mask, 
            mask_directory, 
            img_basename, 
            masks_class_names,
            None  # 시각화 제거로 색상 맵이 필요 없음
        )
    
    # 마스크가 없지만 경계 상자가 있는 경우
    elif hasattr(results, 'xyxy') and len(results.xyxy) > 0:
        log(f"마스크 정보 없음, 경계 상자로 좌표 생성: {image_path}", "DEBUG")
        
        # 원본 이미지 크기 확인 (가능한 경우)
        try:
            image = cv2.imread(image_path)
            image_shape = (image.shape[1], image.shape[0])  # (너비, 높이)
        except:
            # 기본 이미지 크기 설정
            image_shape = (640, 480)
            log(f"이미지 크기를 확인할 수 없어 기본값 사용: {image_shape}", "WARNING")
        
        # class_id가 있는 경우 클래스 이름 매핑
        if hasattr(results, 'class_id') and class_names:
            masks_class_names = [class_names[cls_id] for cls_id in results.class_id]
        else:
            masks_class_names = None
        
        # 경계 상자만으로 좌표 데이터 생성 및 저장
        return save_coords_without_mask(
            results.xyxy.tolist(),
            mask_directory,
            img_basename,
            results.class_id.tolist() if hasattr(results, 'class_id') else None,
            masks_class_names,
            None,  # 시각화 제거로 색상맵 필요 없음
            image_shape
        )
    
    # 마스크도 경계 상자도 없는 경우
    else:
        log(f"마스크 정보 및 경계 상자 정보 없음: {image_path}", "DEBUG")
        return None 