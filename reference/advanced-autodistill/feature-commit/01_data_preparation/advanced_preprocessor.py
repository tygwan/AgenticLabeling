#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Image Preprocessor for Project-AGI

This module implements high-quality image preprocessing.
It prioritizes polygon coordinates from _coords.txt for mask generation,
applies the mask, then crops using box data from _box.json.

Key features:
1. Polygon-first masking: uses _coords.txt for precise mask generation.
2. Mask application: applies binary mask to original image (non-mask areas become black).
3. Box cropping: crops the object from the masked image using _box.json.
4. Class-specific processing and saving.

Processing Order for To-be style:
1. Load polygon coordinates from _coords.txt.
2. For each object's polygon, create a binary mask.
3. Apply this binary mask to the original image.
4. Load corresponding bounding box from _box.json.
5. Crop the box area from the mask-applied image.
6. Resize to target size.
7. Save to class-specific directory.
"""

import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_utils import get_category_path, load_class_mapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_preprocessor")

class RLEMaskProcessor: # RLE 관련 기능은 유지하되, 현재 시나리오에서는 직접 사용되지 않을 수 있음
    """RLE (Run Length Encoding) mask processor for precise object extraction."""
    
    @staticmethod
    def decode_rle(rle_data: List[int], image_shape: Tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros(height * width, dtype=np.uint8)
        for i in range(0, len(rle_data), 2):
            if i + 1 < len(rle_data):
                start = rle_data[i]
                length = rle_data[i + 1]
                mask[start:start + length] = 255
        return mask.reshape((height, width))
    
    @staticmethod
    def apply_mask_to_original_image(image: np.ndarray, mask: np.ndarray, background_mode: str = "black") -> np.ndarray:
        """
        마스크를 원본 이미지에 적용하여 마스크 영역만 컬러 정보를 유지하고 나머지는 지정된 배경색으로 처리합니다.
        
        Args:
            image: 원본 이미지 (RGB)
            mask: 마스크 (단일 채널, 0 또는 255)
            background_mode: 배경 처리 모드
                - "transparent": 투명 배경 (RGBA, 알파=0)
                - "black": 검은색 배경 (RGB 0,0,0)
                - "white": 흰색 배경 (RGB 255,255,255)
                - "gray": 회색 배경 (RGB 128,128,128)
                - "mean": 이미지 평균 색상 배경
            
        Returns:
            처리된 이미지 (RGB 또는 RGBA)
        """
        if mask.shape[:2] != image.shape[:2]:
            logger.warning(f"Mask shape {mask.shape} doesn't match image shape {image.shape[:2]}. Resizing mask.")
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 마스크가 255인 영역은 1로, 나머지는 0으로 변환 (이진 마스크)
        mask_binary = mask.astype(np.bool_).astype(np.uint8)
        
        if background_mode == "transparent":
            # RGBA 이미지 생성 (기존 RGB + 알파 채널)
            if len(image.shape) == 3 and image.shape[2] == 3:  # RGB 이미지
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = image  # RGB 채널 복사
                rgba[:, :, 3] = mask_binary * 255  # 알파 채널: 마스크 영역은 불투명(255), 나머지는 투명(0)
            elif len(image.shape) == 2:  # 흑백 이미지
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[:, :, 0] = image  # 그레이스케일 값을 R 채널에 복사
                rgba[:, :, 1] = image  # 그레이스케일 값을 G 채널에 복사
                rgba[:, :, 2] = image  # 그레이스케일 값을 B 채널에 복사
                rgba[:, :, 3] = mask_binary * 255  # 알파 채널
            else:
                logger.error(f"Unsupported image format with shape {image.shape}")
                return image  # 지원되지 않는 형식인 경우 원본 반환
            return rgba
        
        else:
            # RGB 이미지로 처리 (비마스크 영역을 특정 색상으로 채움)
            if len(image.shape) == 3 and image.shape[2] == 3:  # RGB 이미지
                result = image.copy()
            elif len(image.shape) == 2:  # 흑백 이미지를 RGB로 변환
                result = np.stack([image, image, image], axis=2)
            else:
                logger.error(f"Unsupported image format with shape {image.shape}")
                return image
            
            # 비마스크 영역 색상 설정
            if background_mode == "black":
                background_color = [0, 0, 0]
            elif background_mode == "white":
                background_color = [255, 255, 255]
            elif background_mode == "gray":
                background_color = [128, 128, 128]
            elif background_mode == "mean":
                # 마스크 영역의 평균 색상 계산
                masked_pixels = result[mask_binary == 1]
                if len(masked_pixels) > 0:
                    background_color = np.mean(masked_pixels, axis=0).astype(int)
                else:
                    background_color = [128, 128, 128]  # 기본값
            else:
                logger.warning(f"Unknown background_mode: {background_mode}. Using black.")
                background_color = [0, 0, 0]
            
            # 비마스크 영역을 지정된 색상으로 채움
            inverse_mask = (mask_binary == 0)
            result[inverse_mask] = background_color
            
            return result
    
    @staticmethod
    def crop_box_from_masked_image(masked_image: np.ndarray, box: Tuple[int, int, int, int], 
                                 padding: int = 0) -> np.ndarray: # To-be는 패딩이 거의 없어 보이므로 기본값 0
        """
        마스크가 적용된 이미지(RGBA 포함)에서 경계 상자 영역을 크롭합니다.
        
        Args:
            masked_image: 마스크가 적용된 이미지 (RGB 또는 RGBA)
            box: 경계 상자 좌표 (x1, y1, x2, y2)
            padding: 경계 상자 주위에 추가할 패딩 픽셀 수
            
        Returns:
            크롭된 이미지 (RGB 또는 RGBA)
        """
        x1, y1, x2, y2 = map(int, box) # Ensure integer coordinates
        image_height, image_width = masked_image.shape[:2]
        
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image_width, x2 + padding)
        y2_pad = min(image_height, y2 + padding)

        if y1_pad >= y2_pad or x1_pad >= x2_pad:
            logger.warning(f"Invalid crop dimensions after padding: y1={y1_pad}, y2={y2_pad}, x1={x1_pad}, x2={x2_pad}. Box: {box}")
            return np.array([]) # Return empty array for invalid crop
            
        cropped_image = masked_image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        return cropped_image

class AdvancedPreprocessor:
    def __init__(self, category_name: str, target_size: Tuple[int, int] = (224, 224), background_mode: str = "black"):
        self.category_name = category_name
        self.target_size = target_size
        self.background_mode = background_mode
        self.category_path = get_category_path(category_name)
        
        self.images_dir = os.path.join(self.category_path, "1.images")
        self.box_dir = os.path.join(self.category_path, "3.box")
        self.mask_coords_dir = os.path.join(self.category_path, "4.mask") # _coords.txt 파일이 있는 디렉토리
        self.output_dir = os.path.join(self.category_path, "6.preprocessed")
        
        try:
            self.class_mapping = load_class_mapping(category_name)
            logger.info(f"Loaded class mapping: {self.class_mapping}")
        except Exception as e:
            logger.warning(f"Could not load class mapping: {e}. Using default class names.")
            self.class_mapping = {} # Fallback
        
        self.utils = RLEMaskProcessor() # Re-using helper methods
        self._setup_output_directories()
        logger.info(f"배경 모드: {self.background_mode} (마스크되지 않은 영역 처리 방식)")
    
    def _setup_output_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Assuming up to 4 classes based on context, adjust if more are possible
        num_expected_classes = len(self.class_mapping) if self.class_mapping else 4 
        for class_id in range(num_expected_classes):
            class_name = self.class_mapping.get(str(class_id), f"Class_{class_id}")
            class_dir_name = f"Class_{class_id}" # 폴더명은 ID 기반 유지
            class_dir = os.path.join(self.output_dir, class_dir_name)
            os.makedirs(class_dir, exist_ok=True)
            logger.debug(f"Ensured output directory: {class_dir}")

    def _parse_box_points_file(self, file_path: str) -> Optional[Dict]:
        """
        Parse a box points file (debug format) into the same structure as box.json
        
        Box points format example:
        Box: [0.7919999957084656, 1.4079999923706055, 108.50399780273438, 153.4720001220703], Class ID: 0, Confidence: 1.0000
        
        Returns:
            Dict with 'boxes' and 'class_ids' keys (same structure as _box.json)
        """
        try:
            boxes = []
            class_ids = []
            confidence_scores = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse box information
                    try:
                        # Extract box coordinates using regex or simple parsing
                        box_start = line.find("[") + 1
                        box_end = line.find("]")
                        if box_start > 0 and box_end > box_start:
                            box_str = line[box_start:box_end]
                            box_coords = [float(x.strip()) for x in box_str.split(",")]
                            
                            # Extract class ID
                            class_id_start = line.find("Class ID:") + len("Class ID:")
                            class_id_end = line.find(",", class_id_start)
                            if class_id_end == -1:  # If no comma (end of line)
                                class_id_end = len(line)
                            class_id = int(line[class_id_start:class_id_end].strip())
                            
                            # Extract confidence if available
                            confidence = 1.0
                            conf_start = line.find("Confidence:")
                            if conf_start > 0:
                                conf_start += len("Confidence:")
                                conf_end = len(line)
                                confidence = float(line[conf_start:conf_end].strip())
                            
                            boxes.append(box_coords)
                            class_ids.append(class_id)
                            confidence_scores.append(confidence)
                    except Exception as e:
                        logger.warning(f"Failed to parse line in box points file: {line} - Error: {e}")
            
            if boxes:
                return {
                    "boxes": boxes,
                    "class_ids": class_ids,
                    "confidence": confidence_scores
                }
            return None
        except Exception as e:
            logger.error(f"Error parsing box points file {file_path}: {e}")
            return None

    def load_image_data(self, image_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict], List[Dict]]:
        """
        Load image, box data, and polygon coordinate data.
        Polygon data is a list of dictionaries, each for one object: {'class_id': int, 'polygon_norm': [x,y,x,y...]}
        """
        # Log directories being used for debugging
        logger.debug(f"Loading data for image: {image_name}")
        logger.debug(f"- Images directory: {self.images_dir}")
        logger.debug(f"- Box directory: {self.box_dir}")
        logger.debug(f"- Mask/coords directory: {self.mask_coords_dir}")
        
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            potential_path = os.path.join(self.images_dir, f"{image_name}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            logger.warning(f"Image not found for: {image_name}")
            return None, None, []
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, None, []
        
        # Try to load box data from standard format first (_box.json)
        box_file = os.path.join(self.box_dir, f"{image_name}_box.json")
        box_data = None
        
        if os.path.exists(box_file):
            try:
                with open(box_file, 'r') as f:
                    box_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load box data from {box_file}: {e}")
        
        # If standard box format doesn't exist or failed to load, try debug format (_box_points.txt)
        if not box_data:
            # First try looking in the mask directory
            box_points_file = os.path.join(self.mask_coords_dir, f"{image_name}_box_points.txt")
            
            # If not found in mask directory, try in the box directory
            if not os.path.exists(box_points_file):
                box_points_file = os.path.join(self.box_dir, f"{image_name}_box_points.txt")
            
            if os.path.exists(box_points_file):
                logger.debug(f"Using debug format box file: {box_points_file}")
                box_data = self._parse_box_points_file(box_points_file)
        
        # Try multiple polygon coordinate file formats
        polygon_objects_data = []  # [{'class_id': id, 'polygon_norm': [coords]}, ...]
        
        # Check for standard format (_coords.txt)
        coords_txt_file = os.path.join(self.mask_coords_dir, f"{image_name}_coords.txt")
        
        # If standard format doesn't exist, check for debug format ({image_name}.txt)
        if not os.path.exists(coords_txt_file):
            debug_txt_file = os.path.join(self.mask_coords_dir, f"{image_name}.txt")
            if os.path.exists(debug_txt_file):
                coords_txt_file = debug_txt_file
                logger.debug(f"Using debug format polygon file: {debug_txt_file}")
        
        if os.path.exists(coords_txt_file):
            try:
                with open(coords_txt_file, 'r') as f:
                    for line_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 3 and (len(parts) -1) % 2 == 0 : # class_id + at least one pair of coords
                            try:
                                class_id = int(parts[0])
                                norm_coords = [float(p) for p in parts[1:]]
                                polygon_objects_data.append({'class_id': class_id, 'polygon_norm': norm_coords, 'original_line_idx': line_idx})
                            except ValueError:
                                logger.warning(f"Invalid data in {coords_txt_file}, line {line_idx+1}: {line.strip()}")
                        elif parts: # Non-empty line that doesn't match format
                             logger.warning(f"Malformed line in {coords_txt_file}, line {line_idx+1}: {line.strip()}")
            except Exception as e:
                logger.error(f"Failed to load polygon coordinate data from {coords_txt_file}: {e}")
        else:
            logger.debug(f"No polygon coordinate file found (tried both standard and debug formats). Proceeding without polygon masks.")

        # Log what was successfully loaded
        logger.debug(f"Data loaded for {image_name}: " + 
                   f"Image {'✓' if image is not None else '✗'}, " +
                   f"Box data {'✓' if box_data else '✗'}, " +
                   f"Polygon data {'✓' if polygon_objects_data else '✗'} ({len(polygon_objects_data)} objects)")
                   
        return image, box_data, polygon_objects_data
    
    def process_single_image(self, image_name: str, selected_class_id: Optional[int] = None) -> Dict[str, Any]:
        results = {"image_name": image_name, "processed_objects": 0, "saved_files": [], "errors": []}
        image, box_data, polygon_objects_data = self.load_image_data(image_name)

        if image is None:
            results["errors"].append("Failed to load image")
            return results
        
        image_height, image_width = image.shape[:2]

        if not box_data or not box_data.get("boxes") or not box_data.get("class_ids"):
            results["errors"].append("Box data (boxes or class_ids) is missing or empty.")
            # If no polygon data either, then nothing to process
            if not polygon_objects_data:
                return results
        
        # Use polygon_objects_data as the primary source for objects to process if available
        # Fallback to box_data if polygon_objects_data is empty
        
        objects_to_process = []
        if polygon_objects_data:
            # Assuming the order in _coords.txt matches the order in _box.json
            # This is a common assumption for outputs from models like GroundedSAM
            for i, poly_obj in enumerate(polygon_objects_data):
                if i < len(box_data.get("boxes", [])):
                    box = box_data["boxes"][i]
                    box_class_id = box_data["class_ids"][i]
                    
                    # Consistency check (optional, but good for debugging)
                    if poly_obj['class_id'] != box_class_id:
                        logger.warning(f"Class ID mismatch for object {i} in {image_name}. "
                                       f"Polygon Class ID: {poly_obj['class_id']}, Box Class ID: {box_class_id}. "
                                       f"Using Polygon Class ID: {poly_obj['class_id']}.")
                    
                    objects_to_process.append({
                        "class_id": poly_obj['class_id'], # Prioritize polygon's class_id
                        "box": box, # Absolute box coordinates
                        "polygon_norm": poly_obj['polygon_norm'], # Normalized polygon
                        "obj_idx_in_file": poly_obj['original_line_idx'] # Original index from _coords.txt
                    })
                else:
                    logger.warning(f"Polygon object at index {i} in {image_name} does not have a corresponding box. Skipping.")
        elif box_data and box_data.get("boxes"): # Fallback if no polygon data
            logger.info(f"No polygon data for {image_name}. Processing with boxes only (no mask application).")
            for i, box in enumerate(box_data["boxes"]):
                 objects_to_process.append({
                        "class_id": box_data["class_ids"][i],
                        "box": box,
                        "polygon_norm": None, # No polygon
                        "obj_idx_in_file": i 
                    })
        else: # No box data and no polygon data
            results["errors"].append("No box data or polygon data to process.")
            return results

        # 마스크 NPY 파일 확인 (마스크 배열 그대로 저장한 경우)
        mask_npy_file = os.path.join(self.mask_coords_dir, f"{image_name}_mask.npy")
        mask_multi_npy_file = os.path.join(self.mask_coords_dir, f"{image_name}_masks.npy")
        
        masks_from_npy = None
        if os.path.exists(mask_multi_npy_file):
            try:
                masks_from_npy = np.load(mask_multi_npy_file)
                logger.info(f"다중 마스크 NPY 파일을 로드했습니다: {mask_multi_npy_file}, 형태: {masks_from_npy.shape}")
            except Exception as e:
                logger.warning(f"다중 마스크 NPY 파일 로드 실패: {e}")
        elif os.path.exists(mask_npy_file):
            try:
                masks_from_npy = np.load(mask_npy_file)
                # 단일 마스크인 경우 차원 추가 (n, h, w) 형태로 통일
                if len(masks_from_npy.shape) == 2:
                    masks_from_npy = np.expand_dims(masks_from_npy, axis=0)
                logger.info(f"단일 마스크 NPY 파일을 로드했습니다: {mask_npy_file}, 형태: {masks_from_npy.shape}")
            except Exception as e:
                logger.warning(f"단일 마스크 NPY 파일 로드 실패: {e}")

        for obj_idx, obj_data in enumerate(objects_to_process):
            try:
                current_class_id = obj_data["class_id"]
                if selected_class_id is not None and current_class_id != selected_class_id:
                    continue

                abs_box = obj_data["box"] # These are already absolute pixel values from _box.json
                x1, y1, x2, y2 = map(int, abs_box)
                
                # 마스크 생성 단계
                binary_mask = None
                
                # 1. NPY 파일에서 마스크를 로드한 경우 (우선순위 높음)
                if masks_from_npy is not None and obj_idx < masks_from_npy.shape[0]:
                    mask_from_array = masks_from_npy[obj_idx]
                    if mask_from_array.dtype == bool:
                        binary_mask = mask_from_array.astype(np.uint8) * 255
                    else:
                        binary_mask = (mask_from_array > 0).astype(np.uint8) * 255
                    logger.info(f"NPY 파일에서 마스크를 로드했습니다 (객체 {obj_idx})")
                
                # 2. 폴리곤 좌표에서 마스크 생성 (NPY 파일이 없는 경우)
                elif obj_data["polygon_norm"]:
                    binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    poly_points_abs = []
                    norm_coords = obj_data["polygon_norm"]
                    for i in range(0, len(norm_coords), 2):
                        if i+1 < len(norm_coords):  # 좌표 쌍 확인
                            px, py = int(norm_coords[i] * image_width), int(norm_coords[i+1] * image_height)
                            poly_points_abs.append([px, py])
                    
                    if poly_points_abs:
                        pts_array = np.array(poly_points_abs, dtype=np.int32)
                        cv2.fillPoly(binary_mask, [pts_array], 255)
                        logger.info(f"폴리곤 좌표에서 마스크를 생성했습니다 ({len(poly_points_abs)} 점)")
                    else:
                        logger.warning(f"객체 {obj_idx}에 대한 유효한 폴리곤 점이 없습니다. 마스크 없이 진행합니다.")
                
                # 마스크 적용 및 크롭
                if binary_mask is not None:
                    # 마스크 적용 (배경 모드에 따라 처리)
                    masked_image = self.utils.apply_mask_to_original_image(image, binary_mask, self.background_mode)
                    if self.background_mode == "transparent":
                        logger.info(f"마스크가 적용된 RGBA 이미지 생성 (형태: {masked_image.shape}, 배경: 투명)")
                    else:
                        logger.info(f"마스크가 적용된 RGB 이미지 생성 (형태: {masked_image.shape}, 배경: {self.background_mode})")
                else:
                    # 마스크가 없는 경우 원본 이미지 사용
                    logger.warning(f"객체 {obj_idx}에 대한 마스크가 없습니다. 마스크 없이 원본 이미지를 크롭합니다.")
                    masked_image = image.copy()
                
                # 마스크가 적용된 이미지에서 박스 영역 크롭
                cropped_object_img = self.utils.crop_box_from_masked_image(masked_image, (x1, y1, x2, y2), padding=0)

                if cropped_object_img.size == 0:
                    results["errors"].append(f"Empty cropped object for obj_idx {obj_data['obj_idx_in_file']} of class {current_class_id}")
                    continue
                
                # 크롭된 이미지 리사이즈
                if self.target_size:
                    resized_object_img = cv2.resize(cropped_object_img, self.target_size, interpolation=cv2.INTER_AREA)
                else:
                    resized_object_img = cropped_object_img
                
                # 출력 파일 경로 설정
                class_name_str = self.class_mapping.get(str(current_class_id), f"unknown_class_{current_class_id}")
                output_filename = f"{image_name}_obj{obj_data['obj_idx_in_file']}_cls{current_class_id}_{class_name_str}.png"
                
                class_dir_name = f"Class_{current_class_id}"
                output_class_dir = os.path.join(self.output_dir, class_dir_name)
                os.makedirs(output_class_dir, exist_ok=True)

                output_path = os.path.join(output_class_dir, output_filename)
                
                # PNG 형식으로 저장 (알파 채널 지원)
                if cv2.imwrite(output_path, resized_object_img):
                    logger.info(f"객체 이미지 저장 완료: {output_path}")
                    results["saved_files"].append(output_path)
                    results["processed_objects"] += 1
                else:
                    results["errors"].append(f"Failed to save {output_path}")

            except Exception as e:
                error_msg = f"Error processing object_idx {obj_data['obj_idx_in_file']} in {image_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results["errors"].append(error_msg)
        
        return results

    def process_all_images(self, selected_class_id: Optional[int] = None, 
                          max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all images in the category, but first prefilter to only valid images with data.
        This optimizes processing by avoiding images without mask/box data.
        
        Args:
            selected_class_id: Optional class ID to filter for
            max_images: Maximum number of images to process
            
        Returns:
            Dict with processing statistics
        """
        image_files_basenames = []
        for file in os.listdir(self.images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files_basenames.append(os.path.splitext(file)[0])
        
        # 전처리 최적화: 유효한 데이터가 있는 이미지만 먼저 필터링
        logger.info(f"전체 이미지 {len(image_files_basenames)}개 중 유효한 데이터가 있는 이미지 필터링 중...")
        valid_image_basenames = []
        
        # 마스크 폴더에서 폴리곤 데이터가 있는 이미지 찾기
        polygon_files = set()
        for file in os.listdir(self.mask_coords_dir):
            # 표준 형식 (_coords.txt) 또는 디버그 형식 (.txt, _box_points.txt 제외)
            if (file.endswith('_coords.txt') or 
                (file.endswith('.txt') and not file.endswith('_box_points.txt'))):
                if file.endswith('_coords.txt'):
                    img_name = file[:-11]  # Remove '_coords.txt'
                else:
                    img_name = os.path.splitext(file)[0]
                polygon_files.add(img_name)
                
        # 박스 폴더에서 박스 데이터가 있는 이미지 찾기
        box_files = set()
        for file in os.listdir(self.box_dir):
            if file.endswith('_box.json'):
                img_name = file[:-9]  # Remove '_box.json'
                box_files.add(img_name)
            elif file.endswith('_box_points.txt'):
                img_name = file[:-15]  # Remove '_box_points.txt'
                box_files.add(img_name)
        
        # 폴리곤 데이터나 박스 데이터가 있는 이미지만 필터링
        for img_name in image_files_basenames:
            if img_name in polygon_files or img_name in box_files:
                valid_image_basenames.append(img_name)
        
        logger.info(f"전체 이미지: {len(image_files_basenames)}개")
        logger.info(f"폴리곤 데이터가 있는 이미지: {len(polygon_files)}개")
        logger.info(f"박스 데이터가 있는 이미지: {len(box_files)}개")
        logger.info(f"유효한 데이터가 있는 이미지: {len(valid_image_basenames)}개")
        
        # 최대 처리 이미지 수 제한 적용
        if max_images is not None and max_images > 0 :
            valid_image_basenames = valid_image_basenames[:max_images]
        
        logger.info(f"처리할 이미지 수: {len(valid_image_basenames)}개")
        if selected_class_id is not None:
            logger.info(f"클래스 ID 필터링: {selected_class_id}")
        
        overall_results = {
            "total_images_scanned": len(image_files_basenames), 
            "valid_images_found": len(valid_image_basenames),
            "processed_images_with_objects": 0,
            "total_objects_extracted": 0, 
            "total_files_saved": 0, 
            "errors": [], 
            "class_counts": {}
        }
        
        for image_name in tqdm(valid_image_basenames, desc="Advanced Preprocessing Images"):
            try:
                result = self.process_single_image(image_name, selected_class_id)
                if result["processed_objects"] > 0:
                    overall_results["processed_images_with_objects"] += 1
                overall_results["total_objects_extracted"] += result["processed_objects"]
                overall_results["total_files_saved"] += len(result["saved_files"])
                if result["errors"]:
                    overall_results["errors"].extend([f"{image_name}: {err}" for err in result["errors"]])
            except Exception as e:
                error_msg = f"Major failure processing {image_name}: {str(e)}"
                overall_results["errors"].append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        for class_id_str, class_name_mapped in self.class_mapping.items():
            class_dir = os.path.join(self.output_dir, f"Class_{class_id_str}")
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
                overall_results["class_counts"][class_name_mapped] = count
            else: # also check for Class_X if mapping is not exhaustive
                class_dir_fallback = os.path.join(self.output_dir, f"Class_{class_id_str}")
                if os.path.exists(class_dir_fallback) and class_dir_fallback != class_dir : # if exists and not already counted
                     count = len([f for f in os.listdir(class_dir_fallback) if f.endswith('.png')])
                     overall_results["class_counts"][f"Class_{class_id_str}"] = count

        return overall_results
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """
        처리 결과에 대한 요약 보고서 생성
        """
        report_lines = []
        report_lines.append(f"Category: {self.category_name}")
        report_lines.append(f"Target Size: {self.target_size}")
        report_lines.append(f"Background Mode: {self.background_mode} (마스크 비적용 영역 처리)")
        report_lines.append(f"Processing Method: Mask Array/Polygon > Crop Box > Resize")
        report_lines.append("")
        report_lines.append("Processing Details:")
        report_lines.append(f"- First Priority: NPY 마스크 배열 (.npy) - 원본 마스크 데이터 사용")
        report_lines.append(f"- Second Priority: 폴리곤 좌표 (_coords.txt) - 폴리곤으로 마스크 생성")
        
        # 배경 모드에 따른 설명 추가
        if self.background_mode == "black":
            report_lines.append(f"- 마스크는 원본 이미지에 적용되어 객체 영역만 보존, 배경은 검은색(0,0,0) - CNN 모델에 적합")
        elif self.background_mode == "white":
            report_lines.append(f"- 마스크는 원본 이미지에 적용되어 객체 영역만 보존, 배경은 흰색(255,255,255)")
        elif self.background_mode == "gray":
            report_lines.append(f"- 마스크는 원본 이미지에 적용되어 객체 영역만 보존, 배경은 회색(128,128,128)")
        elif self.background_mode == "transparent":
            report_lines.append(f"- 마스크는 원본 이미지에 적용되어 객체 영역만 보존, 배경은 투명(알파=0) - PNG 파일 필요")
        elif self.background_mode == "mean":
            report_lines.append(f"- 마스크는 원본 이미지에 적용되어 객체 영역만 보존, 배경은 객체 영역의 평균 색상")
        
        report_lines.append(f"- 박스 데이터는 크롭 영역 지정에만 사용")
        report_lines.append("")
        report_lines.append("Preprocessing Optimization:")
        report_lines.append(f"- Total Images Available: {results.get('total_images_scanned', 0)}")
        report_lines.append(f"- Images with Valid Data: {results.get('valid_images_found', 0)}")
        
        if results.get('total_images_scanned', 0) > 0:
            optimization_ratio = results.get('valid_images_found', 0) / results.get('total_images_scanned', 0) * 100
            report_lines.append(f"- Optimization Ratio: {optimization_ratio:.1f}% of images processed")
        
        report_lines.append("")
        report_lines.append("Processing Results:")
        report_lines.append(f"- Images with Valid Data: {results.get('valid_images_found', 0)}")
        report_lines.append(f"- Images with Objects Processed: {results.get('processed_images_with_objects', 0)}")
        report_lines.append(f"- Total Objects Extracted: {results.get('total_objects_extracted', 0)}")
        report_lines.append(f"- Total Files Saved: {results.get('total_files_saved', 0)}")
        
        report_lines.append("")
        
        # 클래스별 저장된 파일 수 표시
        report_lines.append("Class Distribution of Saved Files:")
        class_counts = results.get('class_counts', {})
        if class_counts:
            for class_name, count in class_counts.items():
                report_lines.append(f"- {class_name}: {count}")
        else:
            report_lines.append("- No objects saved or class counts not available.")
        
        # 오류 표시 (요약 수준)
        errors = results.get('errors', [])
        if errors:
            report_lines.append("")
            report_lines.append(f"Errors Summary: {len(errors)} errors occurred.")
            report_lines.append("- See logs for details.")
        
        return "\n".join(report_lines)

def parse_args_advanced(): # Renamed to avoid conflict if run from main_launcher
    parser = argparse.ArgumentParser(description="Advanced Image Preprocessor (CLI standalone)")
    parser.add_argument("--category", type=str, required=True, help="Category name to process")
    parser.add_argument("--class_id", type=int, default=None, help="Specific class ID to extract. If None, extracts all classes.")
    parser.add_argument("--target_size", type=str, default="224,224", help="Target size (width,height)")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--background_mode", type=str, default="black", 
                        choices=["black", "white", "gray", "transparent", "mean"],
                        help="배경 처리 방식: black(검정, CNN 기본), white(흰색), gray(회색), transparent(투명, RGBA), mean(평균색)")
    parser.add_argument("--output_report_path", type=str, default=None, help="Path to save processing report txt file")
    return parser.parse_args()

def main_advanced_preprocessor_cli(): # Renamed main function
    args = parse_args_advanced()
    try:
        width, height = map(int, args.target_size.split(','))
        target_sz = (width, height)
    except ValueError:
        logger.error(f"Invalid target size format: {args.target_size}. Use 'width,height'.")
        return

    preprocessor = AdvancedPreprocessor(
        category_name=args.category, 
        target_size=target_sz,
        background_mode=args.background_mode
    )
    logger.info(f"Starting advanced preprocessing for category '{args.category}'...")
    results_data = preprocessor.process_all_images(selected_class_id=args.class_id, max_images=args.max_images)
    
    summary_report_str = preprocessor.create_summary_report(results_data)
    print("\n" + summary_report_str)
    
    if args.output_report_path:
        try:
            with open(args.output_report_path, 'w', encoding='utf-8') as f:
                f.write(summary_report_str)
            logger.info(f"Summary report saved to: {args.output_report_path}")
        except Exception as e:
            logger.error(f"Failed to save report to {args.output_report_path}: {e}")
            
    logger.info("Advanced preprocessing (CLI standalone) finished.")

if __name__ == "__main__":
    # This allows running advanced_preprocessor.py directly as a script for testing
    main_advanced_preprocessor_cli() 