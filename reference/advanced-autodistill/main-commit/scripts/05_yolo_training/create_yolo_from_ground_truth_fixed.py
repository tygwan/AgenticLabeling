#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth 기반 YOLO Segmentation 데이터셋 생성

Ground truth 폴더의 9,417개 이미지를 기준으로 
YOLO segmentation 데이터셋을 생성합니다.
"""

import os
import cv2
import json
import numpy as np
import logging
import shutil
import random
import re
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_frame_from_ground_truth_filename(filename):
    """
    Ground truth 파일명에서 프레임명 추출
    예: 'G_1_2_frame_0073_obj2_cls3_unknown_class_3.png' -> 'G_1_2_frame_0073'
    """
    # _obj가 나오기 전까지의 부분을 추출
    match = re.match(r'(.+?)_obj\d+', filename)
    if match:
        return match.group(1)
    return None

def extract_object_index_from_filename(filename):
    """
    Ground truth 파일명에서 객체 인덱스 추출
    예: 'G_1_2_frame_0073_obj2_cls3_unknown_class_3.png' -> 2
    """
    match = re.search(r'_obj(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def extract_class_from_folder_name(folder_name):
    """
    폴더명에서 클래스 ID 추출
    예: 'Class_0' -> 0, 'Class_1' -> 1
    """
    if folder_name.startswith('Class_'):
        try:
            return int(folder_name.split('_')[1])
        except (IndexError, ValueError):
            return None
    return None

def get_object_mask_from_masks_file(mask_file_path, object_index):
    """
    Masks 파일에서 특정 객체의 마스크 추출
    """
    try:
        if not os.path.exists(mask_file_path):
            return None
        
        masks = np.load(mask_file_path)
        if len(masks.shape) == 3 and object_index < masks.shape[0]:
            return masks[object_index]
        elif len(masks.shape) == 2:
            # 단일 마스크인 경우
            return masks if object_index == 0 else None
        return None
    except Exception as e:
        logger.error(f"마스크 로드 실패 {mask_file_path}: {e}")
        return None

def load_box_annotation(box_dir, frame_name):
    """
    Box annotation 파일 로드
    """
    box_file = os.path.join(box_dir, f"{frame_name}_box.json")
    if os.path.exists(box_file):
        try:
            with open(box_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Box annotation 로드 실패 {box_file}: {e}")
    return None

def load_mask_data(mask_dir, frame_name):
    """
    마스크 데이터 로드 (.npy 파일)
    """
    mask_file = os.path.join(mask_dir, f"{frame_name}_masks.npy")
    if os.path.exists(mask_file):
        try:
            return np.load(mask_file)
        except Exception as e:
            logger.error(f"마스크 로드 실패 {mask_file}: {e}")
    return None

def mask_to_polygon(mask, tolerance=1.0, min_area=100):
    """
    마스크를 polygon 좌표로 변환 (정규화된 좌표)
    """
    try:
        if mask is None or mask.size == 0:
            return None
        
        # 마스크를 uint8로 변환
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        elif mask.dtype == np.uint8:
            mask_uint8 = mask
        else:
            # 0-1 범위라고 가정
            mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 너무 작은 마스크는 스킵
        if np.sum(mask_uint8 > 128) < min_area:
            return None
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 너무 작은 컨투어는 스킵
        if cv2.contourArea(largest_contour) < min_area:
            return None
        
        # 컨투어 단순화
        epsilon = tolerance * cv2.arcLength(largest_contour, True) / 100
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 정규화된 좌표로 변환
        h, w = mask.shape
        polygon = []
        for point in simplified_contour:
            x, y = point[0]
            # 좌표를 정규화하고 범위 제한
            norm_x = max(0.0, min(1.0, x / w))
            norm_y = max(0.0, min(1.0, y / h))
            polygon.extend([norm_x, norm_y])
        
        # 최소 3개 점 (6개 좌표) 필요
        return polygon if len(polygon) >= 6 else None
        
    except Exception as e:
        logger.error(f"Polygon 변환 실패: {e}")
        return None

def create_yolo_from_ground_truth(
    category_path: str,
    target_classes: List[int] = [0, 1, 2, 3],
    output_dir: str = "yolo_dataset_from_gt",
    target_size: Tuple[int, int] = (640, 640),
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)
):
    """
    Ground Truth 기반 YOLO segmentation 데이터셋 생성
    """
    logger.info(f"Ground Truth 기반 YOLO Segmentation 데이터셋 생성 시작")
    logger.info(f"카테고리: {category_path}")
    logger.info(f"대상 클래스: {target_classes}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"대상 크기: {target_size}")
    logger.info(f"분할 비율: train:{split_ratios[0]}, val:{split_ratios[1]}, test:{split_ratios[2]}")
    
    # 경로 설정
    ground_truth_dir = os.path.join(category_path, "7.results", "ground_truth")
    box_dir = os.path.join(category_path, "3.box")
    mask_dir = os.path.join(category_path, "4.mask")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # Ground truth 이미지 수집
    all_images = []
    processed_frames = set()
    skipped_unknown = 0
    skipped_no_annotation = 0
    
    # 대상 클래스 폴더 정의
    target_class_folders = [f"Class_{i}" for i in target_classes]
    
    logger.info(f"Ground truth 폴더 스캔: {ground_truth_dir}")
    
    # 모든 클래스 폴더에서 이미지 수집
    for class_folder in os.listdir(ground_truth_dir):
        class_path = os.path.join(ground_truth_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        # 폴더명에서 클래스 ID 추출
        folder_class_id = extract_class_from_folder_name(class_folder)
        
        # Unknown 클래스나 대상 클래스가 아닌 경우 스킵
        if folder_class_id is None or folder_class_id not in target_classes:
            unknown_count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
            skipped_unknown += unknown_count
            logger.info(f"스킵: {class_folder} ({unknown_count}개 이미지) - 대상 클래스 아님")
            continue
        
        logger.info(f"처리 중: {class_folder} (클래스 {folder_class_id})")
        
        for filename in os.listdir(class_path):
            if not filename.endswith('.png'):
                continue
            
            # 프레임명 추출
            frame_name = extract_frame_from_ground_truth_filename(filename)
            if frame_name is None:
                continue
            
            # 객체 인덱스 추출
            object_index = extract_object_index_from_filename(filename)
            if object_index is None:
                continue
            
            # Box annotation 존재 확인 (선택사항)
            box_file = os.path.join(box_dir, f"{frame_name}_box.json")
            has_annotation = os.path.exists(box_file)
            
            # Mask 파일 존재 확인
            mask_file = os.path.join(mask_dir, f"{frame_name}_masks.npy")
            has_mask = os.path.exists(mask_file)
            
            image_path = os.path.join(class_path, filename)
            all_images.append({
                'path': image_path,
                'filename': filename,
                'frame_name': frame_name,
                'object_index': object_index,
                'class_id': folder_class_id,  # 폴더명 기반 클래스 ID 사용
                'class_folder': class_folder,
                'has_annotation': has_annotation,
                'has_mask': has_mask
            })
            processed_frames.add(frame_name)
    
    logger.info(f"필터링 결과:")
    logger.info(f"  - 총 수집된 이미지: {len(all_images)}개")
    logger.info(f"  - 고유 프레임: {len(processed_frames)}개")
    logger.info(f"  - 스킵된 Unknown: {skipped_unknown}개")
    logger.info(f"  - Annotation 있는 이미지: {sum(1 for img in all_images if img['has_annotation'])}개")
    logger.info(f"  - Annotation 없는 이미지: {sum(1 for img in all_images if not img['has_annotation'])}개")
    logger.info(f"  - Mask 있는 이미지: {sum(1 for img in all_images if img['has_mask'])}개")
    logger.info(f"  - Mask 없는 이미지: {sum(1 for img in all_images if not img['has_mask'])}개")
    
    logger.info(f"클래스별 분포:")
    for class_id in target_classes:
        count = sum(1 for img in all_images if img['class_id'] == class_id)
        with_anno = sum(1 for img in all_images if img['class_id'] == class_id and img['has_annotation'])
        with_mask = sum(1 for img in all_images if img['class_id'] == class_id and img['has_mask'])
        logger.info(f"  Class {class_id}: {count}개 (annotation: {with_anno}개, mask: {with_mask}개)")
    
    # 이미지 수가 너무 적으면 경고
    if len(all_images) < 100:
        logger.warning(f"이미지 수가 예상보다 적습니다: {len(all_images)}개")
        logger.warning("Ground Truth 폴더 구조를 확인하세요.")
    elif len(all_images) > 8000:
        logger.info(f"대용량 데이터셋 생성 중: {len(all_images)}개 이미지")
    
    # 데이터셋 분할 (프레임 기준)
    unique_frames = list(processed_frames)
    random.shuffle(unique_frames)
    
    train_split = int(len(unique_frames) * split_ratios[0])
    val_split = int(len(unique_frames) * (split_ratios[0] + split_ratios[1]))
    
    frame_splits = {
        'train': set(unique_frames[:train_split]),
        'val': set(unique_frames[train_split:val_split]),
        'test': set(unique_frames[val_split:])
    }
    
    # 이미지를 split별로 분류
    splits = {'train': [], 'val': [], 'test': []}
    for img_info in all_images:
        for split_name, frame_set in frame_splits.items():
            if img_info['frame_name'] in frame_set:
                splits[split_name].append(img_info)
                break
    
    # 각 split별 처리
    stats = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}
    
    for split_name, images in splits.items():
        logger.info(f"{split_name} 분할 처리 중: {len(images)}개 이미지")
        
        for img_info in tqdm(images, desc=f"Processing {split_name}"):
            try:
                # 이미지 로드
                image = cv2.imread(img_info['path'])
                if image is None:
                    logger.error(f"이미지 로드 실패: {img_info['path']}")
                    stats['errors'] += 1
                    continue
                
                # 타겟 크기로 리사이즈
                original_h, original_w = image.shape[:2]
                image_resized = cv2.resize(image, target_size)
                
                # 출력 파일명 생성 (고유한 이름 생성)
                base_name = f"{img_info['frame_name']}_cls{img_info['class_id']}_{os.path.splitext(img_info['filename'])[0]}"
                image_filename = f"{base_name}.jpg"
                label_filename = f"{base_name}.txt"
                
                # 이미지 저장
                image_output_path = os.path.join(output_dir, split_name, 'images', image_filename)
                cv2.imwrite(image_output_path, image_resized)
                
                # 라벨 생성 - 우선순위: Mask > Box Annotation > 전체 이미지
                label_lines = []
                polygon = None
                
                # 1. 먼저 Mask 정보 시도
                if img_info['has_mask']:
                    mask_file_path = os.path.join(mask_dir, f"{img_info['frame_name']}_masks.npy")
                    object_mask = get_object_mask_from_masks_file(mask_file_path, img_info['object_index'])
                    
                    if object_mask is not None:
                        polygon = mask_to_polygon(object_mask)
                        if polygon:
                            logger.debug(f"Mask에서 polygon 추출 성공: {img_info['filename']}")
                
                # 2. Mask가 없거나 실패하면 Box Annotation 시도
                if polygon is None and img_info['has_annotation']:
                    box_data = load_box_annotation(box_dir, img_info['frame_name'])
                    
                    if box_data:
                        # boxes와 class_ids를 함께 사용
                        if 'boxes' in box_data and 'class_ids' in box_data:
                            boxes = box_data['boxes']
                            class_ids = box_data['class_ids']
                            
                            # 객체 인덱스에 해당하는 box 찾기
                            if img_info['object_index'] < len(boxes) and img_info['object_index'] < len(class_ids):
                                box = boxes[img_info['object_index']]
                                box_class_id = class_ids[img_info['object_index']]
                                
                                # 클래스가 일치하는지 확인
                                if box_class_id == img_info['class_id']:
                                    # Box를 polygon으로 변환 (x1,y1,x2,y2 -> polygon)
                                    x1, y1, x2, y2 = box
                                    # 정규화된 좌표로 변환
                                    norm_x1 = max(0.0, min(1.0, x1 / original_w))
                                    norm_y1 = max(0.0, min(1.0, y1 / original_h))
                                    norm_x2 = max(0.0, min(1.0, x2 / original_w))
                                    norm_y2 = max(0.0, min(1.0, y2 / original_h))
                                    
                                    # Rectangle을 polygon으로 변환 (시계방향)
                                    polygon = [norm_x1, norm_y1, norm_x2, norm_y1, norm_x2, norm_y2, norm_x1, norm_y2]
                                    logger.debug(f"Box annotation에서 polygon 추출 성공: {img_info['filename']}")
                        
                        # 이전 방식의 shapes도 시도 (fallback)
                        elif polygon is None and 'shapes' in box_data:
                            for shape in box_data['shapes']:
                                shape_class = shape.get('label', '')
                                
                                # 클래스 매핑
                                shape_class_id = None
                                if shape_class.isdigit():
                                    shape_class_id = int(shape_class)
                                elif shape_class.startswith('class_') and shape_class[6:].isdigit():
                                    shape_class_id = int(shape_class[6:])
                                elif shape_class.startswith('Class_') and shape_class[6:].isdigit():
                                    shape_class_id = int(shape_class[6:])
                                
                                if shape_class_id == img_info['class_id']:
                                    points = shape.get('points', [])
                                    if len(points) >= 3:
                                        # 정규화된 polygon 좌표 생성
                                        polygon = []
                                        for point in points:
                                            x, y = point
                                            norm_x = max(0.0, min(1.0, x / original_w))
                                            norm_y = max(0.0, min(1.0, y / original_h))
                                            polygon.extend([norm_x, norm_y])
                                        
                                        if len(polygon) >= 6:
                                            logger.debug(f"Shape annotation에서 polygon 추출 성공: {img_info['filename']}")
                                            break
                
                # 3. 최후의 수단: 전체 이미지 bbox
                if polygon is None:
                    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
                    logger.debug(f"전체 이미지를 bbox로 처리: {img_info['filename']}")
                
                # YOLO 형식 라벨 생성
                if polygon and len(polygon) >= 6:
                    line_parts = [str(img_info['class_id'])] + [f"{coord:.6f}" for coord in polygon]
                    label_lines.append(' '.join(line_parts))
                
                # 라벨 파일 저장
                label_output_path = os.path.join(output_dir, split_name, 'labels', label_filename)
                with open(label_output_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                
                stats[split_name] += 1
                
            except Exception as e:
                logger.error(f"이미지 처리 실패 {img_info['filename']}: {e}")
                stats['errors'] += 1
    
    # dataset.yaml 생성
    yaml_content = f"""# YOLO Segmentation Dataset Configuration
path: {os.path.abspath(output_dir)}

train: train/images
val: val/images  
test: test/images

# Classes
nc: {len(target_classes)}  # number of classes
names: {target_classes}  # class names

# Dataset info
created_from: ground_truth_{os.path.basename(category_path)}
method: folder_based_classification_with_masks
total_images: {sum(stats[k] for k in ['train', 'val', 'test'])}
unique_frames: {len(processed_frames)}
target_classes_only: true
filtered_unknown: {skipped_unknown}
with_annotations: {sum(1 for img in all_images if img['has_annotation'])}
without_annotations: {sum(1 for img in all_images if not img['has_annotation'])}
with_masks: {sum(1 for img in all_images if img['has_mask'])}
without_masks: {sum(1 for img in all_images if not img['has_mask'])}
splits:
  train: {stats['train']}
  val: {stats['val']}
  test: {stats['test']}
errors: {stats['errors']}
"""
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # 결과 출력
    logger.info("=" * 50)
    logger.info("Ground Truth 기반 YOLO Segmentation 데이터셋 생성 완료!")
    logger.info(f"출력 디렉토리: {os.path.abspath(output_dir)}")
    logger.info(f"최종 처리된 이미지: {len(all_images)}개")
    logger.info(f"고유 프레임: {len(processed_frames)}개")
    logger.info(f"스킵된 Unknown: {skipped_unknown}개")
    logger.info(f"Annotation 없음: {skipped_no_annotation}개")
    logger.info(f"Train: {stats['train']} 이미지")
    logger.info(f"Val: {stats['val']} 이미지") 
    logger.info(f"Test: {stats['test']} 이미지")
    logger.info(f"총 성공: {sum(stats[k] for k in ['train', 'val', 'test'])} 이미지")
    logger.info(f"오류: {stats['errors']} 이미지")
    logger.info("=" * 50)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Ground Truth 기반 YOLO Segmentation 데이터셋 생성')
    parser.add_argument('--category', default='test_category', 
                        help='카테고리 이름 (기본: test_category)')
    parser.add_argument('--target-classes', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='대상 클래스 ID 리스트 (기본: 0 1 2 3)')
    parser.add_argument('--output', default='data/test_category/9.yolo-dataset-gt', 
                        help='출력 디렉토리 (기본: data/test_category/9.yolo-dataset-gt)')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 640],
                        help='이미지 크기 [width height] (기본: 640 640)')
    parser.add_argument('--split', nargs=3, type=float, default=[0.7, 0.2, 0.1],
                        help='데이터셋 분할 비율 [train val test] (기본: 0.7 0.2 0.1)')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DEBUG 모드 활성화: 상세 로그가 출력됩니다.")
    
    # 경로 설정
    category_path = f"data/{args.category}"
    if not os.path.exists(category_path):
        logger.error(f"카테고리 경로가 존재하지 않습니다: {category_path}")
        return
    
    # 데이터셋 생성
    stats = create_yolo_from_ground_truth(
        category_path=category_path,
        target_classes=args.target_classes,
        output_dir=args.output,
        target_size=tuple(args.size),
        split_ratios=tuple(args.split)
    )

if __name__ == "__main__":
    main() 