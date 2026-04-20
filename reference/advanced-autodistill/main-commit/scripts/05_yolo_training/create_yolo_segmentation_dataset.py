#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Segmentation 데이터셋 생성 (원본 재구성 방식)

7.results/ground_truth 내 파일 목록을 정답으로 간주하여,
1.images의 원본 이미지와 4.mask의 NPY 마스크를 재구성하여
YOLO segmentation 형식의 데이터셋을 생성합니다.

특징:
1. 'ground_truth'의 파일 목록을 기반으로 처리할 객체 선정
2. 원본 전체 이미지(1.images)에서 객체 영역을 동적으로 크롭
3. 크롭된 이미지를 사용자 지정 크기(기본 640x640)로 리사이즈
4. 크롭된 마스크에서 폴리곤 좌표를 추출하여 정확한 라벨 생성
5. 7:2:1 비율로 train:val:test 분할
"""

import os
import cv2
import json
import numpy as np
import logging
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Optional
import re
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_frame_and_obj_info(filename):
    """
    preprocessed 파일명에서 프레임명과 객체 정보 추출
    예: 'H_1_1_frame_0117_obj0_cls0_unknown_class_0.png' 
    -> ('H_1_1_frame_0117', 0, 0)
    """
    # 정규표현식으로 파싱
    pattern = r'([^_]+_[^_]+_[^_]+_frame_\d+)_obj(\d+)_cls(\d+)_'
    match = re.match(pattern, filename)
    
    if match:
        frame_name = match.group(1)
        obj_index = int(match.group(2))
        class_id = int(match.group(3))
        return frame_name, obj_index, class_id
    return None, None, None

def load_mask_data(mask_dir, frame_name):
    """
    NPY 마스크 파일 로드
    """
    # 복수형과 단수형 모두 시도
    for suffix in ['_masks.npy', '_mask.npy']:
        mask_path = os.path.join(mask_dir, f"{frame_name}{suffix}")
        if os.path.exists(mask_path):
            try:
                masks = np.load(mask_path)
                logger.debug(f"마스크 로드 성공: {mask_path}, shape: {masks.shape}")
                return masks
            except Exception as e:
                logger.error(f"마스크 로드 실패 {mask_path}: {e}")
    return None

def mask_to_polygon(mask):
    """
    Binary mask를 polygon 좌표로 변환
    """
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # 가장 큰 컨투어 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 윤곽선 단순화를 최소화하여 거의 모든 점 유지
    # 노이즈 제거를 위한 최소한의 단순화만 적용
    epsilon = 0.0001 * cv2.arcLength(largest_contour, True)  # 훨씬 더 작은 값으로 변경
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 포인트 수 제한 코드 제거 - 모든 포인트 유지
    
    # polygon 좌표 정규화 (0-1 범위)
    height, width = mask.shape
    polygon = []
    for point in approx:
        x, y = point[0]
        polygon.extend([x / width, y / height])
    
    return polygon

def create_yolo_segmentation_dataset(
    category_path: str,
    target_classes: List[int] = [0, 1, 2, 3],
    output_dir: str = "yolo_dataset",
    target_size: Tuple[int, int] = (640, 640),
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    mask_background: bool = False
):
    """
    YOLO segmentation 데이터셋 생성
    """
    logger.info(f"YOLO Segmentation 데이터셋 생성 시작")
    logger.info(f"카테고리: {category_path}")
    logger.info(f"대상 클래스: {target_classes}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"대상 크기: {target_size}")
    logger.info(f"분할 비율: train:{split_ratios[0]}, val:{split_ratios[1]}, test:{split_ratios[2]}")
    logger.info(f"배경 마스킹 활성화: {mask_background}")
    
    # 경로 설정
    ground_truth_dir = os.path.join(category_path, "7.results", "ground_truth")
    original_images_dir = os.path.join(category_path, "1.images")
    mask_dir = os.path.join(category_path, "4.mask")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # 모든 preprocessed 이미지 수집
    all_images = []
    processed_frames = set()  # 중복 프레임 체크용
    
    for class_id in target_classes:
        class_dir = os.path.join(ground_truth_dir, f"Class_{class_id}")
        if not os.path.exists(class_dir):
            logger.warning(f"클래스 디렉토리 없음: {class_dir}")
            continue
        
        for filename in os.listdir(class_dir):
            if not filename.endswith('.png'):
                continue
            
            frame_name, obj_index, file_class_id = extract_frame_and_obj_info(filename)
            if frame_name is None or file_class_id != class_id:
                continue
            
            image_path = os.path.join(class_dir, filename)
            all_images.append({
                'path': image_path,
                'filename': filename,
                'frame_name': frame_name,
                'obj_index': obj_index,
                'class_id': class_id
            })
            processed_frames.add(frame_name)
    
    logger.info(f"총 {len(all_images)}개 이미지 발견 ({len(processed_frames)}개 고유 프레임)")
    
    # 데이터셋 분할
    random.shuffle(all_images)
    train_split = int(len(all_images) * split_ratios[0])
    val_split = int(len(all_images) * (split_ratios[0] + split_ratios[1]))
    
    splits = {
        'train': all_images[:train_split],
        'val': all_images[train_split:val_split],
        'test': all_images[val_split:]
    }
    
    # 각 split별 처리
    stats = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}
    frame_mask_cache = {}  # 프레임별 마스크 캐시
    
    for split_name, images in splits.items():
        logger.info(f"{split_name} 분할 처리 중: {len(images)}개 이미지")
        
        for img_info in tqdm(images, desc=f"Processing {split_name}"):
            try:
                # 1. 원본 전체 프레임 이미지 로드
                frame_name = img_info['frame_name']
                original_image_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    path_candidate = os.path.join(original_images_dir, f"{frame_name}{ext}")
                    if os.path.exists(path_candidate):
                        original_image_path = path_candidate
                        break
                
                if not original_image_path:
                    logger.warning(f"원본 이미지를 찾을 수 없습니다: {frame_name} in {original_images_dir}")
                    stats['errors'] += 1
                    continue

                original_image = cv2.imread(original_image_path)
                if original_image is None:
                    logger.error(f"원본 이미지 로드 실패: {original_image_path}")
                    stats['errors'] += 1
                    continue
                
                # 2. 마스크 데이터 로드 (캐시 활용)
                if frame_name not in frame_mask_cache:
                    masks = load_mask_data(mask_dir, frame_name)
                    frame_mask_cache[frame_name] = masks
                else:
                    masks = frame_mask_cache[frame_name]
                    
                if masks is None or img_info['obj_index'] >= len(masks):
                    logger.warning(f"마스크 데이터를 찾을 수 없거나 obj_index가 범위를 벗어남: {img_info['filename']}")
                    stats['errors'] += 1
                    continue

                # 3. 객체 마스크를 사용하여 바운딩 박스 계산
                obj_mask = masks[img_info['obj_index']]
                
                # 마스크에서 컨투어 및 바운딩 박스 찾기
                binary_mask = obj_mask.astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    logger.warning(f"마스크에서 컨투어를 찾을 수 없음: {img_info['filename']}")
                    stats['errors'] += 1
                    continue
                
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

                # 4. 크롭할 이미지 결정 (배경 마스킹 여부)
                if mask_background:
                    # 마스크를 사용해 원본 이미지에서 객체 영역만 남기고 배경은 검게 처리
                    image_to_crop = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
                else:
                    # 원본 이미지를 그대로 사용
                    image_to_crop = original_image
                
                # 5. 이미지와 마스크를 타이트하게 크롭
                cropped_image = image_to_crop[y:y+h, x:x+w]
                cropped_mask = obj_mask[y:y+h, x:x+w] # For polygon extraction

                if cropped_image.size == 0 or cropped_mask.size == 0:
                    logger.warning(f"크롭 결과가 비어있음: {img_info['filename']}")
                    stats['errors'] += 1
                    continue

                # 6. 크롭된 이미지를 타겟 크기로 리사이즈
                image_resized = cv2.resize(cropped_image, target_size)
                
                # 출력 파일명 생성 (기존과 동일)
                base_name = f"{img_info['frame_name']}_obj{img_info['obj_index']}_cls{img_info['class_id']}"
                image_filename = f"{base_name}.jpg" # jpg로 통일
                label_filename = f"{base_name}.txt"
                
                # 7. 리사이즈된 이미지 저장
                image_output_path = os.path.join(output_dir, split_name, 'images', image_filename)
                cv2.imwrite(image_output_path, image_resized)
                
                # 8. 크롭된 마스크에서 폴리곤 추출 및 라벨 생성
                label_lines = []
                polygon = mask_to_polygon(cropped_mask) # 크롭된 마스크 전달
                
                if polygon and len(polygon) >= 6:
                    # YOLO 형식: class_id x1 y1 x2 y2 x3 y3 ...
                    line_parts = [str(img_info['class_id'])] + [f"{coord:.6f}" for coord in polygon]
                    label_lines.append(' '.join(line_parts))
                
                # 9. 라벨 파일 저장
                label_output_path = os.path.join(output_dir, split_name, 'labels', label_filename)
                with open(label_output_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                
                stats[split_name] += 1
                
            except Exception as e:
                logger.error(f"이미지 처리 중 예외 발생 {img_info['filename']}: {e}", exc_info=True)
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
created_from: reconstructed_from_ground_truth
total_images: {sum(stats[k] for k in ['train', 'val', 'test'])}
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
    logger.info("YOLO Segmentation 데이터셋 생성 완료!")
    logger.info(f"출력 디렉토리: {os.path.abspath(output_dir)}")
    logger.info(f"Train: {stats['train']} 이미지")
    logger.info(f"Val: {stats['val']} 이미지") 
    logger.info(f"Test: {stats['test']} 이미지")
    logger.info(f"총 성공: {sum(stats[k] for k in ['train', 'val', 'test'])} 이미지")
    logger.info(f"오류: {stats['errors']} 이미지")
    logger.info("=" * 50)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='YOLO Segmentation 데이터셋 생성')
    parser.add_argument('--category', default='test_category', 
                        help='카테고리 이름 (기본: test_category)')
    parser.add_argument('--target-classes', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='대상 클래스 ID 리스트 (기본: 0 1 2 3)')
    parser.add_argument('--output', default='data/test_category/8.refine-dataset', 
                        help='출력 디렉토리 (기본: data/test_category/8.refine-dataset)')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 640],
                        help='이미지 크기 [width height] (기본: 640 640)')
    parser.add_argument('--split', nargs=3, type=float, default=[0.7, 0.2, 0.1],
                        help='데이터셋 분할 비율 [train val test] (기본: 0.7 0.2 0.1)')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    parser.add_argument('--mask-background', action='store_true',
                        help='객체 마스크 외부의 배경을 검은색으로 처리합니다.')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 경로 설정
    # data/test_category 경로가 존재하는지 확인
    base_data_path = "data"
    category_path = os.path.join(base_data_path, args.category)
    if not os.path.exists(category_path):
        logger.error(f"카테고리 경로가 존재하지 않습니다: {category_path}")
        return

    # 출력 디렉토리 경로 설정
    output_path = args.output
    if not os.path.isabs(output_path):
         output_path = os.path.join(base_data_path, args.category, os.path.basename(args.output))

    # 데이터셋 생성
    stats = create_yolo_segmentation_dataset(
        category_path=category_path,
        target_classes=args.target_classes,
        output_dir=output_path,
        target_size=tuple(args.size),
        split_ratios=tuple(args.split),
        mask_background=args.mask_background
    )

if __name__ == "__main__":
    main() 