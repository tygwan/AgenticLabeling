#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth 기반 YOLO 데이터셋 생성

이 스크립트는 ground truth 폴더의 Class 0,1,2,3 이미지들과 
box annotation 정보를 동기화하여 YOLO 데이터셋을 생성합니다.
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_original_frame_name(image_filename):
    """
    ground truth 이미지 파일명에서 원본 프레임 이름 추출
    예: 'G_1_2_frame_0073_obj0_cls1_unknown_class_1.png' -> 'G_1_2_frame_0073'
    """
    parts = image_filename.split('_')
    if len(parts) >= 4:
        # '_obj'가 나오기 전까지의 부분을 합침
        frame_parts = []
        for part in parts:
            if part.startswith('obj'):
                break
            frame_parts.append(part)
        return '_'.join(frame_parts)
    return os.path.splitext(image_filename)[0]

def load_box_annotations(box_dir):
    """
    Box annotation 디렉토리에서 모든 박스 정보 로드
    
    Args:
        box_dir: box annotation 디렉토리 경로
        
    Returns:
        dict: {frame_name: box_data} 형태의 딕셔너리
    """
    box_data = {}
    
    if not os.path.exists(box_dir):
        logger.warning(f"Box annotation 디렉토리를 찾을 수 없음: {box_dir}")
        return box_data
    
    for file in os.listdir(box_dir):
        if file.endswith('_box.json'):
            frame_name = file.replace('_box.json', '')
            box_file_path = os.path.join(box_dir, file)
            
            try:
                with open(box_file_path, 'r') as f:
                    data = json.load(f)
                    box_data[frame_name] = data
            except Exception as e:
                logger.warning(f"Box annotation 파일 로드 실패: {file} - {e}")
    
    logger.info(f"Box annotation 로드 완료: {len(box_data)}개 프레임")
    return box_data

def create_yolo_dataset_from_ground_truth(category_path, output_dir=None, target_classes=None, 
                                        target_size=(640, 640), val_split=0.2, test_split=0.1):
    """
    Ground truth 폴더에서 YOLO 데이터셋 생성
    
    Args:
        category_path: 카테고리 폴더 경로 (data/test_category)
        output_dir: 출력 디렉토리 (기본값: category_path/yolo_dataset)
        target_classes: 대상 클래스 리스트 (기본값: ['Class_0', 'Class_1', 'Class_2', 'Class_3'])
        target_size: 대상 이미지 크기 (너비, 높이)
        val_split: 검증 데이터셋 비율 (0~1)
        test_split: 테스트 데이터셋 비율 (0~1)
        
    Returns:
        생성된 데이터셋 정보 딕셔너리
    """
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join(category_path, "yolo_dataset")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 대상 클래스 설정
    if target_classes is None:
        target_classes = ['Class_0', 'Class_1', 'Class_2', 'Class_3']
    
    # 클래스 매핑 (YOLO는 0부터 시작)
    class_mapping = {i: cls for i, cls in enumerate(target_classes)}
    class_name_to_id = {cls: i for i, cls in enumerate(target_classes)}
    
    logger.info(f"클래스 매핑: {class_mapping}")
    
    # 훈련/검증/테스트 디렉토리 설정
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    
    # Ground truth 폴더 경로
    gt_dir = os.path.join(category_path, "7.results", "ground_truth")
    box_dir = os.path.join(category_path, "3.box")
    original_images_dir = os.path.join(category_path, "1.images")
    
    # Box annotation 로드
    box_data = load_box_annotations(box_dir)
    
    # Ground truth 이미지 수집
    all_images = []
    
    for class_name in target_classes:
        class_dir = os.path.join(gt_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"클래스 디렉토리를 찾을 수 없음: {class_dir}")
            continue
        
        class_id = class_name_to_id[class_name]
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                
                # 원본 프레임 이름 추출
                frame_name = extract_original_frame_name(img_file)
                
                # 원본 이미지 경로 찾기
                original_img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate_path = os.path.join(original_images_dir, f"{frame_name}{ext}")
                    if os.path.exists(candidate_path):
                        original_img_path = candidate_path
                        break
                
                if original_img_path is None:
                    logger.warning(f"원본 이미지를 찾을 수 없음: {frame_name}")
                    continue
                
                # Box annotation 확인
                if frame_name not in box_data:
                    logger.warning(f"Box annotation을 찾을 수 없음: {frame_name}")
                    continue
                
                all_images.append({
                    'gt_image_path': img_path,
                    'original_image_path': original_img_path,
                    'frame_name': frame_name,
                    'class_id': class_id,
                    'class_name': class_name,
                    'image_filename': img_file
                })
    
    logger.info(f"총 {len(all_images)}개의 ground truth 이미지 발견")
    
    if len(all_images) == 0:
        logger.error("처리할 이미지가 없습니다.")
        return {"error": "No images to process"}
    
    # 이미지 리스트 섞기
    random.seed(42)  # 재현성을 위한 시드 설정
    random.shuffle(all_images)
    
    # 데이터셋 분할
    total_count = len(all_images)
    test_count = int(total_count * test_split)
    val_count = int(total_count * val_split)
    train_count = total_count - test_count - val_count
    
    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]
    
    logger.info(f"데이터셋 분할: 훈련 {len(train_images)}개, 검증 {len(val_images)}개, 테스트 {len(test_images)}개")
    
    # 처리 통계
    stats = {
        "processed": 0,
        "skipped": 0,
        "no_boxes": 0,
        "total_boxes": 0
    }
    
    # 각 분할에 대해 처리
    splits = [
        ("train", train_images, train_dir),
        ("val", val_images, val_dir),
        ("test", test_images, test_dir)
    ]
    
    for split_name, image_list, split_dir in splits:
        logger.info(f"{split_name} 세트 처리 중... ({len(image_list)}개 이미지)")
        
        for img_info in tqdm(image_list, desc=f"{split_name} 처리"):
            try:
                frame_name = img_info['frame_name']
                original_img_path = img_info['original_image_path']
                class_id = img_info['class_id']
                
                # 원본 이미지 로드
                img = cv2.imread(original_img_path)
                if img is None:
                    logger.warning(f"이미지 로드 실패: {original_img_path}")
                    stats["skipped"] += 1
                    continue
                
                original_h, original_w = img.shape[:2]
                
                # 이미지 리사이징
                resized_img = cv2.resize(img, target_size)
                
                # 이미지 저장
                img_output_path = os.path.join(split_dir, "images", f"{frame_name}.jpg")
                cv2.imwrite(img_output_path, resized_img)
                
                # Box annotation 처리
                box_info = box_data[frame_name]
                boxes = box_info.get('boxes', [])
                class_ids = box_info.get('class_ids', [])
                
                if not boxes:
                    stats["no_boxes"] += 1
                    # 박스가 없는 이미지는 빈 레이블 파일 생성
                    label_output_path = os.path.join(split_dir, "labels", f"{frame_name}.txt")
                    with open(label_output_path, 'w') as f:
                        pass  # 빈 파일 생성
                    stats["processed"] += 1
                    continue
                
                # YOLO 형식으로 레이블 생성
                label_output_path = os.path.join(split_dir, "labels", f"{frame_name}.txt")
                valid_boxes = 0
                
                with open(label_output_path, 'w') as f:
                    for i, box in enumerate(boxes):
                        if i >= len(class_ids):
                            continue
                        
                        box_class_id = class_ids[i] - 1  # class_ids는 1부터 시작하므로 0부터 시작하도록 조정
                        
                        # 대상 클래스만 포함 (0-3)
                        if 0 <= box_class_id < len(target_classes):
                            x1, y1, x2, y2 = box
                            
                            # YOLO 형식으로 변환 (정규화된 중심점 좌표와 크기)
                            x_center = ((x1 + x2) / 2) / original_w
                            y_center = ((y1 + y2) / 2) / original_h
                            width = abs(x2 - x1) / original_w
                            height = abs(y2 - y1) / original_h
                            
                            # 범위 제한 (0-1)
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            # 너무 작은 박스 제외
                            if width > 0.01 and height > 0.01:
                                f.write(f"{box_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                valid_boxes += 1
                
                stats["processed"] += 1
                stats["total_boxes"] += valid_boxes
                
            except Exception as e:
                logger.error(f"이미지 {img_info['frame_name']} 처리 중 오류: {e}")
                stats["skipped"] += 1
                import traceback
                logger.error(traceback.format_exc())
    
    # 데이터셋 정보 생성
    dataset_info = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "test_images": len(test_images),
        "target_size": list(target_size),
        "classes": class_mapping,
        "class_names": target_classes,
        "stats": stats
    }
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    # YOLO 데이터셋 YAML 파일 생성
    create_yolo_yaml(output_dir, target_classes)
    
    logger.info(f"YOLO 데이터셋 생성 완료:")
    logger.info(f"  - 훈련 이미지: {dataset_info['train_images']}개")
    logger.info(f"  - 검증 이미지: {dataset_info['val_images']}개") 
    logger.info(f"  - 테스트 이미지: {dataset_info['test_images']}개")
    logger.info(f"  - 총 박스: {stats['total_boxes']}개")
    logger.info(f"  - 처리된 이미지: {stats['processed']}개")
    logger.info(f"  - 건너뛴 이미지: {stats['skipped']}개")
    logger.info(f"  - 박스 없는 이미지: {stats['no_boxes']}개")
    
    return dataset_info

def create_yolo_yaml(dataset_dir, class_names):
    """
    YOLO 학습용 data.yaml 파일 생성
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        class_names: 클래스 이름 리스트
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO Dataset Configuration for Ground Truth Classes\n")
        f.write(f"path: {os.path.abspath(dataset_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"# Classes\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names:\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"  {i}: {class_name}\n")
    
    logger.info(f"YOLO YAML 파일 생성됨: {yaml_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ground Truth 기반 YOLO 데이터셋 생성")
    parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름")
    parser.add_argument("--output", type=str, help="출력 디렉토리 (기본값: data/{category}/yolo_dataset)")
    parser.add_argument("--classes", type=str, nargs='+', 
                       default=['Class_0', 'Class_1', 'Class_2', 'Class_3'],
                       help="대상 클래스 리스트")
    parser.add_argument("--target-size", type=str, default="640,640", help="대상 이미지 크기 (너비,높이)")
    parser.add_argument("--val-split", type=float, default=0.2, help="검증 데이터셋 비율 (0~1)")
    parser.add_argument("--test-split", type=float, default=0.1, help="테스트 데이터셋 비율 (0~1)")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅 활성화")
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 카테고리 경로
    category_path = os.path.join("data", args.category)
    
    if not os.path.exists(category_path):
        logger.error(f"카테고리 경로를 찾을 수 없음: {category_path}")
        return
    
    # 대상 이미지 크기 파싱
    try:
        width, height = map(int, args.target_size.split(','))
        target_size = (width, height)
    except ValueError:
        logger.error("잘못된 target-size 형식. '너비,높이' 형식으로 입력하세요 (예: 640,640)")
        return
    
    # 출력 디렉토리 설정
    if args.output is None:
        output_dir = os.path.join(category_path, "yolo_dataset")
    else:
        output_dir = args.output
    
    # YOLO 데이터셋 생성
    result = create_yolo_dataset_from_ground_truth(
        category_path=category_path,
        output_dir=output_dir,
        target_classes=args.classes,
        target_size=target_size,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    if "error" in result:
        logger.error(f"데이터셋 생성 실패: {result['error']}")
    else:
        logger.info("YOLO 데이터셋 생성이 완료되었습니다!")
        logger.info(f"출력 경로: {output_dir}")
        logger.info("YOLOv8로 학습을 시작하려면 다음 명령을 사용하세요:")
        logger.info(f"yolo detect train data='{os.path.join(output_dir, 'data.yaml')}' model=yolov8n.pt epochs=100")

if __name__ == "__main__":
    main() 