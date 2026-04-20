#!/usr/bin/env python3
"""
전처리 유틸리티 모듈

이미지 전처리를 위한 다양한 함수들을 제공합니다.
"""

import os
import cv2
import json
import numpy as np
import datetime
from pathlib import Path

def load_coords_data(coords_file):
    """
    좌표 JSON 파일 로드
    
    Args:
        coords_file: 좌표 JSON 파일 경로
        
    Returns:
        dict: 좌표 데이터
    """
    try:
        with open(coords_file, 'r') as f:
            coords_data = json.load(f)
        return coords_data
    except Exception as e:
        print(f"[ERROR] 좌표 파일 로드 오류: {e}")
        return None

def create_mask_from_contours(contours, image_shape):
    """
    윤곽선으로부터 마스크 생성
    
    Args:
        contours: 윤곽선 목록
        image_shape: 이미지 크기 (높이, 너비)
        
    Returns:
        numpy.ndarray: 마스크 배열
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
    for contour in contours:
        # 윤곽선 포인트를 정수로 변환
        points = np.array(contour, dtype=np.int32)
        
        # 다각형으로 마스크 채우기
        cv2.fillPoly(mask, [points], 255)
    
    return mask

def apply_mask_to_image(image, mask):
    """
    이미지에 마스크 적용
    
    Args:
        image: 원본 이미지
        mask: 마스크 배열
        
    Returns:
        numpy.ndarray: 마스크가 적용된 이미지
    """
    # 마스크 차원 확장 (3채널)
    mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
    
    # 마스크 적용
    return (image * mask_3ch).astype(np.uint8)

def crop_object_from_box(image, box, padding=0):
    """
    경계 상자로 객체 크롭
    
    Args:
        image: 원본 이미지
        box: 경계 상자 [x, y, w, h]
        padding: 패딩 픽셀 수
        
    Returns:
        numpy.ndarray: 크롭된 이미지
    """
    x, y, w, h = box
    
    # 패딩 적용
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + padding * 2)
    h = min(image.shape[0] - y, h + padding * 2)
    
    # 이미지 크롭
    return image[y:y+h, x:x+w]

def process_image_with_coords(image_path, coords_file, output_dir, crop=True, apply_mask=True, resize=None, prefix=""):
    """
    좌표 정보를 이용하여 이미지 처리
    
    Args:
        image_path: 이미지 파일 경로
        coords_file: 좌표 JSON 파일 경로
        output_dir: 출력 디렉토리
        crop: 객체 크롭 여부
        apply_mask: 마스크 적용 여부
        resize: 크기 조정 (너비, 높이) 튜플 또는 None
        prefix: 출력 파일 이름 접두사
        
    Returns:
        dict: 처리된 파일 경로 목록
    """
    # 이미지 로드
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 이미지 로드 실패: {image_path}")
            return {"files": [], "metadata": {}}
    except Exception as e:
        print(f"[ERROR] 이미지 로드 중 오류: {e}")
        return {"files": [], "metadata": {}}
    
    # 좌표 데이터 로드
    coords_data = load_coords_data(coords_file)
    if not coords_data:
        return {"files": [], "metadata": {}}
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 처리된 객체 메타데이터
    metadata = {
        "source_image": image_path,
        "coords_file": coords_file,
        "processing_options": {
            "crop": crop,
            "apply_mask": apply_mask,
            "resize": resize
        },
        "processed_objects": []
    }
    
    # 각 마스크 처리
    processed_files = []
    
    # 기본 파일 이름 생성
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if prefix:
        base_name = f"{prefix}_{base_name}"
    
    for mask_info in coords_data.get("masks", []):
        # 마스크 메타데이터
        mask_metadata = {
            "index": mask_info.get("index", 0),
            "class_name": mask_info.get("class_name", "unknown"),
            "class_id": mask_info.get("class_id", 0)
        }
        
        # 경계 상자 및 윤곽선 가져오기
        bounding_boxes = mask_info.get("bounding_boxes", [])
        contours = mask_info.get("contours", [])
        
        if not bounding_boxes or not contours:
            continue
        
        # 첫 번째 경계 상자 사용
        box = bounding_boxes[0]
        
        # 마스크 생성
        mask = None
        if apply_mask:
            mask = create_mask_from_contours(contours, image.shape[:2])
        
        # 객체 처리
        for i, box in enumerate(bounding_boxes):
            # 출력 파일 이름
            obj_index = mask_info.get("index", 0)
            class_name = mask_info.get("class_name", "unknown").replace(" ", "_").lower()
            output_name = f"{base_name}_obj{obj_index}_box{i}_{class_name}.jpg"
            output_path = os.path.join(output_dir, output_name)
            
            # 객체 크롭
            obj_image = None
            if crop:
                obj_image = crop_object_from_box(image, box, padding=10)
                
                # 마스크 적용
                if apply_mask and mask is not None:
                    # 크롭된 영역에 해당하는 마스크 추출
                    x, y, w, h = box
                    obj_mask = mask[y:y+h, x:x+w]
                    
                    # 크롭된 이미지에 마스크 적용
                    if obj_mask.shape[:2] == obj_image.shape[:2]:
                        obj_image = apply_mask_to_image(obj_image, obj_mask)
            else:
                # 전체 이미지 사용
                obj_image = image.copy()
                
                # 마스크 적용
                if apply_mask and mask is not None:
                    obj_image = apply_mask_to_image(obj_image, mask)
            
            # 크기 조정
            if resize and obj_image is not None:
                obj_image = cv2.resize(obj_image, resize)
            
            # 이미지 저장
            if obj_image is not None:
                cv2.imwrite(output_path, obj_image)
                processed_files.append(output_path)
                
                # 메타데이터 추가
                obj_metadata = mask_metadata.copy()
                obj_metadata.update({
                    "box_index": i,
                    "box": box,
                    "output_file": output_path
                })
                metadata["processed_objects"].append(obj_metadata)
    
    return {
        "files": processed_files,
        "metadata": metadata
    }

def batch_process_with_coords(image_paths, coords_dir, output_dir, crop=True, apply_mask=True, resize=None):
    """
    여러 이미지를 좌표 정보를 이용하여 배치 처리
    
    Args:
        image_paths: 이미지 경로 목록
        coords_dir: 좌표 파일 디렉토리
        output_dir: 출력 디렉토리
        crop: 객체 크롭 여부
        apply_mask: 마스크 적용 여부
        resize: 크기 조정 (너비, 높이) 튜플 또는 None
        
    Returns:
        dict: 처리 결과
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 저장
    result = {}
    skipped_images = []
    error_images = []
    total_objects = 0
    all_metadata = []
    
    # 각 이미지 처리
    for image_path in image_paths:
        # 이미지 기본 이름
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 좌표 파일 경로
        coords_file = os.path.join(coords_dir, f"{base_name}_coords.json")
        
        # 좌표 파일이 존재하는지 확인
        if not os.path.exists(coords_file):
            print(f"[WARNING] 좌표 파일이 존재하지 않습니다: {coords_file} - 이미지를 건너뜁니다.")
            skipped_images.append(image_path)
            continue
        
        # 이미지 처리
        try:
            process_result = process_image_with_coords(
                image_path,
                coords_file,
                output_dir,
                crop=crop,
                apply_mask=apply_mask,
                resize=resize
            )
            
            if process_result["files"]:
                result[image_path] = process_result["files"]
                all_metadata.append(process_result["metadata"])
                total_objects += len(process_result["files"])
            else:
                # 파일이 처리되지 않았지만 오류는 아님 (예: 객체가 없는 이미지)
                skipped_images.append(image_path)
        except Exception as e:
            print(f"[ERROR] 이미지 처리 중 오류 발생: {image_path} - {e}")
            error_images.append({"path": image_path, "error": str(e)})
    
    # 전체 메타데이터 요약 파일 저장
    summary_metadata = {
        "processing_options": {
            "crop": crop,
            "apply_mask": apply_mask,
            "resize": resize
        },
        "total_images": len(result),
        "total_objects": total_objects,
        "skipped_images": len(skipped_images),
        "error_images": len(error_images),
        "processed_at": datetime.datetime.now().isoformat(),
        "image_summaries": {}
    }
    
    for metadata in all_metadata:
        source_image = metadata.get("source_image", "unknown")
        source_basename = os.path.basename(source_image)
        summary_metadata["image_summaries"][source_basename] = {
            "object_count": len(metadata.get("processed_objects", [])),
            "classes": list(set(obj.get("class_name") for obj in metadata.get("processed_objects", [])))
        }
    
    # 에러 및 스킵된 이미지 목록 추가
    if skipped_images:
        summary_metadata["skipped_image_list"] = [os.path.basename(path) for path in skipped_images]
    
    if error_images:
        summary_metadata["error_image_list"] = error_images
    
    # 전체 요약 메타데이터 저장
    summary_file = os.path.join(output_dir, "preprocessing_summary.json")
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary_metadata, f, indent=2)
    except Exception as e:
        print(f"[ERROR] 요약 메타데이터 저장 오류: {e}")
    
    # 최종 결과 출력
    if skipped_images:
        print(f"[INFO] {len(skipped_images)}개 이미지가 건너뛰어졌습니다 (좌표 파일 누락 또는 처리 실패)")
    
    if error_images:
        print(f"[WARNING] {len(error_images)}개 이미지에서 오류가 발생했습니다")
    
    return {
        "result": result,
        "total_objects": total_objects,
        "skipped_images": skipped_images,
        "error_images": error_images,
        "summary_file": summary_file
    }

# 기존 전처리 함수들은 유지하되, 새 함수가 우선 사용되도록
def batch_preprocess_images(image_paths, results_dir, output_dir, target_size=None, apply_masks=True, crop_objects=True):
    """
    이미지 배치 전처리 (기존 함수와 호환)
    
    Args:
        image_paths: 이미지 경로 목록
        results_dir: 탐지 결과가 있는 디렉토리
        output_dir: 출력 디렉토리
        target_size: 목표 이미지 크기 (선택적)
        apply_masks: 마스크 적용 여부
        crop_objects: 객체 크롭 여부
        
    Returns:
        dict: 처리 결과
    """
    # 좌표 파일이 있는 디렉토리 찾기
    coords_dir = os.path.dirname(results_dir)
    
    # 디렉토리 루트에서 4.mask 디렉토리 찾기
    parts = coords_dir.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("results"):
            # results가 포함된 부분을 찾으면 그 부모 디렉토리에서 4.mask 찾기
            if i > 0:
                root_dir = os.sep.join(parts[:i])
                mask_dir = os.path.join(root_dir, "4.mask")
                if os.path.exists(mask_dir):
                    coords_dir = mask_dir
                    break
    
    # coords 디렉토리가 존재하는지 확인
    if not os.path.exists(coords_dir):
        # 기본 시도
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "4.mask")
        if os.path.exists(mask_dir):
            coords_dir = mask_dir
    
    # 크기 조정 설정
    resize = None
    if target_size:
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            resize = (int(target_size[0]), int(target_size[1]))
        else:
            resize = (int(target_size), int(target_size))
    
    return batch_process_with_coords(
        image_paths, 
        coords_dir,
        output_dir,
        crop=crop_objects,
        apply_mask=apply_masks,
        resize=resize
    )

def preprocess_image(image_path, results_dir, output_dir, target_size=None, apply_masks=True, crop_objects=True):
    """
    단일 이미지 전처리 (batch_preprocess_images의 래퍼)
    
    Args:
        image_path: 이미지 경로
        results_dir: 탐지 결과가 있는 디렉토리
        output_dir: 출력 디렉토리
        target_size: 목표 이미지 크기 (선택적)
        apply_masks: 마스크 적용 여부
        crop_objects: 객체 크롭 여부
        
    Returns:
        dict: 처리 결과
    """
    # 단일 이미지를 배치 처리 함수로 전달
    return batch_preprocess_images(
        [image_path],  # 단일 이미지 경로를 리스트로 변환
        results_dir,
        output_dir,
        target_size=target_size,
        apply_masks=apply_masks,
        crop_objects=crop_objects
    )

# 추가: 분류 메타데이터 생성
def prepare_classification_structure(base_dir, category_name, classification_methods=None):
    """
    분류 결과를 저장하기 위한 디렉토리 구조 생성
    
    Args:
        base_dir: 기본 디렉토리 경로
        category_name: 카테고리 이름
        classification_methods: 분류 방법 목록 (기본값: ['method1', 'method2', 'method3', 'method4'])
    
    Returns:
        dict: 생성된 디렉토리 경로
    """
    if classification_methods is None:
        classification_methods = ['method1', 'method2', 'method3', 'method4']
    
    refine_dir = Path(base_dir) / category_name / "8.refine-dataset"
    refine_dir.mkdir(exist_ok=True, parents=True)
    
    # 각 분류 방법에 대한 디렉토리 생성
    paths = {}
    for method in classification_methods:
        method_dir = refine_dir / method
        method_dir.mkdir(exist_ok=True, parents=True)
        paths[method] = str(method_dir)
    
    # 메타데이터 파일 생성
    metadata = {
        "category": category_name,
        "classification_methods": classification_methods,
        "created_at": datetime.datetime.now().isoformat(),
        "paths": paths
    }
    
    metadata_file = refine_dir / "classification_structure.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "paths": paths,
        "metadata_file": str(metadata_file)
    }

def classify_objects(preprocessed_dir, classification_dir, method_name, class_mapping=None):
    """
    전처리된 객체를 분류 방법에 따라 분류하여 저장
    
    Args:
        preprocessed_dir: 전처리된 이미지가 있는 디렉토리
        classification_dir: 분류 결과를 저장할 디렉토리
        method_name: 분류 방법 이름
        class_mapping: 클래스 매핑 정보 (없으면 원본 클래스명 사용)
    
    Returns:
        dict: 분류 결과 정보
    """
    # 구현 예정 - 4가지 분류 방식에 따른 실제 분류 코드
    # 이 함수는 Task 4에서 구현할 수 있습니다.
    pass

# 기본 테스트 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="좌표 정보를 이용한 이미지 전처리")
    parser.add_argument("--image", help="처리할 이미지 파일 경로")
    parser.add_argument("--coords", help="좌표 JSON 파일 경로")
    parser.add_argument("--output", help="출력 디렉토리", default="output")
    parser.add_argument("--no-crop", help="객체 크롭 비활성화", action="store_true")
    parser.add_argument("--no-mask", help="마스크 적용 비활성화", action="store_true")
    parser.add_argument("--resize", help="크기 조정 (너비,높이)", default=None)
    
    args = parser.parse_args()
    
    if args.image and args.coords:
        # 크기 조정 파싱
        resize = None
        if args.resize:
            try:
                width, height = map(int, args.resize.split(","))
                resize = (width, height)
            except:
                print("[WARNING] 잘못된 크기 형식. 예: 224,224")
        
        # 이미지 처리
        result = process_image_with_coords(
            args.image,
            args.coords,
            args.output,
            crop=not args.no_crop,
            apply_mask=not args.no_mask,
            resize=resize
        )
        
        print(f"처리된 파일: {result['files']}")
    else:
        parser.print_help() 