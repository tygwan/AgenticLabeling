#!/usr/bin/env python3
"""
마스크 처리를 위한 유틸리티 모듈

마스크 데이터를 다양한 형식으로 변환하고 저장하는 함수들을 제공합니다.
"""

import sys
import os

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import cv2
import os
import json
from pathlib import Path

def extract_contours_from_mask(mask, min_area=10):
    """
    마스크에서 윤곽선 및 경계 상자 추출
    
    Args:
        mask: 불리언 마스크 (H, W)
        min_area: 최소 면적 (너무 작은 객체는 무시)
        
    Returns:
        tuple: (윤곽선 목록, 경계 상자 목록)
    """
    # 불리언 마스크를 uint8로 변환
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # OpenCV를 사용하여 윤곽선 찾기
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    refined_contours = []
    bounding_boxes = []
    
    for contour in contours:
        # 너무 작은 영역은 무시
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 윤곽선 단순화
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 좌표를 리스트로 변환
        coords = approx.reshape(-1, 2).tolist()
        refined_contours.append(coords)
        
        # 경계 상자 계산
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([int(x), int(y), int(w), int(h)])
    
    return refined_contours, bounding_boxes

def convert_mask_to_coords(mask_data, class_names=None):
    """
    3D 마스크 데이터를 좌표 기반 형식으로 변환
    
    Args:
        mask_data: 마스크 배열 (N, H, W)
        class_names: 클래스 이름 목록 (선택적)
        
    Returns:
        dict: 좌표 기반 데이터
    """
    result = {
        "shape": mask_data.shape,
        "masks": []
    }
    
    for i in range(mask_data.shape[0]):
        mask = mask_data[i]
        contours, boxes = extract_contours_from_mask(mask)
        
        mask_info = {
            "index": i,
            "true_pixels": int(np.sum(mask)),
            "contours": contours,
            "bounding_boxes": boxes
        }
        
        if class_names and i < len(class_names):
            mask_info["class_name"] = class_names[i]
        
        result["masks"].append(mask_info)
    
    return result

def save_mask_as_json(mask_data, output_path, class_names=None):
    """
    마스크 데이터를 JSON 파일로 저장
    
    Args:
        mask_data: 마스크 배열 (N, H, W)
        output_path: 출력 파일 경로
        class_names: 클래스 이름 목록 (선택적)
        
    Returns:
        dict: 변환된 좌표 데이터
    """
    coords_data = convert_mask_to_coords(mask_data, class_names)
    
    with open(output_path, 'w') as f:
        json.dump(coords_data, f, indent=2)
    
    return coords_data

def save_mask_as_text(coords_data, output_path):
    """
    마스크 좌표 데이터를 텍스트 파일로 저장
    
    Args:
        coords_data: 좌표 데이터 (convert_mask_to_coords의 결과)
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w') as f:
        f.write(f"마스크 형태: {coords_data['shape']}\n\n")
        
        for i, mask_info in enumerate(coords_data["masks"]):
            f.write(f"마스크 {i}")
            if "class_name" in mask_info:
                f.write(f" (클래스: {mask_info['class_name']})")
            f.write(":\n")
            
            f.write(f"  True 픽셀 수: {mask_info['true_pixels']}\n")
            f.write(f"  윤곽선 수: {len(mask_info['contours'])}\n")
            f.write(f"  경계 상자 수: {len(mask_info['bounding_boxes'])}\n")
            
            for j, box in enumerate(mask_info['bounding_boxes']):
                f.write(f"  경계 상자 {j}: x={box[0]}, y={box[1]}, 너비={box[2]}, 높이={box[3]}\n")
            
            for j, contour in enumerate(mask_info['contours']):
                f.write(f"  윤곽선 {j} ({len(contour)} 점):\n")
                for point in contour[:10]:  # 처음 10개 점만 표시
                    f.write(f"    ({point[0]}, {point[1]})\n")
                if len(contour) > 10:
                    f.write(f"    ... 외 {len(contour)-10}개 점\n")

def convert_and_save_mask(mask_data, output_dir, base_name, class_names=None, class_colors=None):
    """
    마스크 데이터를 변환하고 여러 형식으로 저장
    
    Args:
        mask_data: 마스크 배열 (N, H, W)
        output_dir: 출력 디렉토리
        base_name: 기본 파일 이름
        class_names: 클래스 이름 목록 (선택적)
        class_colors: 클래스별 색상 맵 (선택적) - 더 이상 사용되지 않음
        
    Returns:
        dict: 저장된 파일 경로
    """
    # 출력 디렉토리 확인
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 결과물 경로 설정
    json_path = output_dir / f"{base_name}_coords.json"
    txt_path = output_dir / f"{base_name}_coords.txt"
    
    # JSON 형식으로 변환 및 저장
    coords_data = save_mask_as_json(mask_data, json_path, class_names)
    
    # 텍스트 형식으로 저장
    save_mask_as_text(coords_data, txt_path)
    
    return {
        "json": str(json_path),
        "txt": str(txt_path)
    }

def load_and_verify_mask(npy_file):
    """
    마스크 NPY 파일 로드 및 검증
    
    Args:
        npy_file: NPY 파일 경로
        
    Returns:
        numpy.ndarray or None: 로드된 마스크 데이터
    """
    if not os.path.exists(npy_file):
        print(f"오류: 파일을 찾을 수 없습니다 - {npy_file}")
        return None
    
    try:
        mask_data = np.load(npy_file)
        print(f"마스크 로드 성공: {npy_file}")
        print(f"형태: {mask_data.shape}, 데이터 타입: {mask_data.dtype}")
        
        for i in range(mask_data.shape[0]):
            print(f"마스크 {i}: True 픽셀 수 = {np.sum(mask_data[i])}")
        
        return mask_data
    except Exception as e:
        print(f"마스크 로드 오류: {e}")
        return None

def process_mask_file(npy_file, output_dir=None, class_names=None, class_colors=None):
    """
    마스크 NPY 파일 처리
    
    Args:
        npy_file: NPY 파일 경로
        output_dir: 출력 디렉토리 (기본값: NPY 파일이 있는 디렉토리)
        class_names: 클래스 이름 목록 (선택적)
        class_colors: 클래스별 색상 맵 (선택적) - 더 이상 사용되지 않음
        
    Returns:
        dict or None: 저장된 파일 경로
    """
    # 마스크 로드
    mask_data = load_and_verify_mask(npy_file)
    if mask_data is None:
        return None
    
    # 출력 디렉토리 및 기본 이름 설정
    if output_dir is None:
        output_dir = os.path.dirname(npy_file)
    
    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    
    # 변환 및 저장
    return convert_and_save_mask(mask_data, output_dir, base_name, class_names, class_colors)

def create_coords_from_boxes(boxes, class_ids=None, class_names=None, image_shape=(640, 480)):
    """
    경계 상자만으로 좌표 데이터 생성
    
    Args:
        boxes: 경계 상자 목록 [[x, y, w, h], ...]
        class_ids: 클래스 ID 목록 (선택적)
        class_names: 클래스 이름 목록 또는 매핑 함수 (선택적)
        image_shape: 이미지 크기 (너비, 높이)
        
    Returns:
        dict: 좌표 데이터
    """
    # 입력 검증
    if not boxes:
        print("[WARNING] 빈 경계 상자 목록입니다")
        boxes = []
    
    if not isinstance(boxes, list):
        print(f"[WARNING] 경계 상자가 리스트가 아닙니다: {type(boxes)}")
        try:
            boxes = list(boxes)
        except:
            boxes = []
    
    # 결과 구조 초기화
    result = {
        "shape": [len(boxes), image_shape[1], image_shape[0]],  # 클래스 수, 높이, 너비
        "masks": []
    }
    
    # 클래스별 그룹화
    class_to_boxes = {}
    
    for i, box in enumerate(boxes):
        # 클래스 ID 가져오기
        class_id = 0  # 기본값
        if class_ids and i < len(class_ids):
            class_id = int(class_ids[i])
        
        # 클래스 이름 가져오기
        class_name = f"class_{class_id}"
        if class_names:
            if callable(class_names):
                # 함수일 경우
                class_name = class_names(class_id)
            elif isinstance(class_names, list) and i < len(class_names):
                # 리스트일 경우
                class_name = class_names[i]
            elif isinstance(class_names, dict) and class_id in class_names:
                # 딕셔너리일 경우
                class_name = class_names[class_id]
        
        # 클래스별 그룹에 추가
        if class_id not in class_to_boxes:
            class_to_boxes[class_id] = {
                "boxes": [],
                "class_name": class_name
            }
        
        # 유효한 경계 상자 확인 (4개 값 있는지)
        if len(box) != 4:
            print(f"[WARNING] 잘못된 경계 상자 형식: {box} - 건너뜁니다")
            continue
            
        try:
            # 숫자형으로 변환
            box = [float(val) for val in box]
        except:
            print(f"[WARNING] 경계 상자에 숫자가 아닌 값이 포함됨: {box} - 건너뜁니다")
            continue
            
        class_to_boxes[class_id]["boxes"].append(box)
    
    # 각 클래스에 대한 마스크 정보 생성
    for class_id, data in class_to_boxes.items():
        class_name = data["class_name"]
        boxes = data["boxes"]
        
        if not boxes:
            continue
        
        # 가상 윤곽선 생성
        contours = []
        for box in boxes:
            x, y, w, h = box
            # 경계 상자 주변 윤곽선을 간단하게 만듦
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            contour = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
            ]
            contours.append(contour)
        
        # 마스크 정보 추가
        mask_info = {
            "index": class_id,
            "class_id": class_id,
            "class_name": class_name,
            "true_pixels": sum([int(box[2]) * int(box[3]) for box in boxes]),  # 면적의 합
            "contours": contours,
            "bounding_boxes": [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes]
        }
        
        result["masks"].append(mask_info)
    
    return result

def save_coords_without_mask(boxes, output_dir, base_name, class_ids=None, class_names=None, 
                            class_colors=None, image_shape=(640, 480)):
    """
    마스크 없이 경계 상자 좌표만으로 좌표 파일 저장
    
    Args:
        boxes: 경계 상자 목록 [[x1, y1, x2, y2], ...]
        output_dir: 출력 디렉토리
        base_name: 기본 파일 이름
        class_ids: 클래스 ID 목록 (선택적)
        class_names: 클래스 이름 목록 (선택적)
        class_colors: 클래스별 색상 맵 (선택적) - 더 이상 사용되지 않음
        image_shape: 이미지 크기 (너비, 높이)
        
    Returns:
        dict: 저장된 파일 경로
    """
    # 유효성 검사
    if not boxes or not isinstance(boxes, list):
        print(f"[WARNING] 유효하지 않은 경계 상자 데이터: {boxes}")
        # 빈 좌표 데이터 생성
        coords_data = {
            "shape": [0, image_shape[1], image_shape[0]],
            "masks": []
        }
    else:
        # 좌표 데이터 생성
        coords_data = create_coords_from_boxes(boxes, class_ids, class_names, image_shape)
    
    # 출력 디렉토리 확인
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 결과물 경로 설정
    json_path = output_dir / f"{base_name}_coords.json"
    txt_path = output_dir / f"{base_name}_coords.txt"
    
    # JSON 형식으로 저장
    try:
        with open(json_path, 'w') as f:
            json.dump(coords_data, f, indent=2)
        print(f"[INFO] 좌표 JSON 파일 저장 완료: {json_path}")
    except Exception as e:
        print(f"[ERROR] 좌표 JSON 파일 저장 중 오류: {e}")
    
    # 텍스트 형식으로 저장
    try:
        with open(txt_path, 'w') as f:
            # 특정 YOLO 형식이나 다른 형식으로 저장
            # 여기서는 간단한 텍스트 형식 사용
            for mask_info in coords_data.get("masks", []):
                class_id = 0  # 기본값
                if "class_id" in mask_info:
                    class_id = mask_info["class_id"]
                    
                for box in mask_info.get("bounding_boxes", []):
                    # YOLO 형식으로 변환 (클래스 ID, x_center, y_center, width, height)
                    x, y, w, h = box
                    # 이미지 크기로 정규화
                    x_center = (x + w/2) / image_shape[0]
                    y_center = (y + h/2) / image_shape[1]
                    norm_w = w / image_shape[0]
                    norm_h = h / image_shape[1]
                    
                    # 클래스 ID와 함께 저장
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        print(f"[INFO] 좌표 TXT 파일 저장 완료: {txt_path}")
    except Exception as e:
        print(f"[ERROR] 좌표 TXT 파일 저장 중 오류: {e}")
    
    return {
        "json": str(json_path),
        "txt": str(txt_path)
    }

def save_polygon_coords_format(coords_data, output_path, image_shape=(640, 480)):
    """
    폴리곤 좌표를 YOLO 형식으로 저장 (A_1_1_frame_0001.txt 형태)
    
    Args:
        coords_data: 좌표 데이터 (convert_mask_to_coords의 결과)
        output_path: 출력 파일 경로
        image_shape: 이미지 크기 (너비, 높이)
    """
    with open(output_path, 'w') as f:
        for mask_info in coords_data.get("masks", []):
            class_id = mask_info.get("class_id", 0)
            contours = mask_info.get("contours", [])
            
            for contour in contours:
                # 클래스 ID로 시작
                line_parts = [str(class_id)]
                
                # 각 점을 정규화된 좌표로 변환
                for point in contour:
                    x, y = point
                    # 이미지 크기로 정규화
                    norm_x = x / image_shape[0]
                    norm_y = y / image_shape[1]
                    line_parts.extend([f"{norm_x:.5f}", f"{norm_y:.5f}"])
                
                # 라인 작성
                f.write(" ".join(line_parts) + "\n")

def save_box_points_format(boxes, class_ids, confidence_scores, output_path):
    """
    박스 정보를 box points 형식으로 저장 (A_1_1_frame_0001.png_box_points.txt 형태)
    
    Args:
        boxes: 경계 상자 목록 [[x1, y1, x2, y2], ...]
        class_ids: 클래스 ID 목록
        confidence_scores: 신뢰도 점수 목록
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w') as f:
        for i, box in enumerate(boxes):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                class_id = class_ids[i] if i < len(class_ids) else 0
                confidence = confidence_scores[i] if i < len(confidence_scores) else 1.0
                
                # 박스 형식: [x1, y1, x2, y2], Class ID: class_id, Confidence: confidence
                f.write(f"Box: [{x1}, {y1}, {x2}, {y2}], Class ID: {class_id}, Confidence: {confidence:.4f}\n")

def convert_and_save_mask_debug_format(mask_data, boxes, class_ids, confidence_scores, 
                                     output_dir, box_dir, base_name, class_names=None, image_shape=(640, 480)):
    """
    마스크 데이터를 디버그 형식으로 변환하고 저장 (요청된 형태)
    
    Args:
        mask_data: 마스크 배열 (N, H, W)
        boxes: 경계 상자 목록
        class_ids: 클래스 ID 목록
        confidence_scores: 신뢰도 점수 목록
        output_dir: 출력 디렉토리 (폴리곤 데이터용)
        box_dir: 박스 데이터 출력 디렉토리
        base_name: 기본 파일 이름
        class_names: 클래스 이름 목록 (선택적)
        image_shape: 이미지 크기 (너비, 높이)
        
    Returns:
        dict: 저장된 파일 경로
    """
    # 출력 디렉토리 확인 (폴리곤 데이터용)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 박스 디렉토리 확인
    box_dir = Path(box_dir)
    box_dir.mkdir(exist_ok=True, parents=True)
    
    # 결과물 경로 설정 (요청된 형태)
    polygon_path = output_dir / f"{base_name}.txt"  # A_1_1_frame_0001.txt 형태 (마스크 폴더에 저장)
    box_points_path = box_dir / f"{base_name}_box_points.txt"  # A_1_1_frame_0001.png_box_points.txt 형태 (박스 폴더에 저장)
    
    # 좌표 데이터 변환
    coords_data = convert_mask_to_coords(mask_data, class_names)
    
    # 폴리곤 좌표 형식으로 저장
    try:
        save_polygon_coords_format(coords_data, polygon_path, image_shape)
        print(f"[INFO] 폴리곤 좌표 파일 저장 완료: {polygon_path}")
    except Exception as e:
        print(f"[ERROR] 폴리곤 좌표 파일 저장 중 오류: {e}")
    
    # 박스 포인트 형식으로 저장 (박스 폴더에)
    try:
        save_box_points_format(boxes, class_ids, confidence_scores, box_points_path)
        print(f"[INFO] 박스 포인트 파일 저장 완료: {box_points_path}")
    except Exception as e:
        print(f"[ERROR] 박스 포인트 파일 저장 중 오류: {e}")
    
    return {
        "polygon": str(polygon_path),
        "box_points": str(box_points_path)
    }

def save_coords_without_mask_debug_format(boxes, class_ids, confidence_scores, output_dir, box_dir, base_name, 
                                        class_names=None, image_shape=(640, 480)):
    """
    마스크 없이 경계 상자만으로 디버그 형식 좌표 파일 저장
    향상된 버전: 더 상세한 폴리곤 생성 (단순 사각형이 아닌 세부 형태)
    
    Args:
        boxes: 경계 상자 목록 [[x1, y1, x2, y2], ...]
        class_ids: 클래스 ID 목록
        confidence_scores: 신뢰도 점수 목록
        output_dir: 출력 디렉토리 (폴리곤 데이터용)
        box_dir: 박스 데이터 출력 디렉토리
        base_name: 기본 파일 이름
        class_names: 클래스 이름 목록 (선택적)
        image_shape: 이미지 크기 (너비, 높이)
        
    Returns:
        dict: 저장된 파일 경로
    """
    # 유효성 검사
    if not boxes or not isinstance(boxes, list):
        print(f"[WARNING] 유효하지 않은 경계 상자 데이터: {boxes}")
        boxes = []
        class_ids = []
        confidence_scores = []
    
    # 출력 디렉토리 확인 (폴리곤 데이터용)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 박스 디렉토리 확인
    box_dir = Path(box_dir)
    box_dir.mkdir(exist_ok=True, parents=True)
    
    # 결과물 경로 설정 (요청된 형태)
    polygon_path = output_dir / f"{base_name}.txt"  # A_1_1_frame_0001.txt 형태 (마스크 폴더에 저장)
    box_points_path = box_dir / f"{base_name}_box_points.txt"  # A_1_1_frame_0001.png_box_points.txt 형태 (박스 폴더에 저장)
    
    # 폴리곤 좌표 형식으로 저장 (박스를 세부 폴리곤으로 변환)
    try:
        with open(polygon_path, 'w') as f:
            for i, box in enumerate(boxes):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    class_id = class_ids[i] if i < len(class_ids) else 0
                    
                    # 박스 중심 계산
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 박스 크기
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 중심에서 더 가까운 원형 폴리곤 생성 (타원 형태)
                    num_points = 12  # 더 많은 포인트로 세밀한 폴리곤 생성
                    
                    # 시계 방향으로 점 생성
                    polygon_points = []
                    for j in range(num_points):
                        angle = 2 * np.pi * j / num_points
                        # 타원 형태로 점 생성 (x, y 비율에 따라 다르게)
                        px = center_x + 0.5 * width * np.cos(angle)
                        py = center_y + 0.5 * height * np.sin(angle)
                        
                        # 정규화된 좌표로 변환
                        norm_px = px / image_shape[0]
                        norm_py = py / image_shape[1]
                        polygon_points.append((norm_px, norm_py))
                    
                    # 클래스 ID와 폴리곤 점들을 공백으로 구분하여 작성
                    line_parts = [str(class_id)]
                    for px, py in polygon_points:
                        line_parts.append(f"{px:.5f}")
                        line_parts.append(f"{py:.5f}")
                    
                    f.write(" ".join(line_parts) + "\n")
        print(f"[INFO] 향상된 폴리곤 좌표 파일 저장 완료: {polygon_path}")
    except Exception as e:
        print(f"[ERROR] 폴리곤 좌표 파일 저장 중 오류: {e}")
    
    # 박스 포인트 형식으로 저장 (박스 폴더에)
    try:
        save_box_points_format(boxes, class_ids, confidence_scores, box_points_path)
        print(f"[INFO] 박스 포인트 파일 저장 완료: {box_points_path}")
    except Exception as e:
        print(f"[ERROR] 박스 포인트 파일 저장 중 오류: {e}")
    
    return {
        "polygon": str(polygon_path),
        "box_points": str(box_points_path)
    }

def extract_polygon_from_mask(mask):
    """
    마스크 배열에서 경계선을 추출하여 폴리곤 좌표 목록으로 변환합니다.
    
    Args:
        mask (numpy.ndarray): 2D 불리언 마스크 배열 (True/False)
    
    Returns:
        list: 폴리곤 좌표 목록 [[x1,y1], [x2,y2], ...] 형식
    """
    import cv2
    import numpy as np
    
    # 마스크가 3D인 경우 첫 번째 채널만 사용
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]
    
    # 불리언 마스크를 uint8로 변환 (0, 255)
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # OpenCV로 윤곽선 추출
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽선 선택 (노이즈 제거)
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 윤곽선 단순화 (Douglas-Peucker 알고리즘)
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 폴리곤 좌표 변환 ([x,y] 형식으로)
    polygon = approx_contour.reshape(-1, 2).tolist()
    
    return polygon

def save_mask_as_polygon(mask, mask_directory, img_basename, class_id=None, class_name=None):
    """
    마스크에서 폴리곤을 추출하여 좌표 파일로 저장합니다.
    
    Args:
        mask (numpy.ndarray): 마스크 배열
        mask_directory (str): 폴리곤 좌표를 저장할 디렉토리 경로
        img_basename (str): 이미지 기본 파일명
        class_id (int, optional): 클래스 ID
        class_name (str, optional): 클래스 이름
    
    Returns:
        dict: 저장된 파일 경로 정보
    """
    import os
    import json
    
    results = {}
    
    # 마스크가 3D 배열인 경우 (여러 객체)
    if len(mask.shape) == 3:
        num_masks = mask.shape[0]
        
        # 각 마스크별 폴리곤 추출
        all_polygons = []
        for i in range(num_masks):
            polygon = extract_polygon_from_mask(mask[i])
            if polygon:
                # 클래스 정보 추가
                polygon_data = {
                    "polygon": polygon,
                    "class_id": class_id[i] if class_id is not None and i < len(class_id) else None,
                    "class_name": class_name[i] if class_name is not None and i < len(class_name) else None
                }
                all_polygons.append(polygon_data)
        
        # JSON 파일로 저장
        json_path = os.path.join(mask_directory, f"{img_basename}_polygon.json")
        with open(json_path, 'w') as f:
            json.dump(all_polygons, f, indent=2)
        results['json'] = json_path
        
        # TXT 파일로 저장 (각 객체별 별도 파일)
        for i, polygon_data in enumerate(all_polygons):
            polygon = polygon_data["polygon"]
            class_str = f"_{polygon_data['class_name']}" if polygon_data['class_name'] else f"_{i}"
            txt_path = os.path.join(mask_directory, f"{img_basename}{class_str}_coords.txt")
            
            with open(txt_path, 'w') as f:
                for point in polygon:
                    f.write(f"{point[0]} {point[1]}\n")
            
            if i == 0:
                results['txt'] = txt_path
            else:
                results[f'txt_{i}'] = txt_path
    
    # 단일 마스크인 경우
    else:
        polygon = extract_polygon_from_mask(mask)
        if polygon:
            # JSON 파일로 저장
            json_path = os.path.join(mask_directory, f"{img_basename}_polygon.json")
            polygon_data = {
                "polygon": polygon,
                "class_id": class_id[0] if class_id is not None else None,
                "class_name": class_name[0] if class_name is not None else None
            }
            
            with open(json_path, 'w') as f:
                json.dump([polygon_data], f, indent=2)
            results['json'] = json_path
            
            # TXT 파일로 저장
            class_str = f"_{polygon_data['class_name']}" if polygon_data['class_name'] else ""
            txt_path = os.path.join(mask_directory, f"{img_basename}{class_str}_coords.txt")
            
            with open(txt_path, 'w') as f:
                for point in polygon:
                    f.write(f"{point[0]} {point[1]}\n")
            results['txt'] = txt_path
    
    return results

def save_mask_as_yolo_format(mask, output_path, class_id=0, image_shape=(640, 480)):
    """
    마스크에서 YOLO 형식의 폴리곤 좌표 파일을 생성합니다.
    A_1_1_frame_0001.txt와 같은 형식으로 저장합니다.
    
    Args:
        mask (numpy.ndarray): 마스크 배열 (2D 또는 3D)
        output_path (str): 출력 파일 경로
        class_id (int): 객체 클래스 ID
        image_shape (tuple): 이미지 크기 (너비, 높이)
    
    Returns:
        bool: 성공 여부
    """
    import numpy as np
    import cv2
    
    # 마스크가 3D인 경우 첫 번째 채널만 사용
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            print(f"[WARNING] 여러 객체 마스크가 감지되었습니다. 첫 번째 객체만 처리합니다.")
            mask = mask[0]
    
    # 이미지 너비, 높이
    img_width, img_height = image_shape
    
    # 마스크를 uint8로 변환 (0, 255)
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # 마스크에서 윤곽선 추출 (CHAIN_APPROX_NONE으로 모든 점 유지)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print(f"[WARNING] 마스크에서 윤곽선을 찾을 수 없습니다.")
        return False
    
    # 가장 큰 윤곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 윤곽선을 단순화하지 않고 모든 포인트 유지
    # 단, 너무 세밀한 노이즈를 제거하기 위한 아주 작은 단순화만 적용
    epsilon = 0.0001 * cv2.arcLength(largest_contour, True)  # 최소한의 단순화
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 포인트 수 제한 코드 제거 - 모든 포인트 유지
    
    # YOLO 형식으로 좌표 정규화 및 변환
    line_parts = [str(class_id)]
    
    for point in approx_contour:
        x, y = point[0]
        # 이미지 크기로 정규화 (0~1 범위)
        norm_x = x / img_width
        norm_y = y / img_height
        line_parts.append(f"{norm_x:.6f}")
        line_parts.append(f"{norm_y:.6f}")
    
    try:
        with open(output_path, 'w') as f:
            # 공백으로 구분하여 한 줄에 작성
            f.write(" ".join(line_parts) + "\n")
        
        print(f"[INFO] YOLO 형식 마스크 좌표 파일 저장 완료: {output_path} (포인트 수: {len(approx_contour)})")
        return True
    except Exception as e:
        print(f"[ERROR] YOLO 형식 마스크 좌표 파일 저장 중 오류: {e}")
        return False

def save_mask_as_raw(mask, output_dir, img_basename, class_id=None, image_shape=(640, 480), save_png=True):
    """
    마스크 배열을 그대로 저장합니다 (NPY 및 선택적으로 PNG 형식).
    
    Args:
        mask (numpy.ndarray): 마스크 배열 (2D 또는 3D)
        output_dir (str): 출력 디렉토리 경로
        img_basename (str): 이미지 기본 이름 (확장자 제외)
        class_id (int 또는 list): 클래스 ID (선택적)
        image_shape (tuple): 이미지 크기 (너비, 높이)
        save_png (bool): PNG 이미지 저장 여부 (기본값: True)
    
    Returns:
        dict: 저장된 파일 경로
    """
    import numpy as np
    import cv2
    import os
    
    # 결과 저장 경로
    results = {}
    
    # 출력 디렉토리 확인
    os.makedirs(output_dir, exist_ok=True)
    
    # 마스크가 3D 배열인 경우 (여러 객체)
    if len(mask.shape) == 3 and mask.shape[0] > 1:
        # NPY 파일 저장 (원본 배열 그대로)
        npy_path = os.path.join(output_dir, f"{img_basename}_masks.npy")
        np.save(npy_path, mask)
        results['npy'] = npy_path
        
        # 각 마스크를 개별 PNG 파일로 저장
        png_paths = []
        for i in range(mask.shape[0]):
            mask_uint8 = mask[i].astype(np.uint8) * 255
            png_path = os.path.join(output_dir, f"{img_basename}_mask_{i}.png")
            cv2.imwrite(png_path, mask_uint8)
            png_paths.append(png_path)
        
        results['png'] = png_paths
        
        # 통합 마스크 이미지 생성 (모든 마스크를 하나의 이미지로)
        combined_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            # 각 마스크는 다른 그레이스케일 값을 가짐 (1번 마스크: 50, 2번 마스크: 100, ...)
            val = min(255, (i + 1) * 50)
            combined_mask = np.maximum(combined_mask, mask[i].astype(np.uint8) * val)
        
        combined_path = os.path.join(output_dir, f"{img_basename}_masks_combined.png")
        cv2.imwrite(combined_path, combined_mask)
        results['combined'] = combined_path
        
        # YOLO 레이블 파일도 생성 (클래스 ID가 있는 경우)
        if class_id is not None:
            txt_path = os.path.join(output_dir, f"{img_basename}.txt")
            with open(txt_path, 'w') as f:
                for i in range(mask.shape[0]):
                    cls = class_id[i] if isinstance(class_id, (list, np.ndarray)) else class_id
                    f.write(f"{cls}\n")
            results['txt'] = txt_path
    
    # 단일 마스크인 경우
    else:
        # 단일 마스크 처리
        if len(mask.shape) == 3:
            mask = mask[0]  # 첫 번째 채널만 사용
        
        # NPY 파일로 저장
        npy_path = os.path.join(output_dir, f"{img_basename}_mask.npy")
        np.save(npy_path, mask)
        results['npy'] = npy_path
        
        # PNG 이미지 저장 부분을 조건부로 변경
        if save_png:
            # 기존 PNG 저장 코드
            mask_uint8 = mask.astype(np.uint8) * 255
            png_path = os.path.join(output_dir, f"{img_basename}_mask.png")
            cv2.imwrite(png_path, mask_uint8)
            results['png'] = png_path
        
        # YOLO 레이블 파일 생성 (클래스 ID가 있는 경우)
        if class_id is not None:
            txt_path = os.path.join(output_dir, f"{img_basename}.txt")
            with open(txt_path, 'w') as f:
                cls = class_id[0] if isinstance(class_id, (list, np.ndarray)) else class_id
                f.write(f"{cls}\n")
            results['txt'] = txt_path
    
    print(f"[INFO] 마스크 배열 저장 완료: NPY={results.get('npy')}, PNG={results.get('png')}")
    return results

# 간단한 명령행 인터페이스
def get_category_mask_stats(category_name):
    """
    카테고리의 마스크 파일 통계 정보를 반환합니다.
    
    Args:
        category_name (str): 카테고리 이름
        
    Returns:
        dict: 마스크 통계 정보
    """
    import os
    import glob
    import numpy as np
    
    # 데이터 경로 설정
    data_path = os.path.join("data", category_name)
    mask_dir = os.path.join(data_path, "4.mask")
    
    stats = {
        "total_masks": 0,
        "npy_files": 0,
        "png_files": 0,
        "coords_files": 0,
        "average_masks_per_file": 0,
        "total_masks_pixels": 0
    }
    
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(mask_dir):
        return stats
    
    # NPY 파일 통계
    npy_files = glob.glob(os.path.join(mask_dir, "*.npy"))
    stats["npy_files"] = len(npy_files)
    
    # PNG 파일 통계
    png_files = glob.glob(os.path.join(mask_dir, "*.png"))
    stats["png_files"] = len(png_files)
    
    # 좌표 파일 통계
    coords_files = glob.glob(os.path.join(mask_dir, "*_coords.json"))
    stats["coords_files"] = len(coords_files)
    
    # 마스크 상세 정보 (최대 10개 파일까지만 분석)
    mask_details = []
    total_masks = 0
    total_pixels = 0
    
    sample_npy_files = npy_files[:10]  # 최대 10개 파일만 분석
    
    for npy_file in sample_npy_files:
        try:
            mask_data = np.load(npy_file)
            
            # 3D 마스크 배열인 경우
            if len(mask_data.shape) == 3:
                num_masks = mask_data.shape[0]
                total_masks += num_masks
                
                # 각 마스크의 True 픽셀 수
                for i in range(num_masks):
                    true_pixels = int(np.sum(mask_data[i]))
                    total_pixels += true_pixels
                    
                    if len(mask_details) < 10:  # 최대 10개 마스크 상세 정보만 저장
                        mask_details.append({
                            "file": os.path.basename(npy_file),
                            "mask_index": i,
                            "true_pixels": true_pixels
                        })
            
            # 2D 마스크 배열인 경우
            else:
                total_masks += 1
                true_pixels = int(np.sum(mask_data))
                total_pixels += true_pixels
                
                if len(mask_details) < 10:
                    mask_details.append({
                        "file": os.path.basename(npy_file),
                        "true_pixels": true_pixels
                    })
        except Exception as e:
            print(f"[ERROR] 마스크 파일 {npy_file} 분석 중 오류: {e}")
    
    stats["total_masks"] = total_masks
    stats["mask_samples"] = mask_details
    stats["total_masks_pixels"] = total_pixels
    
    if stats["npy_files"] > 0:
        stats["average_masks_per_file"] = total_masks / len(sample_npy_files)
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="마스크 NPY 파일을 다양한 형식으로 변환")
    parser.add_argument("npy_file", help="마스크 NPY 파일 경로")
    parser.add_argument("--output", "-o", help="출력 디렉토리 (기본값: NPY 파일이 있는 디렉토리)")
    
    args = parser.parse_args()
    
    process_mask_file(args.npy_file, args.output) 