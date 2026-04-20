#!/usr/bin/env python3
"""
좌표 정보 확인 및 시각화 도구

마스크/박스 좌표 정보를 확인하고 시각화하는 스크립트입니다.
"""

import sys
import os
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import logging

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def load_coords_data(coords_file):
    """
    좌표 데이터 파일 로드
    
    Args:
        coords_file: 좌표 데이터 파일 경로
    
    Returns:
        dict: 좌표 데이터 또는 로드 실패 시 None
    """
    try:
        with open(coords_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"좌표 파일 로딩 실패: {e}")
        return None

def print_coords_info(coords_data):
    """
    좌표 데이터 정보 출력
    
    Args:
        coords_data: 좌표 데이터 딕셔너리
    """
    if not coords_data:
        logger.warning("좌표 데이터가 없습니다.")
        return
    
    logger.info(f"마스크 형태: {coords_data['shape']}")
    logger.info(f"마스크 수: {len(coords_data['masks'])}")
    
    for i, mask_info in enumerate(coords_data['masks']):
        class_name = mask_info.get('class_name', f'객체_{i}')
        logger.info(f"\n마스크 {i} (클래스: {class_name}):")
        logger.info(f"  True 픽셀 수: {mask_info.get('true_pixels', 'N/A')}")
        logger.info(f"  윤곽선 수: {len(mask_info.get('contours', []))}")
        logger.info(f"  경계 상자 수: {len(mask_info.get('bounding_boxes', []))}")
        
        for j, box in enumerate(mask_info.get('bounding_boxes', [])):
            x, y, w, h = box
            logger.info(f"  경계 상자 {j}: x={x}, y={y}, 너비={w}, 높이={h}")

def create_mask_from_contours(contours, image_shape):
    """
    윤곽선 좌표로부터 마스크 생성
    
    Args:
        contours: 윤곽선 좌표 목록
        image_shape: 이미지 크기 (높이, 너비)
    
    Returns:
        numpy.ndarray: 생성된 마스크 (0 또는 255)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for contour in contours:
        contour_points = np.array(contour, dtype=np.int32)
        cv2.fillPoly(mask, [contour_points], 255)
    return mask

def visualize_coords(image_path, coords_data, output_path=None, show=False):
    """
    좌표 데이터를 시각화
    
    Args:
        image_path: 원본 이미지 경로
        coords_data: 좌표 데이터 딕셔너리
        output_path: 시각화 결과 저장 경로 (선택적)
        show: 결과 이미지 표시 여부 (선택적)
    
    Returns:
        numpy.ndarray: 시각화된 이미지
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"이미지 로드 실패: {image_path}")
        return None
    
    # 시각화 이미지 생성
    vis_image = image.copy()
    height, width = image.shape[:2]
    
    # 각 마스크 정보 시각화
    for i, mask_info in enumerate(coords_data.get('masks', [])):
        # 클래스 정보
        class_name = mask_info.get('class_name', f'객체_{i}')
        
        # 색상 생성 (클래스마다 다른 색상)
        color = (
            hash(class_name) % 255,
            (hash(class_name) * 2) % 255,
            (hash(class_name) * 3) % 255
        )
        
        # 윤곽선 그리기
        for contour in mask_info.get('contours', []):
            contour_points = np.array(contour, dtype=np.int32)
            cv2.polylines(vis_image, [contour_points], True, color, 2)
        
        # 경계 상자 그리기
        for j, box in enumerate(mask_info.get('bounding_boxes', [])):
            x, y, w, h = box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # 텍스트 정보
            label = f"{class_name}"
            
            # 텍스트 배경 그리기
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x, y - text_height - 5), (x + text_width, y), color, -1)
            
            # 텍스트 그리기
            cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, vis_image)
        logger.info(f"시각화 이미지 저장됨: {output_path}")
    
    # 결과 표시
    if show:
        cv2.imshow("Visualization", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis_image

def create_mask_overlay(image_path, coords_data, output_path=None, show=False):
    """
    이미지에 마스크 오버레이 생성
    
    Args:
        image_path: 원본 이미지 경로
        coords_data: 좌표 데이터 딕셔너리
        output_path: 결과 저장 경로 (선택적)
        show: 결과 이미지 표시 여부 (선택적)
    
    Returns:
        numpy.ndarray: 생성된 오버레이 이미지
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"이미지 로드 실패: {image_path}")
        return None
    
    # 이미지 크기
    height, width = image.shape[:2]
    
    # 오버레이 이미지 생성
    overlay = image.copy()
    
    # 각 마스크를 개별 색상으로 렌더링
    for i, mask_info in enumerate(coords_data.get('masks', [])):
        # 클래스 정보
        class_name = mask_info.get('class_name', f'객체_{i}')
        
        # 윤곽선으로부터 마스크 생성
        mask = create_mask_from_contours(mask_info.get('contours', []), (height, width))
        
        # 색상 생성 (클래스마다 다른 색상)
        color = (
            hash(class_name) % 255,
            (hash(class_name) * 2) % 255,
            (hash(class_name) * 3) % 255
        )
        
        # 컬러 마스크 생성
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = color
        
        # 마스크 영역을 원본 이미지에 반투명하게 적용
        alpha = 0.5  # 투명도
        mask_area = (mask > 0).astype(bool)
        overlay[mask_area] = cv2.addWeighted(image[mask_area], 1 - alpha, color_mask[mask_area], alpha, 0)
    
    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, overlay)
        logger.info(f"마스크 오버레이 저장됨: {output_path}")
    
    # 결과 표시
    if show:
        cv2.imshow("Mask Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return overlay

def main():
    parser = argparse.ArgumentParser(description="좌표 파일 정보 확인 및 시각화")
    parser.add_argument("--coords", required=True, help="좌표 JSON 파일 경로")
    parser.add_argument("--image", help="원본 이미지 경로 (시각화 시 필요)")
    parser.add_argument("--output", help="시각화 결과 저장 경로")
    parser.add_argument("--show", action="store_true", help="시각화 결과 화면에 표시")
    parser.add_argument("--overlay", action="store_true", help="반투명 마스크 오버레이 생성")
    parser.add_argument("--verbose", action="store_true", help="상세 정보 출력")
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 좌표 파일 로드
    coords_data = load_coords_data(args.coords)
    if not coords_data:
        sys.exit(1)
    
    # 좌표 정보 출력
    print_coords_info(coords_data)
    
    # 시각화 (이미지 경로가 제공된 경우)
    if args.image:
        if args.overlay:
            create_mask_overlay(args.image, coords_data, args.output, args.show)
        else:
            visualize_coords(args.image, coords_data, args.output, args.show)
    
    logger.info("완료")

if __name__ == "__main__":
    main() 