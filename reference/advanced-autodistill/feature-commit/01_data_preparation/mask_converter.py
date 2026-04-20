#!/usr/bin/env python3
import numpy as np
import json
import os
import argparse
from pathlib import Path
import cv2

def mask_to_coords(mask, min_area=10):
    """마스크에서 좌표 정보 추출
    
    Args:
        mask: 불리언 마스크 (H, W)
        min_area: 최소 면적 (이것보다 작은 객체는 무시)
        
    Returns:
        coords_list: 윤곽선 좌표 목록
        bounding_boxes: 경계 상자 목록
    """
    # 불리언 마스크를 uint8로 변환
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coords_list = []
    bounding_boxes = []
    
    for contour in contours:
        # 면적이 너무 작은 객체는 무시
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # 윤곽선 좌표 추출 (단순화)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 좌표를 평탄화하여 리스트로 변환
        coords = approx.reshape(-1, 2).tolist()
        coords_list.append(coords)
        
        # 경계 상자 계산 [x, y, width, height]
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([int(x), int(y), int(w), int(h)])
    
    return coords_list, bounding_boxes

def save_mask_data(npy_file, output_formats=None):
    """마스크 NPY 파일을 다양한 형식으로 저장
    
    Args:
        npy_file: NPY 파일 경로
        output_formats: 출력 형식 목록 ('json', 'txt', 'csv')
    """
    if output_formats is None:
        output_formats = ['json']
    
    # NPY 파일 로드
    try:
        mask_data = np.load(npy_file)
        print(f"Loaded mask with shape: {mask_data.shape}, dtype: {mask_data.dtype}")
    except Exception as e:
        print(f"Error loading NPY file: {e}")
        return
    
    # 파일 이름에서 확장자 제거
    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    dir_name = os.path.dirname(npy_file)
    
    # 결과를 저장할 데이터 구조
    result = {
        "original_file": npy_file,
        "shape": mask_data.shape,
        "masks": []
    }
    
    # 각 마스크 처리
    for i in range(mask_data.shape[0]):
        mask = mask_data[i]
        print(f"Processing mask {i+1}/{mask_data.shape[0]}")
        print(f"  - True pixels: {np.sum(mask)}")
        
        # 마스크에서 좌표 추출
        coords_list, bounding_boxes = mask_to_coords(mask)
        
        mask_info = {
            "mask_index": i,
            "true_pixels": int(np.sum(mask)),
            "total_pixels": int(mask.size),
            "contours": coords_list,
            "bounding_boxes": bounding_boxes
        }
        
        result["masks"].append(mask_info)
    
    # 다양한 형식으로 저장
    for format_type in output_formats:
        if format_type.lower() == 'json':
            # JSON 형식으로 저장
            json_file = os.path.join(dir_name, f"{base_name}_coords.json")
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved JSON file: {json_file}")
            
        elif format_type.lower() == 'txt':
            # TXT 형식으로 저장 (단순 텍스트)
            txt_file = os.path.join(dir_name, f"{base_name}_coords.txt")
            with open(txt_file, 'w') as f:
                f.write(f"Original file: {npy_file}\n")
                f.write(f"Shape: {mask_data.shape}\n\n")
                
                for i, mask_info in enumerate(result["masks"]):
                    f.write(f"Mask {i}:\n")
                    f.write(f"  True pixels: {mask_info['true_pixels']}\n")
                    f.write(f"  Total pixels: {mask_info['total_pixels']}\n")
                    f.write(f"  Bounding boxes: {mask_info['bounding_boxes']}\n")
                    f.write("  Contours:\n")
                    
                    for j, contour in enumerate(mask_info['contours']):
                        f.write(f"    Contour {j} ({len(contour)} points):\n")
                        for point in contour[:10]:  # 처음 10개 점만 표시
                            f.write(f"      {point}\n")
                        if len(contour) > 10:
                            f.write(f"      ... ({len(contour)-10} more points)\n")
            
            print(f"Saved TXT file: {txt_file}")
            
        elif format_type.lower() == 'csv':
            # CSV 형식으로 저장 (경계 상자만)
            csv_file = os.path.join(dir_name, f"{base_name}_boxes.csv")
            with open(csv_file, 'w') as f:
                f.write("mask_index,x,y,width,height\n")
                
                for i, mask_info in enumerate(result["masks"]):
                    for box in mask_info['bounding_boxes']:
                        f.write(f"{i},{box[0]},{box[1]},{box[2]},{box[3]}\n")
            
            print(f"Saved CSV file: {csv_file}")
    
    # 시각화 이미지 생성 (선택적)
    if 'vis' in output_formats or 'visualization' in output_formats:
        # 마스크 크기에 맞는 이미지 생성
        h, w = mask_data.shape[1], mask_data.shape[2]
        vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 각 마스크 별로 다른 색상 사용
        colors = [
            (0, 0, 255),   # 빨강
            (0, 255, 0),   # 녹색
            (255, 0, 0),   # 파랑
            (255, 255, 0), # 청록
            (255, 0, 255), # 자홍
            (0, 255, 255)  # 노랑
        ]
        
        for i in range(mask_data.shape[0]):
            mask = mask_data[i].astype(np.uint8) * 255
            color = colors[i % len(colors)]
            
            # 마스크 영역 색칠
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            colored_mask[mask > 0] = color
            
            # 반투명 마스크 합성
            alpha = 0.5
            vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
            
            # 윤곽선 그리기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)
            
            # 경계 상자 그리기
            for x, y, w, h in result["masks"][i]["bounding_boxes"]:
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(vis_image, f"Mask {i}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 시각화 이미지 저장
        vis_file = os.path.join(dir_name, f"{base_name}_visualization.png")
        cv2.imwrite(vis_file, vis_image)
        print(f"Saved visualization: {vis_file}")

def process_directory(input_dir, output_formats=None):
    """디렉토리 내의 모든 NPY 파일 처리
    
    Args:
        input_dir: 입력 디렉토리 경로
        output_formats: 출력 형식 목록
    """
    if output_formats is None:
        output_formats = ['json', 'txt', 'vis']
    
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No NPY files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} NPY files")
    for npy_file in npy_files:
        print(f"\nProcessing {npy_file}")
        save_mask_data(str(npy_file), output_formats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mask NPY files to other formats")
    parser.add_argument("input", help="Input NPY file or directory")
    parser.add_argument("--formats", nargs="+", default=["json", "txt", "vis"],
                       help="Output formats (json, txt, csv, vis)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.formats)
    elif os.path.isfile(args.input) and args.input.endswith(".npy"):
        save_mask_data(args.input, args.formats)
    else:
        print(f"Invalid input: {args.input}") 