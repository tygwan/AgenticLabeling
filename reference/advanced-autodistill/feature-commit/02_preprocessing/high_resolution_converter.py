import os
import cv2
import json
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_box_points_file(file_path):
    """
    box_points.txt 파일을 파싱하여 박스 데이터 추출
    
    Args:
        file_path: 박스 포인트 파일 경로
        
    Returns:
        Dictionary 형태의 박스 데이터
    """
    boxes = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id x1 y1 x2 y2 (confidence)
                    try:
                        class_id = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        confidence = float(parts[5]) if len(parts) > 5 else 1.0
                        
                        boxes.append({
                            'xyxy': [x1, y1, x2, y2],
                            'class_id': class_id,
                            'confidence': confidence
                        })
                    except ValueError as e:
                        logger.warning(f"파싱 오류 (무시됨): {e} - 라인: {line}")
                        continue
        
        return {'boxes': boxes}
    except Exception as e:
        logger.error(f"Error parsing box points file {file_path}: {e}")
        return None

def get_class_id_from_name(class_name):
    """
    클래스 이름을 ID로 변환
    
    Args:
        class_name: 클래스 이름
        
    Returns:
        클래스 ID
    """
    class_mapping = {
        'car': 0,
        'fence_person': 1,
        'sidewalk': 2,
        'traffic cone': 3
    }
    
    return class_mapping.get(class_name.lower(), None)

def load_class_mapping(category_path):
    """
    클래스 매핑 파일 로드
    
    Args:
        category_path: 카테고리 경로
        
    Returns:
        클래스 매핑 딕셔너리
    """
    mapping_file = os.path.join(category_path, "class_mapping.json")
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"클래스 매핑 파일 로드 실패: {e}")
    
    # 기본 매핑 반환
    return {
        '0': 'car',
        '1': 'fence_person',
        '2': 'sidewalk',
        '3': 'traffic cone'
    }

def find_latest_classification_result(category_path):
    """
    가장 최근의 Few-shot 분류 결과 파일 찾기
    
    Args:
        category_path: 카테고리 경로
        
    Returns:
        최신 결과 파일 경로
    """
    results_dir = os.path.join(category_path, "7.results")
    
    if not os.path.exists(results_dir):
        logger.error(f"결과 디렉토리를 찾을 수 없음: {results_dir}")
        return None
    
    latest_result = None
    latest_time = 0
    
    for shot_dir in os.listdir(results_dir):
        shot_path = os.path.join(results_dir, shot_dir)
        if os.path.isdir(shot_path):
            result_file = os.path.join(shot_path, "results.json")
            if os.path.exists(result_file):
                file_time = os.path.getmtime(result_file)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_result = result_file
    
    if not latest_result:
        logger.warning("Few-shot 분류 결과를 찾을 수 없습니다.")
    else:
        logger.info(f"사용할 분류 결과 파일: {latest_result}")
    
    return latest_result

def create_high_res_from_annotations(category_path, output_dir, target_size=(640, 640)):
    """
    Few-shot 분류에서 사용된 224x224 이미지 대신 원본 이미지와 annotation 정보를 활용해
    640x640 고해상도 이미지와 레이블 생성
    
    Args:
        category_path: 카테고리 기본 경로 (data/test_category)
        output_dir: 출력 디렉토리
        target_size: 대상 이미지 크기 (기본: 640x640)
        
    Returns:
        처리된 이미지 수
    """
    # 필요한 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # 원본 이미지 경로와 마스크/어노테이션 경로
    images_dir = os.path.join(category_path, "1.images")
    masks_dir = os.path.join(category_path, "4.mask")
    box_dir = os.path.join(category_path, "3.box")
    
    # 디렉토리 존재 확인
    for required_dir in [images_dir, masks_dir, box_dir]:
        if not os.path.exists(required_dir):
            logger.warning(f"필수 디렉토리를 찾을 수 없음: {required_dir}")
    
    # 클래스 매핑 로드
    class_mapping = load_class_mapping(category_path)
    logger.info(f"클래스 매핑: {class_mapping}")
    
    # 클래스 매핑 역방향 (이름 -> ID)
    class_name_to_id = {}
    for class_id, class_name in class_mapping.items():
        class_name_to_id[class_name.lower()] = int(class_id)
    
    # Few-shot 분류 결과 찾기
    latest_result = find_latest_classification_result(category_path)
    if not latest_result:
        logger.error("Few-shot 분류 결과 없이 계속할 수 없습니다.")
        return 0
    
    # 분류 결과 로드
    try:
        with open(latest_result, 'r') as f:
            classification_results = json.load(f)
        logger.info(f"{len(classification_results)} 개의 분류 결과 로드됨")
    except Exception as e:
        logger.error(f"분류 결과 로드 실패: {e}")
        return 0
    
    # 이미지 처리
    processed_count = 0
    skipped_count = 0
    
    for img_name, result in tqdm(classification_results.items(), desc="이미지 처리 중"):
        try:
            # 원본 이미지 찾기
            original_image = None
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(images_dir, f"{img_name}{ext}")
                if os.path.exists(img_path):
                    original_image = img_path
                    break
            
            if not original_image:
                logger.warning(f"원본 이미지를 찾을 수 없음: {img_name}")
                skipped_count += 1
                continue
            
            # 박스 정보 찾기
            box_file = os.path.join(box_dir, f"{img_name}_box.json")
            box_data = None
            
            if os.path.exists(box_file):
                try:
                    with open(box_file, 'r') as f:
                        box_data = json.load(f)
                except Exception as e:
                    logger.warning(f"박스 데이터 로드 실패: {e}")
            
            # 박스 정보가 없으면 대체 파일 찾기
            if not box_data:
                alt_box_file = os.path.join(masks_dir, f"{img_name}_box_points.txt")
                if os.path.exists(alt_box_file):
                    box_data = parse_box_points_file(alt_box_file)
                    logger.debug(f"대체 박스 파일 사용: {alt_box_file}")
            
            if not box_data or not box_data.get('boxes'):
                logger.warning(f"박스 데이터를 찾을 수 없음: {img_name}")
                skipped_count += 1
                continue
            
            # 원본 이미지 로드
            orig_img = cv2.imread(original_image)
            if orig_img is None:
                logger.warning(f"이미지 로드 실패: {original_image}")
                skipped_count += 1
                continue
            
            orig_h, orig_w = orig_img.shape[:2]
            
            # 640x640 크기로 리사이징
            resized_img = cv2.resize(orig_img, target_size)
            
            # 이미지 저장
            output_img_path = os.path.join(output_dir, "images", f"{img_name}.jpg")
            cv2.imwrite(output_img_path, resized_img)
            
            # 레이블 생성
            label_path = os.path.join(output_dir, "labels", f"{img_name}.txt")
            
            with open(label_path, 'w') as f:
                # 박스 데이터와 분류 결과 조합
                boxes = box_data.get('boxes', [])
                
                annotations_count = 0
                
                for i, box in enumerate(boxes):
                    # 박스 좌표 추출
                    if isinstance(box, dict) and 'xyxy' in box:
                        x1, y1, x2, y2 = box['xyxy']
                    elif isinstance(box, list) and len(box) == 4:
                        x1, y1, x2, y2 = box
                    else:
                        continue
                    
                    # 좌표가 이미지 범위를 벗어나는지 확인
                    if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h:
                        logger.debug(f"범위 밖 좌표 조정: {x1},{y1},{x2},{y2} -> 이미지 크기: {orig_w}x{orig_h}")
                        x1 = max(0, min(x1, orig_w))
                        y1 = max(0, min(y1, orig_h))
                        x2 = max(0, min(x2, orig_w))
                        y2 = max(0, min(y2, orig_h))
                    
                    # Few-shot 분류 결과에서 클래스 ID 가져오기
                    class_id = None
                    if result and 'classifications' in result:
                        for cls_result in result['classifications']:
                            if cls_result.get('box_index') == i:
                                # 클래스 이름을 ID로 변환
                                class_name = cls_result.get('class')
                                if class_name:
                                    class_id = class_name_to_id.get(class_name.lower())
                    
                    # 박스의 클래스 ID도 확인 (백업)
                    if class_id is None and 'class_id' in box:
                        class_id = box['class_id']
                    
                    # 클래스 ID가 없으면 다음 박스로
                    if class_id is None:
                        continue
                    
                    # 좌표를 정규화된 YOLO 형식으로 변환
                    # 원본 이미지 크기로 정규화
                    x1_norm = x1 / orig_w
                    y1_norm = y1 / orig_h
                    x2_norm = x2 / orig_w
                    y2_norm = y2 / orig_h
                    
                    # 중심점과 너비, 높이 계산
                    x_center = (x1_norm + x2_norm) / 2
                    y_center = (y1_norm + y2_norm) / 2
                    width = x2_norm - x1_norm
                    height = y2_norm - y1_norm
                    
                    # YOLO 형식으로 레이블 작성
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    annotations_count += 1
            
            # 어노테이션이 없으면 레이블 파일 삭제
            if annotations_count == 0:
                os.remove(label_path)
                os.remove(output_img_path)
                logger.warning(f"어노테이션이 없는 이미지 건너뜀: {img_name}")
                skipped_count += 1
                continue
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"이미지 {img_name} 처리 중 오류: {e}")
            skipped_count += 1
            continue
    
    # 데이터셋 정보 생성
    dataset_info = {
        "processed_images": processed_count,
        "skipped_images": skipped_count,
        "target_size": list(target_size),
        "classes": class_mapping
    }
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 데이터셋 YAML 파일 생성
    create_dataset_yaml(output_dir, class_mapping)
    
    logger.info(f"처리 완료: {processed_count}개 이미지 및 레이블 생성됨 (건너뜀: {skipped_count}개)")
    return processed_count

def create_dataset_yaml(dataset_dir, class_mapping):
    """
    YOLOv8 학습용 dataset.yaml 파일 생성
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        class_mapping: 클래스 매핑 딕셔너리
    """
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLOv8 Dataset Configuration\n")
        f.write(f"path: {os.path.abspath(dataset_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        
        for class_id, class_name in class_mapping.items():
            f.write(f"  {class_id}: {class_name}\n")
    
    logger.info(f"데이터셋 YAML 파일 생성됨: {yaml_path}")

def resize_dataset_images(dataset_dir, target_size=(640, 640)):
    """
    데이터셋 이미지를 지정된 크기로 리사이징
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        target_size: 대상 이미지 크기 (너비, 높이)
    """
    # 이미지 및 레이블 경로
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        logger.error(f"이미지 디렉토리를 찾을 수 없음: {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"{len(image_files)}개 이미지 리사이징 시작 (target: {target_size})")
    
    for img_file in tqdm(image_files, desc="이미지 리사이징"):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            continue
        
        # 원본 크기 기록
        original_h, original_w = img.shape[:2]
        
        # 이미지 리사이징
        resized_img = cv2.resize(img, target_size)
        
        # 이미지 저장
        cv2.imwrite(img_path, resized_img)
    
    logger.info(f"이미지 리사이징 완료: {len(image_files)}개")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Few-shot 분류 결과를 고해상도 데이터셋으로 변환")
    parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름")
    parser.add_argument("--output", type=str, help="출력 디렉토리 (기본값: data/{category}/8.refine-dataset/high_res)")
    parser.add_argument("--target-size", type=str, default="640,640", help="대상 이미지 크기 (너비,높이)")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅 활성화")
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 카테고리 경로
    category_path = os.path.join("data", args.category)
    
    # 출력 디렉토리
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(category_path, "8.refine-dataset", "high_res")
    
    # 대상 크기 파싱
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # 고해상도 데이터셋 생성
    create_high_res_from_annotations(category_path, output_dir, target_size) 