import os
import cv2
import json
import numpy as np
import logging
import shutil
from tqdm import tqdm
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_class_mapping(category_path, target_classes=None):
    """
    클래스 매핑 파일 로드
    
    Args:
        category_path: 카테고리 경로
        target_classes: 대상 클래스 리스트 (예: ['Class_0', 'Class_1', 'Class_2', 'Class_3'])
        
    Returns:
        클래스 매핑 딕셔너리
    """
    mapping_file = os.path.join(category_path, "class_mapping.json")
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                full_mapping = json.load(f)
                
            # target_classes가 지정된 경우 필터링
            if target_classes:
                filtered_mapping = {}
                class_id = 0
                for target_class in target_classes:
                    # Class_0 -> 0, Class_1 -> 1 등으로 매핑
                    if target_class.startswith('Class_'):
                        original_id = target_class.split('_')[1]
                        if original_id in full_mapping:
                            filtered_mapping[str(class_id)] = full_mapping[original_id]
                            class_id += 1
                return filtered_mapping
                
        except Exception as e:
            logger.warning(f"클래스 매핑 파일 로드 실패: {e}")
    
    # 기본 매핑 반환 (target_classes 지정 시 해당 클래스만)
    if target_classes:
        mapping = {}
        for i, target_class in enumerate(target_classes):
            if target_class == 'Class_0':
                mapping[str(i)] = 'car'
            elif target_class == 'Class_1':
                mapping[str(i)] = 'fence_person'
            elif target_class == 'Class_2':
                mapping[str(i)] = 'sidewalk'
            elif target_class == 'Class_3':
                mapping[str(i)] = 'traffic cone'
            else:
                mapping[str(i)] = target_class
        return mapping
    
    return {
        '0': 'car',
        '1': 'fence_person',
        '2': 'sidewalk',
        '3': 'traffic cone'
    }

def create_refine_dataset(category_path, output_dir=None, class_mapping=None, target_size=(640, 640), 
                          refine_annotations=None, ground_truth_annotations=None,
                          confidence_threshold=0.5, val_split=0.2, target_classes=None):
    """
    Few-shot 분류 결과와 ground truth를 결합하여 refine된 YOLO 데이터셋 생성
    
    Args:
        category_path: 카테고리 폴더 경로 (data/test_category)
        output_dir: 출력 디렉토리 (기본값: category_path/8.refine-dataset)
        class_mapping: 클래스 ID와 이름 매핑 (dict)
        target_size: 대상 이미지 크기 (너비, 높이)
        refine_annotations: Few-shot 분류 결과 경로 (JSON)
        ground_truth_annotations: Ground truth 경로 (JSON 또는 폴더)
        confidence_threshold: 신뢰도 임계값 (0~1)
        val_split: 검증 데이터셋 비율 (0~1)
        target_classes: 대상 클래스 리스트 (예: ['Class_0', 'Class_1', 'Class_2', 'Class_3'])
        
    Returns:
        생성된 데이터셋 정보 딕셔너리
    """
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join(category_path, "8.refine-dataset")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 클래스 매핑 로드 또는 생성
    if class_mapping is None:
        class_mapping = load_class_mapping(category_path, target_classes)
    
    logger.info(f"클래스 매핑: {class_mapping}")
    logger.info(f"대상 클래스: {target_classes}")
    
    # 훈련/검증 디렉토리 설정
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    for split_dir in [train_dir, val_dir]:
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    
    # Few-shot 분류 결과 로드
    refine_data = {}
    if refine_annotations:
        if os.path.exists(refine_annotations):
            try:
                with open(refine_annotations, 'r') as f:
                    refine_data = json.load(f)
                logger.info(f"Few-shot 분류 결과 로드됨: {len(refine_data)} 항목")
            except Exception as e:
                logger.error(f"Few-shot 분류 결과 로드 실패: {e}")
        else:
            logger.warning(f"Few-shot 분류 결과 파일을 찾을 수 없음: {refine_annotations}")
    else:
        # 결과 디렉토리에서 최신 결과 찾기
        results_dir = os.path.join(category_path, "7.results")
        if os.path.exists(results_dir):
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
            
            if latest_result:
                try:
                    with open(latest_result, 'r') as f:
                        refine_data = json.load(f)
                    logger.info(f"최신 Few-shot 분류 결과 로드됨: {latest_result}")
                    logger.info(f"로드된 항목 수: {len(refine_data)}")
                except Exception as e:
                    logger.error(f"Few-shot 분류 결과 로드 실패: {e}")
    
    # Ground truth 로드 (target_classes 필터링 적용)
    gt_data = {}
    if ground_truth_annotations:
        if os.path.isfile(ground_truth_annotations) and ground_truth_annotations.endswith('.json'):
            try:
                with open(ground_truth_annotations, 'r') as f:
                    full_gt_data = json.load(f)
                    
                # target_classes가 지정된 경우 필터링
                if target_classes:
                    for image_name, data in full_gt_data.items():
                        # 이미지가 target_classes에 속하는지 확인
                        if "ground_truth_class" in data:
                            gt_class = data["ground_truth_class"]
                            if gt_class in target_classes:
                                # 클래스 ID를 새로운 매핑으로 변경
                                new_class_id = target_classes.index(gt_class)
                                data_copy = data.copy()
                                data_copy["class_id"] = new_class_id
                                gt_data[image_name] = data_copy
                else:
                    gt_data = full_gt_data
                    
                logger.info(f"Ground truth 로드됨: {len(gt_data)} 항목 (필터링 후)")
            except Exception as e:
                logger.error(f"Ground truth JSON 로드 실패: {e}")
        elif os.path.isdir(ground_truth_annotations):
            # 디렉토리에서 annotation 파일 로드 (target_classes만)
            if target_classes:
                for target_class in target_classes:
                    class_dir = os.path.join(ground_truth_annotations, target_class)
                    if os.path.exists(class_dir):
                        new_class_id = target_classes.index(target_class)
                        
                        for img_file in os.listdir(class_dir):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_name = os.path.splitext(img_file)[0]
                                gt_data[image_name] = {
                                    "ground_truth_class": target_class,
                                    "class_id": new_class_id,
                                    "annotations": []
                                }
            
            logger.info(f"Ground truth 디렉토리에서 로드됨: {len(gt_data)} 항목 (필터링 후)")
        else:
            # Ground truth 폴더 구조에서 직접 로드
            gt_dir = os.path.join(category_path, "7.results", "ground_truth")
            if os.path.exists(gt_dir) and target_classes:
                for target_class in target_classes:
                    class_dir = os.path.join(gt_dir, target_class)
                    if os.path.exists(class_dir):
                        new_class_id = target_classes.index(target_class)
                        
                        for img_file in os.listdir(class_dir):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_name = os.path.splitext(img_file)[0]
                                gt_data[image_name] = {
                                    "ground_truth_class": target_class,
                                    "class_id": new_class_id,
                                    "image_path": os.path.join(class_dir, img_file)
                                }
                
                logger.info(f"Ground truth 폴더에서 로드됨: {len(gt_data)} 항목")

    # 이미지 및 레이블 목록 취합
    image_list = set()
    for data_source in [refine_data, gt_data]:
        image_list.update(data_source.keys())
    
    image_count = len(image_list)
    if image_count == 0:
        logger.error("처리할 이미지가 없습니다.")
        return {"error": "No images to process"}
    
    logger.info(f"총 {image_count}개 이미지에 대한 refine 데이터셋 생성 중...")
    
    # 이미지 목록을 리스트로 변환하고 섞기
    image_list = list(image_list)
    np.random.seed(42)  # 재현성을 위한 시드 설정
    np.random.shuffle(image_list)
    
    # 훈련/검증 세트 분할
    val_count = int(image_count * val_split)
    val_images = set(image_list[:val_count])
    train_images = set(image_list[val_count:])
    
    logger.info(f"훈련 세트: {len(train_images)}개, 검증 세트: {len(val_images)}개")
    
    # 이미지 처리 통계
    stats = {
        "processed": 0,
        "skipped": 0,
        "no_annotations": 0,
        "annotations_count": 0
    }
    
    # 각 이미지 처리
    for image_name in tqdm(image_list, desc="이미지 처리 중"):
        try:
            # 원본 이미지 경로
            original_image = None
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(category_path, "1.images", f"{image_name}{ext}")
                if os.path.exists(img_path):
                    original_image = img_path
                    break
            
            if not original_image:
                logger.warning(f"이미지를 찾을 수 없음: {image_name}")
                stats["skipped"] += 1
                continue
            
            # 대상 경로 결정
            is_val = image_name in val_images
            target_dir = val_dir if is_val else train_dir
            
            # 이미지 로드 및 리사이징
            img = cv2.imread(original_image)
            if img is None:
                logger.warning(f"이미지 로드 실패: {original_image}")
                stats["skipped"] += 1
                continue
            
            original_h, original_w = img.shape[:2]
            
            # 이미지 리사이징
            resized_img = cv2.resize(img, target_size)
            
            # 이미지 저장
            img_output_path = os.path.join(target_dir, "images", f"{image_name}.jpg")
            cv2.imwrite(img_output_path, resized_img)
            
            # 레이블 처리
            annotations = []
            
            # Ground truth annotations 추가
            if image_name in gt_data and "annotations" in gt_data[image_name]:
                for annotation in gt_data[image_name]["annotations"]:
                    # 좌표가 이미 정규화된 YOLO 형식인지 확인
                    bbox = annotation.get("bbox", [0, 0, 0, 0])
                    
                    # 신뢰도 확인 (ground truth는 항상 높은 신뢰도)
                    confidence = annotation.get("confidence", 1.0)
                    if confidence < confidence_threshold:
                        continue
                    
                    # YOLO 형식 확인 (중심 x, 중심 y, 너비, 높이)
                    if len(bbox) == 4 and all(0 <= coord <= 1 for coord in bbox):
                        annotations.append({
                            "class_id": annotation.get("class_id", 0),
                            "bbox_yolo": bbox,
                            "confidence": confidence
                        })
                    # [x1, y1, x2, y2] 형식인 경우 YOLO 형식으로 변환
                    elif len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        
                        # 픽셀 좌표인 경우 정규화
                        if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                            x1 /= original_w
                            y1 /= original_h
                            x2 /= original_w
                            y2 /= original_h
                        
                        # 중심 좌표 및 크기 계산
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        annotations.append({
                            "class_id": annotation.get("class_id", 0),
                            "bbox_yolo": [x_center, y_center, width, height],
                            "confidence": confidence
                        })
            
            # Few-shot 분류 결과 추가
            if image_name in refine_data:
                # 박스 정보 확인
                if "boxes" in refine_data[image_name]:
                    boxes = refine_data[image_name]["boxes"]
                    
                    # 분류 결과 확인
                    classifications = refine_data[image_name].get("classifications", [])
                    
                    for i, box in enumerate(boxes):
                        # 해당 박스의 분류 결과 찾기
                        class_id = None
                        confidence = 0.0
                        
                        for cls in classifications:
                            if cls.get("box_index") == i:
                                class_name = cls.get("class")
                                confidence = cls.get("confidence", 0.0)
                                
                                # 클래스 이름을 ID로 변환
                                if class_name:
                                    class_name_lower = class_name.lower()
                                    for cid, cname in class_mapping.items():
                                        if cname.lower() == class_name_lower:
                                            class_id = int(cid)
                                            break
                        
                        # 신뢰도 임계값 확인
                        if confidence < confidence_threshold:
                            continue
                        
                        # 클래스 ID가 없으면 건너뛰기
                        if class_id is None:
                            continue
                        
                        # 박스 좌표가 있는지 확인
                        if isinstance(box, list) and len(box) == 4:
                            x1, y1, x2, y2 = box
                            
                            # 픽셀 좌표인 경우 정규화
                            if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                                x1 /= original_w
                                y1 /= original_h
                                x2 /= original_w
                                y2 /= original_h
                            
                            # 중심 좌표 및 크기 계산
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            annotations.append({
                                "class_id": class_id,
                                "bbox_yolo": [x_center, y_center, width, height],
                                "confidence": confidence
                            })
                
                # annotations 형식의 데이터 처리
                elif "annotations" in refine_data[image_name]:
                    for annotation in refine_data[image_name]["annotations"]:
                        # 신뢰도 임계값 확인
                        confidence = annotation.get("confidence", 0.0)
                        if confidence < confidence_threshold:
                            continue
                        
                        # 클래스 ID 확인
                        class_id = annotation.get("class_id")
                        if class_id is None:
                            class_name = annotation.get("class")
                            if class_name:
                                class_name_lower = class_name.lower()
                                for cid, cname in class_mapping.items():
                                    if cname.lower() == class_name_lower:
                                        class_id = int(cid)
                                        break
                        
                        # 클래스 ID가 없으면 건너뛰기
                        if class_id is None:
                            continue
                        
                        # 박스 좌표 처리
                        bbox = annotation.get("bbox", [])
                        
                        # YOLO 형식 확인 (중심 x, 중심 y, 너비, 높이)
                        if len(bbox) == 4 and all(0 <= coord <= 1 for coord in bbox):
                            annotations.append({
                                "class_id": class_id,
                                "bbox_yolo": bbox,
                                "confidence": confidence
                            })
                        # [x1, y1, x2, y2] 형식인 경우 YOLO 형식으로 변환
                        elif len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            
                            # 픽셀 좌표인 경우 정규화
                            if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                                x1 /= original_w
                                y1 /= original_h
                                x2 /= original_w
                                y2 /= original_h
                            
                            # 중심 좌표 및 크기 계산
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            annotations.append({
                                "class_id": class_id,
                                "bbox_yolo": [x_center, y_center, width, height],
                                "confidence": confidence
                            })
            
            # 어노테이션이 없으면 건너뛰기
            if not annotations:
                stats["no_annotations"] += 1
                # 생성된 이미지 제거
                os.remove(img_output_path)
                continue
            
            # 중복 제거 (같은 클래스 ID와 매우 유사한 위치의 박스)
            unique_annotations = []
            for anno in annotations:
                is_duplicate = False
                for unique in unique_annotations:
                    if anno["class_id"] == unique["class_id"]:
                        # 바운딩 박스 중심점 비교
                        x1, y1 = anno["bbox_yolo"][0], anno["bbox_yolo"][1]
                        x2, y2 = unique["bbox_yolo"][0], unique["bbox_yolo"][1]
                        
                        # 중심점 거리가 0.05 이하면 중복으로 판단
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if distance < 0.05:
                            # 신뢰도가 더 높은 것으로 업데이트
                            if anno["confidence"] > unique["confidence"]:
                                unique["bbox_yolo"] = anno["bbox_yolo"]
                                unique["confidence"] = anno["confidence"]
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_annotations.append(anno)
            
            # YOLO 형식으로 레이블 저장
            label_output_path = os.path.join(target_dir, "labels", f"{image_name}.txt")
            with open(label_output_path, 'w') as f:
                for annotation in unique_annotations:
                    class_id = annotation["class_id"]
                    x_center, y_center, width, height = annotation["bbox_yolo"]
                    
                    # 좌표가 0~1 범위 내에 있는지 확인
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            stats["processed"] += 1
            stats["annotations_count"] += len(unique_annotations)
            
        except Exception as e:
            logger.error(f"이미지 {image_name} 처리 중 오류: {e}")
            stats["skipped"] += 1
            import traceback
            logger.error(traceback.format_exc())
    
    # 데이터셋 정보 생성
    dataset_info = {
        "train_images": len(os.listdir(os.path.join(train_dir, "images"))),
        "val_images": len(os.listdir(os.path.join(val_dir, "images"))),
        "target_size": list(target_size),
        "classes": class_mapping,
        "stats": stats
    }
    
    # 데이터셋 정보 저장
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 데이터셋 YAML 파일 생성
    create_dataset_yaml(output_dir, class_mapping)
    
    logger.info(f"Refine 데이터셋 생성 완료:")
    logger.info(f"  - 훈련 이미지: {dataset_info['train_images']}개")
    logger.info(f"  - 검증 이미지: {dataset_info['val_images']}개")
    logger.info(f"  - 총 어노테이션: {stats['annotations_count']}개")
    logger.info(f"  - 건너뛴 이미지: {stats['skipped']}개")
    logger.info(f"  - 어노테이션 없는 이미지: {stats['no_annotations']}개")
    
    return dataset_info

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
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n\n")
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        
        for class_id, class_name in class_mapping.items():
            f.write(f"  {class_id}: {class_name}\n")
    
    logger.info(f"데이터셋 YAML 파일 생성됨: {yaml_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Few-shot 분류 결과와 Ground Truth를 결합하여 YOLO 데이터셋 생성")
    parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름")
    parser.add_argument("--output", type=str, help="출력 디렉토리 (기본값: data/{category}/8.refine-dataset)")
    parser.add_argument("--refine", type=str, help="Few-shot 분류 결과 파일 (JSON)")
    parser.add_argument("--ground-truth", type=str, help="Ground Truth 파일 또는 디렉토리")
    parser.add_argument("--target-size", type=str, default="640,640", help="대상 이미지 크기 (너비,높이)")
    parser.add_argument("--confidence", type=float, default=0.5, help="신뢰도 임계값 (0~1)")
    parser.add_argument("--val-split", type=float, default=0.2, help="검증 데이터셋 비율 (0~1)")
    parser.add_argument("--target-classes", type=str, nargs='+', help="대상 클래스 리스트 (예: Class_0 Class_1 Class_2 Class_3)")
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
        output_dir = os.path.join(category_path, "8.refine-dataset")
    
    # 대상 크기 파싱
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Refine 데이터셋 생성
    create_refine_dataset(
        category_path=category_path,
        output_dir=output_dir,
        target_size=target_size,
        refine_annotations=args.refine,
        ground_truth_annotations=args.ground_truth,
        confidence_threshold=args.confidence,
        val_split=args.val_split,
        target_classes=args.target_classes
    ) 