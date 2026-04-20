import os
import cv2
import logging
import glob
import json
import yaml
import shutil
from tqdm import tqdm
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_dataset_images(dataset_dir, target_size=(640, 640), backup=True):
    """
    YOLOv8 데이터셋 이미지를 지정된 크기로 리사이징
    
    Args:
        dataset_dir: 데이터셋 디렉토리 (5.dataset)
        target_size: 대상 이미지 크기 (기본: 640x640)
        backup: 원본 이미지 백업 여부
    
    Returns:
        처리된 이미지 수
    """
    if not os.path.exists(dataset_dir):
        logger.error(f"데이터셋 디렉토리를 찾을 수 없음: {dataset_dir}")
        return 0
    
    # 이미지 경로 찾기 (train/val)
    image_dirs = []
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split, 'images')
        if os.path.exists(split_dir):
            image_dirs.append(split_dir)
    
    if not image_dirs:
        # 이미지 경로가 없으면 다른 방식으로 탐색 (분할 없는 형태)
        image_dir = os.path.join(dataset_dir, 'images')
        if os.path.exists(image_dir):
            image_dirs.append(image_dir)
    
    if not image_dirs:
        logger.error(f"이미지 디렉토리를 찾을 수 없음")
        return 0
    
    # 백업 디렉토리 생성
    if backup:
        backup_dir = os.path.join(os.path.dirname(dataset_dir), 'backup_original_dataset')
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"원본 이미지 백업 디렉토리: {backup_dir}")
        
        # 백업 복사
        try:
            shutil.copytree(dataset_dir, backup_dir, dirs_exist_ok=True)
            logger.info("원본 데이터셋 백업 완료")
        except Exception as e:
            logger.error(f"백업 중 오류 발생: {e}")
            if input("백업 없이 계속하시겠습니까? (y/n): ").lower() != 'y':
                return 0
    
    # 총 이미지 수 계산
    total_images = 0
    for image_dir in image_dirs:
        total_images += len(glob.glob(os.path.join(image_dir, "*.jpg")))
        total_images += len(glob.glob(os.path.join(image_dir, "*.jpeg")))
        total_images += len(glob.glob(os.path.join(image_dir, "*.png")))
    
    logger.info(f"총 {total_images}개 이미지 리사이징 시작 (target: {target_size[0]}x{target_size[1]})")
    
    # 이미지 처리 카운터
    processed_count = 0
    skipped_count = 0
    
    # 각 이미지 디렉토리 처리
    for image_dir in image_dirs:
        logger.info(f"디렉토리 처리 중: {image_dir}")
        
        # 이미지 파일 목록
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        
        # 각 이미지 처리
        for img_path in tqdm(image_files, desc=f"리사이징 중 ({os.path.basename(image_dir)})"):
            try:
                # 이미지 로드
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"이미지 로드 실패: {img_path}")
                    skipped_count += 1
                    continue
                
                # 원본 크기 확인
                h, w = img.shape[:2]
                
                # 이미 타겟 크기와 같으면 건너뛰기
                if h == target_size[1] and w == target_size[0]:
                    logger.debug(f"이미지가 이미 {target_size[0]}x{target_size[1]} 크기임: {img_path}")
                    processed_count += 1
                    continue
                
                # 이미지 리사이징
                resized_img = cv2.resize(img, target_size)
                
                # 같은 경로에 저장
                cv2.imwrite(img_path, resized_img)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"이미지 {img_path} 처리 중 오류: {e}")
                skipped_count += 1
    
    # YAML 파일 업데이트
    try:
        update_dataset_yaml(dataset_dir, target_size)
    except Exception as e:
        logger.warning(f"데이터셋 YAML 파일 업데이트 실패: {e}")
    
    # 처리 결과 요약
    logger.info(f"이미지 리사이징 완료:")
    logger.info(f"  - 처리된 이미지: {processed_count}개")
    logger.info(f"  - 건너뛴 이미지: {skipped_count}개")
    logger.info(f"  - 대상 크기: {target_size[0]}x{target_size[1]}")
    
    if backup:
        logger.info(f"  - 원본 백업 위치: {backup_dir}")
    
    return processed_count

def update_dataset_yaml(dataset_dir, target_size):
    """
    데이터셋 YAML 파일 업데이트
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        target_size: 타겟 이미지 크기
    """
    # YAML 파일 경로 탐색
    yaml_paths = glob.glob(os.path.join(dataset_dir, "*.yaml"))
    yaml_paths.extend(glob.glob(os.path.join(dataset_dir, "*.yml")))
    
    if not yaml_paths:
        logger.warning("데이터셋 YAML 파일을 찾을 수 없음")
        return
    
    for yaml_path in yaml_paths:
        try:
            # YAML 파일 로드
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 이미지 크기 정보 추가
            if 'image_size' not in data:
                data['image_size'] = list(target_size)
            else:
                data['image_size'] = list(target_size)
            
            # 파일 저장
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            logger.info(f"YAML 파일 업데이트 완료: {yaml_path}")
        except Exception as e:
            logger.warning(f"YAML 파일 {yaml_path} 업데이트 실패: {e}")

def resize_all_autodistill_datasets(data_dir="data", target_size=(640, 640), backup=True):
    """
    모든 카테고리의 autodistill 데이터셋을 리사이징
    
    Args:
        data_dir: 데이터 디렉토리
        target_size: 타겟 이미지 크기
        backup: 백업 여부
    
    Returns:
        처리된 데이터셋 수
    """
    if not os.path.exists(data_dir):
        logger.error(f"데이터 디렉토리를 찾을 수 없음: {data_dir}")
        return 0
    
    # 카테고리 목록 확인
    categories = []
    for item in os.listdir(data_dir):
        cat_dir = os.path.join(data_dir, item)
        if os.path.isdir(cat_dir):
            dataset_dir = os.path.join(cat_dir, "5.dataset")
            if os.path.exists(dataset_dir):
                categories.append(item)
    
    if not categories:
        logger.error("autodistill 데이터셋이 있는 카테고리를 찾을 수 없음")
        return 0
    
    logger.info(f"{len(categories)}개 카테고리의 autodistill 데이터셋을 {target_size[0]}x{target_size[1]} 크기로 리사이징합니다.")
    
    # 카테고리별 처리
    processed_categories = 0
    for category in categories:
        dataset_dir = os.path.join(data_dir, category, "5.dataset")
        logger.info(f"카테고리 처리 중: {category}")
        
        try:
            count = resize_dataset_images(dataset_dir, target_size, backup)
            if count > 0:
                processed_categories += 1
                logger.info(f"{category} 카테고리 처리 완료: {count}개 이미지")
            else:
                logger.warning(f"{category} 카테고리 처리 실패")
        except Exception as e:
            logger.error(f"{category} 카테고리 처리 중 오류 발생: {e}")
    
    logger.info(f"총 {processed_categories}/{len(categories)} 카테고리 처리 완료")
    return processed_categories

def process_category_dataset(category, target_size=(640, 640), backup=True):
    """
    특정 카테고리의 autodistill 데이터셋 처리
    
    Args:
        category: 카테고리 이름
        target_size: 타겟 이미지 크기
        backup: 백업 여부
        
    Returns:
        처리된 이미지 수
    """
    dataset_dir = os.path.join("data", category, "5.dataset")
    
    if not os.path.exists(dataset_dir):
        logger.error(f"데이터셋 디렉토리를 찾을 수 없음: {dataset_dir}")
        return 0
    
    logger.info(f"{category} 카테고리의 autodistill 데이터셋을 {target_size[0]}x{target_size[1]} 크기로 리사이징합니다.")
    
    try:
        count = resize_dataset_images(dataset_dir, target_size, backup)
        logger.info(f"처리 완료: {count}개 이미지")
        return count
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        return 0

def create_dataset_info(dataset_dir):
    """
    데이터셋 정보 파일 생성
    
    Args:
        dataset_dir: 데이터셋 디렉토리
    """
    # 이미지 경로 찾기 (train/val)
    image_dirs = []
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split, 'images')
        if os.path.exists(split_dir):
            image_dirs.append((split, split_dir))
    
    if not image_dirs:
        # 이미지 경로가 없으면 다른 방식으로 탐색 (분할 없는 형태)
        image_dir = os.path.join(dataset_dir, 'images')
        if os.path.exists(image_dir):
            image_dirs.append(('all', image_dir))
    
    if not image_dirs:
        logger.error(f"이미지 디렉토리를 찾을 수 없음")
        return
    
    # 정보 수집
    info = {
        "dataset_path": dataset_dir,
        "splits": {}
    }
    
    # 각 분할별 정보
    for split_name, image_dir in image_dirs:
        # 이미지 파일 목록
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        
        # 첫 번째 이미지로 크기 확인
        if image_files:
            try:
                img = cv2.imread(image_files[0])
                if img is not None:
                    h, w = img.shape[:2]
                    info["image_size"] = [w, h]
            except:
                pass
        
        # 해당 분할 정보 저장
        info["splits"][split_name] = {
            "images_count": len(image_files),
            "images_dir": image_dir
        }
        
        # 레이블 디렉토리 확인
        labels_dir = os.path.join(os.path.dirname(image_dir), 'labels')
        if os.path.exists(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
            info["splits"][split_name]["labels_count"] = len(label_files)
            info["splits"][split_name]["labels_dir"] = labels_dir
    
    # YAML 파일 정보
    yaml_paths = glob.glob(os.path.join(dataset_dir, "*.yaml"))
    yaml_paths.extend(glob.glob(os.path.join(dataset_dir, "*.yml")))
    
    if yaml_paths:
        info["yaml_file"] = yaml_paths[0]
        try:
            with open(yaml_paths[0], 'r') as f:
                yaml_data = yaml.safe_load(f)
                if 'names' in yaml_data:
                    info["classes"] = yaml_data['names']
        except:
            pass
    
    # 정보 파일 저장
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"데이터셋 정보 파일 생성됨: {info_path}")
    return info

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="autodistill 데이터셋 이미지 리사이징")
    parser.add_argument("--category", type=str, help="처리할 카테고리 이름 (없으면 모든 카테고리)")
    parser.add_argument("--target-size", type=str, default="640,640", help="대상 이미지 크기 (너비,높이)")
    parser.add_argument("--no-backup", action="store_true", help="원본 이미지 백업 비활성화")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅 활성화")
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 대상 크기 파싱
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # 백업 여부
    backup = not args.no_backup
    
    # 카테고리 처리
    if args.category:
        process_category_dataset(args.category, target_size, backup)
    else:
        resize_all_autodistill_datasets("data", target_size, backup) 