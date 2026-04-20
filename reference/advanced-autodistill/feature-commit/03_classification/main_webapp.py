import os
import sys
import json
import logging
import subprocess
import glob
import time
import threading
import shutil
import tempfile
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main_webapp.log")
    ]
)
logger = logging.getLogger(__name__)

# 프로젝트 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 프로젝트 모듈 임포트
try:
    from scripts.mask_utils import save_mask_as_raw, get_category_mask_stats
    from scripts.preprocess_utils import preprocess_image
    from scripts.data_utils import get_categories, get_files_in_category
    import scripts.main_launcher as main_launcher
    from scripts.high_resolution_converter import create_high_res_from_annotations
    from scripts.refine_dataset import create_refine_dataset
    from scripts.autodistill_dataset_resizer import resize_dataset_images, process_category_dataset
    
    # Ground Truth Labeler 추가
    try:
        from scripts.ground_truth_labeler import GroundTruthLabeler
        GROUND_TRUTH_LABELER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Ground Truth Labeler 모듈을 임포트할 수 없습니다: {e}")
        GROUND_TRUTH_LABELER_AVAILABLE = False
except ImportError as e:
    logger.error(f"모듈 임포트 오류: {e}")
    logger.error("필요한 모듈을 임포트할 수 없습니다. 스크립트 경로를 확인하세요.")
    sys.exit(1)

# 설정 로드
DEFAULT_CONFIG = {
    "data_path": os.path.join(project_dir, "data"),
    "save_mask_png": False,
    "target_size": [640, 640],
    "resize_dataset": True,
    "confidence_threshold": 0.5,
    "val_split": 0.2,
    "models": {
        "detection": "grounding-dino-2",
        "segmentation": "sam-2",
        "few_shot": {
            "CLIP": {"enabled": True},
            "DINOv2": {"enabled": True},
            "Florence-2": {"enabled": True}
        }
    }
}

# 설정 파일 경로
CONFIG_PATH = os.path.join(project_dir, "config", "main_webapp_config.json")

def load_config():
    """설정 파일 로드"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                logger.info("설정 파일 로드 완료")
                return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
    
    # 기본 설정 반환
    logger.info("기본 설정 사용")
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG

def save_config(config):
    """설정 파일 저장"""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("설정 파일 저장 완료")
    except Exception as e:
        logger.error(f"설정 파일 저장 실패: {e}")

# 글로벌 변수
config = load_config()
running_processes = {}  # 실행 중인 프로세스 추적
current_category = None  # 현재 선택된 카테고리

def get_categories_list():
    """데이터 폴더에서 카테고리 목록 가져오기"""
    try:
        data_path = config.get("data_path", os.path.join(project_dir, "data"))
        categories = []
        
        if os.path.exists(data_path):
            for item in os.listdir(data_path):
                item_path = os.path.join(data_path, item)
                if os.path.isdir(item_path):
                    categories.append(item)
        
        if not categories:
            logger.warning("카테고리를 찾을 수 없습니다.")
        
        return categories
    except Exception as e:
        logger.error(f"카테고리 목록 로드 중 오류: {e}")
        return []

def create_category_structure(category_name):
    """카테고리 폴더 구조 생성"""
    try:
        data_path = config.get("data_path", os.path.join(project_dir, "data"))
        category_path = os.path.join(data_path, category_name)
        
        # 기본 폴더 구조
        folders = [
            "1.images",
            "2.support-set",
            "3.box",
            "4.mask",
            "5.dataset",
            "6.preprocessed",
            "7.results",
            "8.refine-dataset"
        ]
        
        # 카테고리 폴더 생성
        os.makedirs(category_path, exist_ok=True)
        
        # 하위 폴더 생성
        for folder in folders:
            os.makedirs(os.path.join(category_path, folder), exist_ok=True)
        
        logger.info(f"카테고리 구조 생성 완료: {category_name}")
        return True, f"카테고리 '{category_name}' 구조가 생성되었습니다."
    except Exception as e:
        logger.error(f"카테고리 구조 생성 실패: {e}")
        return False, f"카테고리 구조 생성 중 오류 발생: {str(e)}"

def update_mask_config(save_png=None):
    """마스크 저장 설정 업데이트"""
    global config
    
    if save_png is not None:
        config["save_mask_png"] = save_png
    
    save_config(config)
    return config["save_mask_png"]

def update_target_size(width, height):
    """대상 이미지 크기 설정 업데이트"""
    global config
    
    try:
        width = int(width)
        height = int(height)
        
        if width <= 0 or height <= 0:
            return False, "너비와 높이는 양수여야 합니다."
        
        config["target_size"] = [width, height]
        save_config(config)
        
        return True, f"대상 이미지 크기가 {width}x{height}로 설정되었습니다."
    except Exception as e:
        logger.error(f"대상 크기 업데이트 실패: {e}")
        return False, f"대상 크기 업데이트 중 오류 발생: {str(e)}"

def get_category_stats(category_name):
    """카테고리 통계 가져오기"""
    try:
        data_path = config.get("data_path", os.path.join(project_dir, "data"))
        category_path = os.path.join(data_path, category_name)
        
        stats = {
            "category": category_name,
            "folders": {}
        }
        
        # 각 폴더 통계
        folders = [
            "1.images",
            "2.support-set",
            "3.box",
            "4.mask",
            "5.dataset",
            "6.preprocessed",
            "7.results",
            "8.refine-dataset"
        ]
        
        for folder in folders:
            folder_path = os.path.join(category_path, folder)
            if os.path.exists(folder_path):
                stats["folders"][folder] = {
                    "exists": True,
                    "count": len(os.listdir(folder_path))
                }
                
                # 특정 폴더에 대한 추가 정보
                if folder == "1.images":
                    image_count = len([f for f in os.listdir(folder_path) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    stats["folders"][folder]["image_count"] = image_count
                
                elif folder == "4.mask":
                    # mask_utils에서 통계 가져오기
                    try:
                        mask_stats = get_category_mask_stats(category_name)
                        stats["folders"][folder]["mask_stats"] = mask_stats
                    except:
                        pass
                
                elif folder == "5.dataset":
                    # 데이터셋 YAML 파일 찾기
                    yaml_files = glob.glob(os.path.join(folder_path, "*.yaml"))
                    yaml_files.extend(glob.glob(os.path.join(folder_path, "*.yml")))
                    stats["folders"][folder]["has_yaml"] = len(yaml_files) > 0
                
                elif folder == "7.results":
                    # 결과 폴더의 샷 수 카운트
                    shot_folders = [d for d in os.listdir(folder_path) 
                                   if os.path.isdir(os.path.join(folder_path, d))]
                    stats["folders"][folder]["shot_folders"] = shot_folders
                    
                    # 각 샷 폴더의 결과 파일 확인
                    for shot in shot_folders:
                        shot_path = os.path.join(folder_path, shot)
                        results_file = os.path.join(shot_path, "results.json")
                        if os.path.exists(results_file):
                            try:
                                with open(results_file, 'r') as f:
                                    results_data = json.load(f)
                                stats["folders"][folder][shot] = {
                                    "has_results": True,
                                    "count": len(results_data)
                                }
                            except:
                                stats["folders"][folder][shot] = {
                                    "has_results": True,
                                    "count": 0
                                }
            else:
                stats["folders"][folder] = {
                    "exists": False,
                    "count": 0
                }
        
        return stats
    except Exception as e:
        logger.error(f"카테고리 통계 로드 실패: {e}")
        return {"error": str(e)}

def run_main_launcher(category, det_model=None, seg_model=None, progress=None, use_mask_png=None, batch_size=100, max_images=10000, test_mode=False, custom_args=None):
    """main_launcher 실행"""
    global config, running_processes, current_category
    
    # 카테고리 설정
    current_category = category
    
    try:
        # 진행 상황을 위한 초기화
        if progress:
            progress(0, f"{category} 카테고리 처리 시작")
        
        # 실행 중인 프로세스가 있는지 확인
        process_key = f"main_launcher_{category}"
        if process_key in running_processes:
            return False, f"이미 {category} 카테고리에 대한 처리가 진행 중입니다."
        
        # 모델 설정
        if det_model is None:
            det_model = config["models"].get("detection", "grounding-dino-2")
        if seg_model is None:
            seg_model = config["models"].get("segmentation", "sam-2")
        
        # 마스크 PNG 저장 설정
        if use_mask_png is not None:
            update_mask_config(use_mask_png)
        
        # main_launcher 설정
        data_path = config.get("data_path", os.path.join(project_dir, "data"))
        save_mask_png = config.get("save_mask_png", False)
        target_size = config.get("target_size", [640, 640])
        
        # 명령행 인수 구성
        cmd_args = [
            f"--category={category}",
            f"--batch-size={batch_size}",
            f"--max-images={max_images}",
        ]
        
        # PNG 마스크 설정
        if not save_mask_png:
            cmd_args.append("--no-mask-png")
            
        # 테스트 모드
        if test_mode:
            cmd_args.append("--test-mode")
            
        # 사용자 정의 인수 추가
        if custom_args:
            cmd_args.extend(custom_args)
        
        # 스레드로 실행
        def run_process():
            try:
                logger.info(f"카테고리 {category}에 대한 main_launcher 실행 시작")
                logger.info(f"Detection 모델: {det_model}, Segmentation 모델: {seg_model}")
                logger.info(f"PNG 마스크 저장: {save_mask_png}")
                logger.info(f"배치 크기: {batch_size}, 최대 이미지: {max_images}")
                logger.info(f"실행 인수: {cmd_args}")
                
                # 명령행 인수 파싱
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--category", type=str)
                parser.add_argument("--batch-size", type=int)
                parser.add_argument("--max-images", type=int)
                parser.add_argument("--no-mask-png", action="store_true")
                parser.add_argument("--test-mode", action="store_true")
                # 추가 인수 처리를 위한 파서 확장
                args, unknown = parser.parse_known_args(cmd_args)
                
                # main_launcher 호출
                import scripts.main_launcher as launcher
                result = launcher.run_main_launcher(
                    category=category,
                    det_model=det_model,
                    seg_model=seg_model,
                    save_mask_png=save_mask_png,
                    batch_size=batch_size,
                    max_images=max_images,
                    test_mode=test_mode,
                    cmd_args=cmd_args
                )
                
                # 완료 후 상태 업데이트
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(1.0, f"{category} 카테고리 처리 완료")
                
                logger.info(f"{category} 카테고리 처리 완료")
                return result
            except Exception as e:
                logger.error(f"main_launcher 실행 중 오류: {e}")
                logger.exception("상세 오류:")
                
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(-1, f"오류 발생: {str(e)}")
                
                return False, f"실행 중 오류 발생: {str(e)}"
        
        # 스레드 시작
        process_thread = threading.Thread(target=run_process)
        process_thread.daemon = True
        process_thread.start()
        
        # 진행 중인 프로세스 추가
        running_processes[process_key] = {
            "thread": process_thread,
            "category": category,
            "start_time": time.time(),
            "det_model": det_model,
            "seg_model": seg_model
        }
        
        return True, f"{category} 카테고리 처리 시작됨 (Detection: {det_model}, Segmentation: {seg_model})"
    except Exception as e:
        logger.error(f"main_launcher 실행 실패: {e}")
        logger.exception("상세 오류:")
        
        if progress:
            progress(-1, f"오류 발생: {str(e)}")
        
        return False, f"처리 시작 중 오류 발생: {str(e)}"

def cancel_process(process_key):
    """실행 중인 프로세스 취소"""
    global running_processes
    
    try:
        if process_key in running_processes:
            # 스레드는 직접 종료할 수 없으므로 플래그로 표시
            running_processes[process_key]["cancel"] = True
            
            # 프로세스 목록에서 제거
            del running_processes[process_key]
            
            return True, f"프로세스 {process_key} 취소 요청됨"
        else:
            return False, f"프로세스 {process_key}를 찾을 수 없음"
    except Exception as e:
        logger.error(f"프로세스 취소 실패: {e}")
        return False, f"프로세스 취소 중 오류 발생: {str(e)}"

def get_running_processes():
    """현재 실행 중인 프로세스 목록"""
    global running_processes
    
    result = {}
    current_time = time.time()
    
    for key, process in running_processes.items():
        elapsed = current_time - process.get("start_time", current_time)
        result[key] = {
            "category": process.get("category", "unknown"),
            "elapsed_seconds": elapsed,
            "elapsed_formatted": f"{int(elapsed // 60)}분 {int(elapsed % 60)}초"
        }
    
    return result

def run_resize_dataset(category, progress=None):
    """dataset 이미지 리사이징 실행"""
    global config, running_processes
    
    try:
        # 진행 상황을 위한 초기화
        if progress:
            progress(0, f"{category} 카테고리 데이터셋 리사이징 시작")
        
        # 실행 중인 프로세스가 있는지 확인
        process_key = f"resize_dataset_{category}"
        if process_key in running_processes:
            return False, f"이미 {category} 카테고리에 대한 처리가 진행 중입니다."
        
        # 대상 크기 설정
        target_size = tuple(config.get("target_size", [640, 640]))
        
        # 스레드로 실행
        def run_process():
            try:
                # 데이터셋 리사이징 실행
                process_category_dataset(category, target_size, backup=True)
                
                # 완료 후 상태 업데이트
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(1.0, f"{category} 카테고리 데이터셋 리사이징 완료")
                
                logger.info(f"{category} 카테고리 데이터셋 리사이징 완료")
            except Exception as e:
                logger.error(f"데이터셋 리사이징 실행 중 오류: {e}")
                
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(-1, f"오류 발생: {str(e)}")
        
        # 스레드 시작
        process_thread = threading.Thread(target=run_process)
        process_thread.daemon = True
        process_thread.start()
        
        # 진행 중인 프로세스 추가
        running_processes[process_key] = {
            "thread": process_thread,
            "category": category,
            "start_time": time.time()
        }
        
        return True, f"{category} 카테고리 데이터셋 리사이징 시작됨"
    except Exception as e:
        logger.error(f"데이터셋 리사이징 실행 실패: {e}")
        
        if progress:
            progress(-1, f"오류 발생: {str(e)}")
        
        return False, f"데이터셋 리사이징 시작 중 오류 발생: {str(e)}"

def run_high_res_conversion(category, progress=None):
    """few-shot 결과를 고해상도 데이터셋으로 변환"""
    global config, running_processes
    
    try:
        # 진행 상황을 위한 초기화
        if progress:
            progress(0, f"{category} 카테고리 고해상도 변환 시작")
        
        # 실행 중인 프로세스가 있는지 확인
        process_key = f"high_res_{category}"
        if process_key in running_processes:
            return False, f"이미 {category} 카테고리에 대한 처리가 진행 중입니다."
        
        # 대상 크기 설정
        target_size = tuple(config.get("target_size", [640, 640]))
        
        # 스레드로 실행
        def run_process():
            try:
                # 데이터 경로
                data_path = config.get("data_path", os.path.join(project_dir, "data"))
                category_path = os.path.join(data_path, category)
                output_dir = os.path.join(category_path, "8.refine-dataset", "high_res")
                
                # 고해상도 변환 실행
                create_high_res_from_annotations(category_path, output_dir, target_size)
                
                # 완료 후 상태 업데이트
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(1.0, f"{category} 카테고리 고해상도 변환 완료")
                
                logger.info(f"{category} 카테고리 고해상도 변환 완료")
            except Exception as e:
                logger.error(f"고해상도 변환 실행 중 오류: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(-1, f"오류 발생: {str(e)}")
        
        # 스레드 시작
        process_thread = threading.Thread(target=run_process)
        process_thread.daemon = True
        process_thread.start()
        
        # 진행 중인 프로세스 추가
        running_processes[process_key] = {
            "thread": process_thread,
            "category": category,
            "start_time": time.time()
        }
        
        return True, f"{category} 카테고리 고해상도 변환 시작됨"
    except Exception as e:
        logger.error(f"고해상도 변환 실행 실패: {e}")
        
        if progress:
            progress(-1, f"오류 발생: {str(e)}")
        
        return False, f"고해상도 변환 시작 중 오류 발생: {str(e)}"

def run_refine_dataset(category, ground_truth_path=None, confidence=None, val_split=None, progress=None):
    """refine 데이터셋 생성"""
    global config, running_processes
    
    try:
        # 진행 상황을 위한 초기화
        if progress:
            progress(0, f"{category} 카테고리 Refine 데이터셋 생성 시작")
        
        # 실행 중인 프로세스가 있는지 확인
        process_key = f"refine_dataset_{category}"
        if process_key in running_processes:
            return False, f"이미 {category} 카테고리에 대한 처리가 진행 중입니다."
        
        # 설정
        data_path = config.get("data_path", os.path.join(project_dir, "data"))
        category_path = os.path.join(data_path, category)
        target_size = tuple(config.get("target_size", [640, 640]))
        
        # 신뢰도 임계값
        if confidence is None:
            confidence = config.get("confidence_threshold", 0.5)
        else:
            confidence = float(confidence)
            
        # 검증 세트 비율
        if val_split is None:
            val_split = config.get("val_split", 0.2)
        else:
            val_split = float(val_split)
        
        # 스레드로 실행
        def run_process():
            try:
                # 출력 디렉토리
                output_dir = os.path.join(category_path, "8.refine-dataset", "yolo_dataset")
                
                # Refine 데이터셋 생성
                create_refine_dataset(
                    category_path=category_path,
                    output_dir=output_dir,
                    target_size=target_size,
                    ground_truth_annotations=ground_truth_path,
                    confidence_threshold=confidence,
                    val_split=val_split
                )
                
                # 완료 후 상태 업데이트
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(1.0, f"{category} 카테고리 Refine 데이터셋 생성 완료")
                
                logger.info(f"{category} 카테고리 Refine 데이터셋 생성 완료")
            except Exception as e:
                logger.error(f"Refine 데이터셋 생성 중 오류: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                if process_key in running_processes:
                    del running_processes[process_key]
                
                if progress:
                    progress(-1, f"오류 발생: {str(e)}")
        
        # 스레드 시작
        process_thread = threading.Thread(target=run_process)
        process_thread.daemon = True
        process_thread.start()
        
        # 진행 중인 프로세스 추가
        running_processes[process_key] = {
            "thread": process_thread,
            "category": category,
            "start_time": time.time()
        }
        
        return True, f"{category} 카테고리 Refine 데이터셋 생성 시작됨"
    except Exception as e:
        logger.error(f"Refine 데이터셋 생성 실패: {e}")
        
        if progress:
            progress(-1, f"오류 발생: {str(e)}")
        
        return False, f"Refine 데이터셋 생성 시작 중 오류 발생: {str(e)}"

def check_running_processes():
    """현재 실행 중인 프로세스 정보 확인"""
    if not running_processes:
        return "실행 중인 프로세스가 없습니다."
    
    info = []
    for key, process in running_processes.items():
        start_time = process.get("start_time", 0)
        elapsed = time.time() - start_time
        
        # 프로세스 정보 형식화
        process_type = key.split("_")[0] if "_" in key else "unknown"
        category = process.get("category", "N/A")
        
        if process_type == "main":
            det_model = process.get("det_model", "N/A")
            seg_model = process.get("seg_model", "N/A")
            info.append(f"- {category}: main_launcher ({det_model}/{seg_model}) - {elapsed:.1f}초 실행 중")
        elif process_type == "resize":
            info.append(f"- {category}: 이미지 리사이징 - {elapsed:.1f}초 실행 중")
        elif process_type == "convert":
            info.append(f"- {category}: 이미지 변환 - {elapsed:.1f}초 실행 중")
        elif process_type == "refine":
            info.append(f"- {category}: 데이터셋 정제 - {elapsed:.1f}초 실행 중")
        else:
            info.append(f"- {key}: {elapsed:.1f}초 실행 중")
    
    return "실행 중인 프로세스:\n" + "\n".join(info)

# main_launcher 직접 실행 함수 추가
def start_main_launcher(category, det_model, seg_model, save_png, batch_size=100, max_images=10000, test_mode=False):
    """웹 UI에서 main_launcher 시작"""
    if not category:
        return "카테고리를 선택해주세요."
    
    # 마스크 PNG 저장 설정 업데이트
    update_mask_config(save_png)
    
    # main_launcher 실행
    success, message = run_main_launcher(
        category=category,
        det_model=det_model,
        seg_model=seg_model,
        use_mask_png=save_png,
        batch_size=batch_size,
        max_images=max_images,
        test_mode=test_mode
    )
    
    if success:
        return f"{message}\n실행 로그는 프로세스 관리 탭에서 확인하세요."
    else:
        return f"오류: {message}"

# Gradio 웹 UI 생성
def create_web_ui():
    with gr.Blocks(title="Few Shot Learning & 데이터셋 관리 도구") as app:
        gr.Markdown("# Few Shot Learning & 데이터셋 관리 도구")
        
        with gr.Tabs():
            # 카테고리 관리 탭
            with gr.Tab("카테고리 관리"):
                with gr.Row():
                    with gr.Column(scale=2):
                        category_dropdown = gr.Dropdown(
                            choices=get_categories_list(),
                            label="카테고리 선택",
                            interactive=True
                        )
                        refresh_btn = gr.Button("새로고침")
                        
                        refresh_btn.click(
                            fn=lambda: update_category_dropdown(),
                            outputs=category_dropdown
                        )
                    
                    with gr.Column(scale=1):
                        new_category_input = gr.Textbox(label="새 카테고리 이름")
                        create_category_btn = gr.Button("카테고리 생성")
                        
                        create_result = gr.Textbox(label="결과")
                        
                        create_category_btn.click(
                            fn=create_new_category,
                            inputs=new_category_input,
                            outputs=[create_result, category_dropdown]
                        )
                
                with gr.Row():
                    category_info = gr.Textbox(label="카테고리 정보", lines=10)
                    sample_image = gr.Image(label="샘플 이미지", show_label=True)
                
                # 카테고리 선택 시 정보 업데이트
                category_dropdown.change(
                    fn=on_category_select,
                    inputs=category_dropdown,
                    outputs=[category_info, sample_image]
                )
            
            # 설정 탭
            with gr.Tab("설정"):
                with gr.Row():
                    with gr.Column():
                        save_mask_png_checkbox = gr.Checkbox(
                            label="마스크 PNG 이미지 저장",
                            value=config.get("save_mask_png", False)
                        )
                        save_mask_result = gr.Textbox(label="결과")
                        
                        save_mask_png_checkbox.change(
                            fn=toggle_mask_png,
                            inputs=save_mask_png_checkbox,
                            outputs=save_mask_result
                        )
                    
                    with gr.Column():
                        target_width = gr.Number(
                            label="대상 이미지 너비",
                            value=config.get("target_size", [640, 640])[0],
                            precision=0
                        )
                        target_height = gr.Number(
                            label="대상 이미지 높이",
                            value=config.get("target_size", [640, 640])[1],
                            precision=0
                        )
                        target_size_btn = gr.Button("크기 설정")
                        target_size_result = gr.Textbox(label="결과")
                        
                        target_size_btn.click(
                            fn=set_target_size,
                            inputs=[target_width, target_height],
                            outputs=target_size_result
                        )
            
            # main_launcher 탭
            with gr.Tab("이미지 처리 (main_launcher)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        launcher_category = gr.Dropdown(
                            choices=get_categories_list(),
                            label="카테고리 선택",
                            interactive=True
                        )
                        
                        with gr.Row():
                            det_model = gr.Dropdown(
                                choices=["grounding-dino-2", "grounding-dino", "yolo", "yolov8"],
                                label="Detection 모델",
                                value="grounding-dino-2"
                            )
                            seg_model = gr.Dropdown(
                                choices=["sam-2", "sam", "mobile-sam"],
                                label="Segmentation 모델",
                                value="sam-2"
                            )
                        
                        with gr.Row():
                            save_png = gr.Checkbox(
                                label="마스크 PNG 이미지 저장",
                                value=config.get("save_mask_png", False)
                            )
                            test_mode = gr.Checkbox(
                                label="테스트 모드 (이미지 수 제한)",
                                value=False
                            )
                        
                        with gr.Row():
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=500,
                                value=100,
                                step=10,
                                label="배치 크기",
                                info="한 번에 처리할 이미지 수"
                            )
                            max_images = gr.Slider(
                                minimum=1,
                                maximum=50000,
                                value=10000,
                                step=100,
                                label="최대 이미지 수",
                                info="처리할 최대 이미지 수"
                            )
                        
                        run_launcher_btn = gr.Button("실행", variant="primary")
                        launcher_result = gr.Textbox(label="결과", lines=5)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 프로세스 정보")
                        launcher_process_info = gr.Textbox(label="현재 실행 상태", lines=5)
                        refresh_process_btn = gr.Button("새로고침")
                        
                        gr.Markdown("### 고급 옵션")
                        with gr.Accordion("고급 옵션", open=False):
                            memory_monitor = gr.Checkbox(
                                label="메모리 모니터링",
                                value=False,
                                info="메모리 사용량을 모니터링합니다 (psutil 필요)"
                            )
                            custom_args = gr.Textbox(
                                label="추가 명령행 인수",
                                placeholder="--verbose --debug",
                                info="추가 명령행 인수를 입력하세요 (공백으로 구분)"
                            )
                
                # 이벤트 핸들러 연결
                run_launcher_btn.click(
                    fn=start_main_launcher,
                    inputs=[
                        launcher_category, 
                        det_model, 
                        seg_model, 
                        save_png, 
                        batch_size, 
                        max_images, 
                        test_mode
                    ],
                    outputs=launcher_result
                )
                
                refresh_process_btn.click(
                    fn=check_running_processes,
                    outputs=launcher_process_info
                )
                
                # 초기 로드 시 프로세스 정보 자동 업데이트
                launcher_category.change(
                    fn=lambda: None,
                    inputs=None,
                    outputs=None,
                    _js="() => {setTimeout(() => document.getElementById('refresh_process_btn').click(), 100)}"
                )
            
            # 데이터셋 관리 탭
            with gr.Tab("데이터셋 관리"):
                with gr.Accordion("데이터셋 리사이징", open=True):
                    with gr.Row():
                        resize_category = gr.Dropdown(
                            choices=get_categories_list(),
                            label="카테고리 선택",
                            interactive=True
                        )
                        
                        resize_btn = gr.Button("데이터셋 리사이징 실행")
                        resize_result = gr.Textbox(label="결과")
                        
                        resize_btn.click(
                            fn=start_resize_dataset,
                            inputs=resize_category,
                            outputs=resize_result
                        )
                
                with gr.Accordion("고해상도 변환", open=True):
                    with gr.Row():
                        highres_category = gr.Dropdown(
                            choices=get_categories_list(),
                            label="카테고리 선택",
                            interactive=True
                        )
                        
                        highres_btn = gr.Button("고해상도 변환 실행")
                        highres_result = gr.Textbox(label="결과")
                        
                        highres_btn.click(
                            fn=start_high_res_conversion,
                            inputs=highres_category,
                            outputs=highres_result
                        )
                
                with gr.Accordion("Refine 데이터셋 생성", open=True):
                    with gr.Row():
                        refine_category = gr.Dropdown(
                            choices=get_categories_list(),
                            label="카테고리 선택",
                            interactive=True
                        )
                        
                        ground_truth_path = gr.Textbox(
                            label="Ground Truth 경로 (선택 사항)",
                            placeholder="Ground Truth 파일 또는 디렉토리 경로"
                        )
                    
                    with gr.Row():
                        confidence_threshold = gr.Number(
                            label="신뢰도 임계값 (0~1)",
                            value=config.get("confidence_threshold", 0.5),
                            minimum=0,
                            maximum=1
                        )
                        
                        val_split = gr.Number(
                            label="검증 세트 비율 (0~1)",
                            value=config.get("val_split", 0.2),
                            minimum=0,
                            maximum=1
                        )
                    
                    refine_btn = gr.Button("Refine 데이터셋 생성")
                    refine_result = gr.Textbox(label="결과")
                    
                    refine_btn.click(
                        fn=start_refine_dataset,
                        inputs=[refine_category, ground_truth_path, confidence_threshold, val_split],
                        outputs=refine_result
                    )
            
            # 프로세스 관리 탭
            with gr.Tab("프로세스 관리"):
                with gr.Row():
                    processes_info = gr.Textbox(label="실행 중인 프로세스", lines=5)
                    check_processes_btn = gr.Button("새로고침")
                    
                    check_processes_btn.click(
                        fn=check_running_processes,
                        outputs=processes_info
                    )
                
                with gr.Row():
                    cancel_process_input = gr.Textbox(
                        label="취소할 프로세스 키",
                        placeholder="예: main_launcher_test_category"
                    )
                    cancel_process_btn = gr.Button("프로세스 취소")
                    cancel_process_result = gr.Textbox(label="결과")
                    
                    cancel_process_btn.click(
                        fn=cancel_running_process,
                        inputs=cancel_process_input,
                        outputs=cancel_process_result
                    )
            
            # Ground Truth Labeling 탭 추가
            with gr.Tab("Ground Truth Labeling"):
                if GROUND_TRUTH_LABELER_AVAILABLE:
                    # Ground Truth Labeler UI 통합
                    labeler = GroundTruthLabeler()
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # 카테고리 및 실험 선택
                            category_dropdown = gr.Dropdown(
                                choices=labeler.get_categories(),
                                label="카테고리 선택",
                                interactive=True
                            )
                            experiment_dropdown = gr.Dropdown(
                                choices=[],
                                label="실험 선택",
                                interactive=True
                            )
                            load_btn = gr.Button("실험 로드")
                            load_result = gr.Textbox(label="상태", lines=2)
                            
                            # Ground Truth 관리
                            gr.Markdown("### Ground Truth 관리")
                            set_gt_dropdown = gr.Dropdown(
                                choices=[],
                                label="Ground Truth 기준으로 설정",
                                interactive=True
                            )
                            set_gt_btn = gr.Button("Ground Truth로 설정")
                            load_gt_btn = gr.Button("기존 Ground Truth 로드")
                            save_gt_btn = gr.Button("현재 레이블을 Ground Truth로 저장")
                            gt_result = gr.Textbox(label="Ground Truth 상태", lines=2)
                            
                            # 필터
                            gr.Markdown("### 필터")
                            filter_class = gr.Dropdown(
                                choices=["all"] + labeler.class_list,
                                label="클래스별 필터",
                                value="all",
                                interactive=True
                            )
                            filter_confidence = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.05,
                                label="최소 신뢰도"
                            )
                            filter_modified = gr.Checkbox(
                                label="수정된 항목만 표시",
                                value=False
                            )
                            apply_filter_btn = gr.Button("필터 적용")
                            
                            # 통계
                            gr.Markdown("### 통계")
                            stats_text = gr.Textbox(label="통계", lines=6)
                            refresh_stats_btn = gr.Button("통계 새로고침")
                            export_metrics_btn = gr.Button("성능 지표 내보내기")
                            
                        with gr.Column(scale=3):
                            # 메인 레이블링 인터페이스
                            gr.Markdown("### 이미지 갤러리")
                            
                            # 페이지 이동 컨트롤
                            with gr.Row():
                                prev_btn = gr.Button("← 이전 페이지")
                                page_info = gr.Textbox(label="페이지", value="0 / 0")
                                next_btn = gr.Button("다음 페이지 →")
                            
                            # 일괄 작업
                            with gr.Row():
                                batch_class = gr.Dropdown(
                                    choices=labeler.class_list,
                                    label="선택한 항목을 다음으로 지정",
                                    interactive=True
                                )
                                apply_batch_btn = gr.Button("선택한 항목에 적용")
                                clear_selection_btn = gr.Button("선택 초기화")
                            
                            # 이미지 갤러리
                            gallery = gr.Gallery(
                                label="이미지",
                                columns=5,
                                rows=5,
                                show_label=True,
                                object_fit="contain",
                                height="600px"
                            ).style(grid=5, height="600px")
                            
                            # 단일 이미지 보기
                            with gr.Row():
                                with gr.Column(scale=2):
                                    selected_image = gr.Image(label="선택한 이미지", type="filepath")
                                with gr.Column(scale=1):
                                    image_info = gr.Textbox(label="이미지 정보", lines=5)
                                    single_class = gr.Dropdown(
                                        choices=labeler.class_list,
                                        label="클래스 지정",
                                        interactive=True
                                    )
                                    apply_single_btn = gr.Button("적용")
                    
                    # 이벤트 핸들러 연결
                    def update_experiments(category):
                        experiments = labeler.get_experiments(category)
                        return gr.Dropdown(choices=experiments)
                    
                    def load_experiment_handler(category, experiment):
                        result = labeler.load_experiment(category, experiment)
                        
                        # 실험 드롭다운 업데이트
                        experiments = list(labeler.experiments.keys())
                        
                        # 클래스 드롭다운 업데이트
                        classes = ["all"] + labeler.class_list
                        
                        # 갤러리 업데이트
                        gallery_data = update_gallery()
                        
                        return [
                            result,
                            gr.Dropdown(choices=experiments),
                            gr.Dropdown(choices=labeler.class_list),
                            gr.Dropdown(choices=labeler.class_list),
                            gr.Dropdown(choices=classes, value="all"),
                            *gallery_data
                        ]
                    
                    def update_gallery():
                        # 현재 페이지 이미지 가져오기
                        current_images = labeler.get_current_page_images()
                        
                        # 갤러리 데이터 준비
                        gallery_images = []
                        for img_name in current_images:
                            img_path = labeler.get_image_path(img_name)
                            if os.path.exists(img_path):
                                label_info = labeler.labels.get(img_name, {})
                                current_class = label_info.get("current", "unknown")
                                confidence = label_info.get("confidence", 0.0)
                                is_modified = label_info.get("is_modified", False)
                                
                                # 레이블 텍스트 생성
                                modified_marker = "✏️" if is_modified else ""
                                selected_marker = "✓" if img_name in labeler.selected_images else ""
                                label = f"{img_name}\n{current_class} ({confidence:.2f})\n{modified_marker}{selected_marker}"
                                
                                gallery_images.append((str(img_path), label))
                        
                        # 페이지 정보 업데이트
                        page_text = f"{labeler.pagination['current_page'] + 1} / {max(1, labeler.pagination['total_pages'])}"
                        
                        return [gallery_images, page_text]
                    
                    def change_page_handler(direction):
                        labeler.change_page(direction)
                        return update_gallery()
                    
                    def select_image_handler(evt: gr.SelectData):
                        """이미지 선택 핸들러"""
                        if not labeler.images or not labeler.get_current_page_images():
                            return ["이미지를 선택하지 않음", None, ""]
                        
                        # 선택한 이미지 가져오기
                        try:
                            idx = evt.index
                            current_page_images = labeler.get_current_page_images()
                            if idx < len(current_page_images):
                                img_name = current_page_images[idx]
                                img_path = labeler.get_image_path(img_name)
                                
                                # 일괄 작업을 위한 선택 토글
                                labeler.toggle_image_selection(img_name)
                                
                                # 레이블 정보 가져오기
                                label_info = labeler.labels.get(img_name, {})
                                current_class = label_info.get("current", "unknown")
                                original_class = label_info.get("original", "unknown")
                                confidence = label_info.get("confidence", 0.0)
                                is_modified = label_info.get("is_modified", False)
                                
                                info_text = (
                                    f"파일명: {img_name}\n"
                                    f"현재 클래스: {current_class}\n"
                                    f"원본 클래스: {original_class}\n"
                                    f"신뢰도: {confidence:.4f}\n"
                                    f"수정됨: {'예' if is_modified else '아니오'}\n"
                                    f"선택됨: {'예' if img_name in labeler.selected_images else '아니오'}"
                                )
                                
                                # 갤러리 업데이트
                                gallery_data = update_gallery()
                                
                                return [info_text, str(img_path), current_class, *gallery_data]
                        except Exception as e:
                            logger.error(f"이미지 선택 중 오류: {e}")
                            return ["이미지 선택 중 오류", None, "", *update_gallery()]
                    
                    def update_single_label(img_path, new_class):
                        if not img_path:
                            return ["이미지를 선택하지 않음", *update_gallery()]
                        
                        img_name = os.path.basename(img_path)
                        result = labeler.update_label(img_name, new_class)
                        
                        # 레이블 변경 후 이미지 정보 업데이트
                        label_info = labeler.labels.get(img_name, {})
                        current_class = label_info.get("current", "unknown")
                        original_class = label_info.get("original", "unknown")
                        confidence = label_info.get("confidence", 0.0)
                        is_modified = label_info.get("is_modified", False)
                        
                        info_text = (
                            f"파일명: {img_name}\n"
                            f"현재 클래스: {current_class}\n"
                            f"원본 클래스: {original_class}\n"
                            f"신뢰도: {confidence:.4f}\n"
                            f"수정됨: {'예' if is_modified else '아니오'}\n"
                            f"선택됨: {'예' if img_name in labeler.selected_images else '아니오'}"
                        )
                        
                        return [result, info_text, *update_gallery()]
                    
                    def update_batch_labels(new_class):
                        result = labeler.update_selected_labels(new_class)
                        return [result, *update_gallery()]
                    
                    def clear_selection_handler():
                        result = labeler.clear_selection()
                        return [result, *update_gallery()]
                    
                    def ground_truth_handler(action, experiment=None):
                        if action == "set" and experiment:
                            result = labeler.set_as_ground_truth(experiment)
                        elif action == "load":
                            result = labeler.load_ground_truth(labeler.current_category)
                            result += "\n" + labeler.apply_ground_truth_to_labels()
                        elif action == "save":
                            result = labeler.save_ground_truth()
                        else:
                            result = "잘못된 작업"
                        
                        return [result, *update_gallery()]
                    
                    def update_statistics():
                        stats = labeler.get_statistics()
                        text = (
                            f"전체 이미지: {stats['total']}\n"
                            f"수정된 레이블: {stats['modified']} ({stats['modified']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%)\n\n"
                            "클래스별 분포:\n"
                        )
                        
                        for cls, count in stats['by_class'].items():
                            if count > 0:
                                text += f"- {cls}: {count} ({count/stats['total']*100 if stats['total'] > 0 else 0:.1f}%)\n"
                        
                        return text
                    
                    def export_metrics_handler():
                        metrics = labeler.export_metrics()
                        if isinstance(metrics, str):
                            return metrics
                        
                        # 지표를 텍스트로 포맷팅
                        text = "실험 성능 지표 (Ground Truth 기준):\n\n"
                        
                        for exp_name, exp_metrics in metrics.items():
                            acc = exp_metrics["accuracy"] * 100
                            text += f"{exp_name}: {acc:.2f}% 정확도 ({exp_metrics['correct']}/{exp_metrics['total']})\n"
                            
                            # 클래스별 정확도
                            text += "  클래스별 정확도:\n"
                            for cls, cls_acc in exp_metrics["class_accuracy"].items():
                                if cls_acc > 0:
                                    text += f"  - {cls}: {cls_acc*100:.2f}%\n"
                            text += "\n"
                        
                        return text
                    
                    # 이벤트 핸들러 연결
                    category_dropdown.change(
                        fn=update_experiments,
                        inputs=category_dropdown,
                        outputs=experiment_dropdown
                    )
                    
                    load_btn.click(
                        fn=load_experiment_handler,
                        inputs=[category_dropdown, experiment_dropdown],
                        outputs=[
                            load_result,
                            set_gt_dropdown,
                            batch_class,
                            single_class,
                            filter_class,
                            gallery,
                            page_info
                        ]
                    )
                    
                    prev_btn.click(
                        fn=lambda: change_page_handler("prev"),
                        outputs=[gallery, page_info]
                    )
                    
                    next_btn.click(
                        fn=lambda: change_page_handler("next"),
                        outputs=[gallery, page_info]
                    )
                    
                    apply_filter_btn.click(
                        fn=lambda: update_gallery(),
                        outputs=[gallery, page_info]
                    )
                    
                    filter_class.change(
                        fn=lambda value: (labeler.update_filter("class", value), *update_gallery())[1:],
                        inputs=filter_class,
                        outputs=[gallery, page_info]
                    )
                    
                    filter_confidence.change(
                        fn=lambda value: (labeler.update_filter("confidence", value), *update_gallery())[1:],
                        inputs=filter_confidence,
                        outputs=[gallery, page_info]
                    )
                    
                    filter_modified.change(
                        fn=lambda value: (labeler.update_filter("is_modified", value), *update_gallery())[1:],
                        inputs=filter_modified,
                        outputs=[gallery, page_info]
                    )
                    
                    gallery.select(
                        fn=select_image_handler,
                        outputs=[image_info, selected_image, single_class, gallery, page_info]
                    )
                    
                    apply_single_btn.click(
                        fn=update_single_label,
                        inputs=[selected_image, single_class],
                        outputs=[load_result, image_info, gallery, page_info]
                    )
                    
                    apply_batch_btn.click(
                        fn=update_batch_labels,
                        inputs=batch_class,
                        outputs=[load_result, gallery, page_info]
                    )
                    
                    clear_selection_btn.click(
                        fn=clear_selection_handler,
                        outputs=[load_result, gallery, page_info]
                    )
                    
                    set_gt_btn.click(
                        fn=lambda exp: ground_truth_handler("set", exp),
                        inputs=set_gt_dropdown,
                        outputs=[gt_result, gallery, page_info]
                    )
                    
                    load_gt_btn.click(
                        fn=lambda: ground_truth_handler("load"),
                        outputs=[gt_result, gallery, page_info]
                    )
                    
                    save_gt_btn.click(
                        fn=lambda: ground_truth_handler("save"),
                        outputs=[gt_result, gallery, page_info]
                    )
                    
                    refresh_stats_btn.click(
                        fn=update_statistics,
                        outputs=stats_text
                    )
                    
                    export_metrics_btn.click(
                        fn=export_metrics_handler,
                        outputs=stats_text
                    )
                    
                else:
                    gr.Markdown("""
                    ## Ground Truth Labeler 모듈을 사용할 수 없습니다
                    
                    이 기능을 사용하려면 `scripts/ground_truth_labeler.py` 모듈이 필요합니다.
                    해당 모듈이 존재하고 모든 의존성이 설치되어 있는지 확인하세요.
                    """)
        
        # 주기적으로 프로세스 정보 업데이트
        processes_info = gr.Textbox(label="실행 중인 프로세스 상태", lines=5)
        
        # 초기 로드 시 자동 업데이트
        app.load(
            fn=check_running_processes,
            outputs=processes_info
        )
    
    return app

# 웹앱 실행
def main():
    app = create_web_ui()
    app.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main() 