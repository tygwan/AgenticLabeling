import time
import datetime
import argparse
import os
import sys
import json
import numpy as np
import traceback
from pathlib import Path
from tqdm import tqdm  # 프로그레스 바를 위한 tqdm 추가

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Modified Packages Path Injection ---
# This ensures that the versions of packages in the 'modified_packages'
# directory are used instead of the system-installed ones.
# This must be at the very top, before any other project imports.
try:
    # Get the absolute path to the project's root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Construct the path to the modified packages directory
    modified_packages_path = os.path.join(project_root, 'modified_packages')
    # Prepend it to the system path
    sys.path.insert(0, modified_packages_path)
    print(f"[INFO] Injected modified packages path: {modified_packages_path}")
except Exception as e:
    print(f"[ERROR] Could not set up modified packages path: {e}")
# --- End of Path Injection ---

# 메모리 모니터링 기능 추가
try:
    import psutil
    
    def log_memory_usage(tag=""):
        """현재 프로세스의 메모리 사용량을 로깅합니다."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        log(f"메모리 사용량 [{tag}]: {memory_mb:.2f} MB", "INFO")
        return memory_mb
        
    MEMORY_MONITORING = True
except ImportError:
    log_memory_usage = lambda tag="": None
    MEMORY_MONITORING = False

# 로깅 레벨 설정
VERBOSE = False  # 상세 로그 출력 여부

# 상대 경로 임포트를 위한 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 명령줄 인자 파싱 추가
parser = argparse.ArgumentParser(description="AutoDistill Pipeline Runner")
parser.add_argument("--category", type=str, default="test_category", help="카테고리 이름 (기본값: test_category)")
parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
parser.add_argument("--verbose", action="store_true", help="상세 로그 출력 활성화")
parser.add_argument("--plot", action="store_true", help="결과 시각화 활성화")
parser.add_argument("--save-npy", action="store_true", default=False, help="NPY 마스크 파일 저장 (기본: False)")
parser.add_argument("--preprocess", action="store_true", help="이미지 전처리 작업 수행")
parser.add_argument("--advanced-preprocess", action="store_true", help="고품질 이미지 전처리 작업 수행 (RLE 마스크 기반)")
parser.add_argument("--target-size", type=str, default="224,224", help="전처리 이미지 크기 (너비,높이 형식, 기본값: 640,640)")
parser.add_argument("--no-crop", action="store_true", help="전처리 시 객체 크롭 비활성화")
parser.add_argument("--no-mask", action="store_true", help="전처리 시 마스크 적용 비활성화")
parser.add_argument("--preprocess-class-id", type=int, default=None, help="전처리할 특정 클래스 ID (0-3)")
parser.add_argument("--prepare-classify", action="store_true", help="분류를 위한 디렉토리 구조 생성")
parser.add_argument("--classification-methods", type=str, default="method1,method2,method3,method4", 
                    help="분류 방법 목록 (쉼표로 구분, 기본값: method1,method2,method3,method4)")
parser.add_argument("--memory-monitor", action="store_true", help="메모리 사용량 모니터링 활성화")
parser.add_argument("--batch-size", type=int, default=100, help="이미지 처리 배치 크기 (기본값: 10)")
parser.add_argument("--max-images", type=int, default=10000, help="처리할 최대 이미지 수 (기본값: 100)")
parser.add_argument("--test-mode", action="store_true", help="테스트 모드: 클래스별 30장 정도만 처리")
parser.add_argument("--mask-only", action="store_true", help="마스크 데이터가 있는 이미지만 처리 (WARNING 메시지 제거 및 성능 최적화)")
parser.add_argument("--test-images", type=int, default=None, help="테스트용 이미지 수 제한 (기본값: 제한 없음)")
parser.add_argument("--debug-format", action="store_true", help="디버그 형식으로 출력 (A_1_1_frame_0001.txt, A_1_1_frame_0001.png_box_points.txt 형태)")
parser.add_argument("--visualize", action="store_true", help="마스크에서 실제 객체 경계를 따라 폴리곤 추출 시각화")
parser.add_argument("--no-mask-png", action="store_true", help="마스크 PNG 이미지 생성 비활성화")
args = parser.parse_args()

# 디버그 모드 설정
DEBUG = args.debug
VERBOSE = args.verbose
PLOT = args.plot

    
# 배치 처리 설정
BATCH_SIZE = args.batch_size
MAX_IMAGES = args.max_images

# 저장 옵션 설정 (선택적으로 NPY 저장)
SAVE_NPY_FILES = args.save_npy

# 로그 출력 함수
def log(message, level="INFO"):
    """레벨에 따른 로그 출력"""
    if level == "DEBUG" and not DEBUG:
        return
    if level == "VERBOSE" and not VERBOSE:
        return
    print(f"[{level}] {message}")


# 메모리 모니터링 설정
if args.memory_monitor and not MEMORY_MONITORING:
    log("메모리 모니터링을 활성화했지만 psutil 라이브러리가 설치되지 않았습니다.", "WARNING")
    log("pip install psutil 명령으로 필요한 라이브러리를 설치하세요.", "WARNING")
elif args.memory_monitor:
    log("메모리 모니터링이 활성화되었습니다.", "INFO")
    MEMORY_MONITORING = True
else:
    MEMORY_MONITORING = False

# 전처리 관련 옵션 처리
target_size = None
if args.target_size:
    try:
        width, height = map(int, args.target_size.split(','))
        target_size = (width, height)
        log(f"전처리 대상 크기: {target_size}", "INFO")
    except ValueError:
        log(f"잘못된 타겟 크기 형식: {args.target_size}. 기본값 (640,640)을 사용합니다.", "WARNING")
        target_size = (640, 640)


# 프로젝트 루트 디렉토리 설정
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

start = time.time() # 시작

# 초기 메모리 사용량 기록
if MEMORY_MONITORING:
    initial_memory = log_memory_usage("시작")

# 라이브러리 로딩 중임을 표시
log("필요한 라이브러리 로드 중...", "VERBOSE")

# 사용자 정의 헬퍼 모듈 import 및 패치 적용 (GroundedSAM2 import 전에 수행)
try:
    from scripts.custom_helpers import patch_grounded_sam2
    patch_successful = patch_grounded_sam2()
    if not patch_successful:
        log("WARNING: 프로젝트 내 디렉토리를 사용한 SAM 모델 로딩 패치 실패", "WARNING")
        log("기본 ~/.cache 경로가 사용될 수 있습니다.", "WARNING")
except Exception as e:
    log(f"패치 적용 중 오류: {e}", "ERROR")
    if DEBUG:
        traceback.print_exc()

# 필요한 라이브러리 import
try:
    # 라이브러리 로딩을 위한 프로그레스 바 표시
    libraries = ["autodistill_grounded_sam_2", "autodistill_florence_2", "autodistill.detection", 
                 "autodistill.utils", "autodistill_yolov8", "cv2"]
    
    for lib in tqdm(libraries, desc="라이브러리 로드 중", disable=not VERBOSE):
        if lib == "autodistill_grounded_sam_2":
            log("GroundedSAM2 로드 중...", "VERBOSE")
            from autodistill_grounded_sam_2 import GroundedSAM2
        elif lib == "autodistill_florence_2":
            log("Florence2 로드 중...", "VERBOSE")
            from autodistill_florence_2 import Florence2
        elif lib == "autodistill.detection":
            log("CaptionOntology 로드 중...", "VERBOSE")
            from autodistill.detection import CaptionOntology
        elif lib == "autodistill.utils":
            log("plot 유틸리티 로드 중...", "VERBOSE")
            from autodistill.utils import plot
        elif lib == "autodistill_yolov8":
            log("YOLOv8 로드 중...", "VERBOSE")
            from autodistill_yolov8 import YOLOv8
        elif lib == "cv2":
            log("OpenCV 로드 중...", "VERBOSE")
            import cv2
except Exception as e:
    log(f"라이브러리 임포트 중 오류: {e}", "ERROR")
    if DEBUG:
        traceback.print_exc()
    sys.exit(1)

# 마스크 유틸리티 모듈 임포트
log("마스크 유틸리티 모듈 로드 중...", "VERBOSE")
from scripts.mask_utils import convert_and_save_mask, save_coords_without_mask, convert_and_save_mask_debug_format, save_coords_without_mask_debug_format, save_mask_as_polygon, save_masks_as_yolo_format, save_mask_as_raw

# 카테고리 설정
category = args.category
log(f"실행 카테고리: {category}")

# 디렉토리 생성 - 수정된 구조 반영
try:
    log("디렉토리 구조 생성 중...", "VERBOSE")
    directories = [
        data_dir / category / "1.images",
        data_dir / category / "2.support-set",
        data_dir / category / "3.box",
        data_dir / category / "4.mask",
        data_dir / category / "5.dataset",
        data_dir / category / "6.preprocessed",
        data_dir / category / "7.results",
        data_dir / category / "8.refine-dataset"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        log(f"디렉토리 생성: {directory}", "DEBUG")
except Exception as e:
    log(f"디렉토리 생성 중 오류: {e}", "ERROR")
    if DEBUG:
        traceback.print_exc()

try:
    log("온톨로지 정의 중...", "VERBOSE")
    # define an ontology to map class names to our Grounded SAM 2 prompt
    # the ontology dictionary has the format {caption: class}
    # where caption is the prompt sent to the base model, and class is the label that will
    # be saved for that caption in the generated annotations
    # then, load the model
    
    # 메모리 최적화: 모델 로드 전 메모리 사용량 로깅
    if MEMORY_MONITORING:
        log_memory_usage("모델 로드 전")
    
    # 샘플 추론 배치 크기 조정 (메모리 사용량을 줄이기 위해)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 큰 텐서를 더 작은 청크로 분할
    
    # 모델 로드
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                "What blue, fabric-like barrier, explicitly designed to shield or guide pedestrians near construction zones, is visible in this image?": "fence_person",
                "What type of pedestrian pathway, found alongside roads or within construction zones, composed of materials like nonwoven fabric, sand, bricks, or asphalt, is visible in this image?": "sidewalk",
                "What motorized vehicle, designed for passenger or cargo transport, commonly seen in urban or road environments, is visible in this image?": "car",
                "What small, brightly colored cone-shaped object, specifically designed to redirect traffic or highlight construction hazards, is visible in this image?": "traffic cone",
            }
        )
    )
    
    # 메모리 최적화: 모델 로드 후 메모리 사용량 로깅
    if MEMORY_MONITORING:
        log_memory_usage("모델 로드 후")

    # 경로 설정 변경
    image_directory = str(data_dir / category / "1.images")
    box_directory = str(data_dir / category / "3.box")
    mask_directory = str(data_dir / category / "4.mask")
    yolo_train_directory = str(data_dir / category / "5.dataset")
    yolo_yaml_directory = str(data_dir / category / "5.dataset" / "data.yaml")

    log(f"이미지 디렉토리: {image_directory}")
    log(f"박스 디렉토리: {box_directory}")
    log(f"마스크 디렉토리: {mask_directory}")
    log(f"YOLO 학습 디렉토리: {yolo_train_directory}")
    log(f"YOLO YAML 경로: {yolo_yaml_directory}")
    log(f"NPY 마스크 저장: {'활성화' if SAVE_NPY_FILES else '비활성화'}")

    def get_image_paths(directory, extensions=('jpg', 'jpeg', 'png')):
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    image_paths = get_image_paths(image_directory)

    # --mask-only 옵션이 활성화된 경우 마스크 데이터가 있는 이미지만 필터링
    if args.mask_only:
        log("--mask-only 옵션 활성화: 마스크 데이터가 있는 이미지만 처리합니다.", "INFO")
        
        # 사용 가능한 마스크 파일 확인
        available_masks = set()
        for mask_file in os.listdir(mask_directory):
            # 디버그 형식에서는 .txt 파일로 저장되고, _box_points.txt는 제외
            if mask_file.endswith('.txt') and not mask_file.endswith('_box_points.txt'):
                # 파일명에서 이미지 이름 추출 (예: "image_name.txt" -> "image_name")
                image_name = os.path.splitext(mask_file)[0]
                available_masks.add(image_name)
        
        log(f"사용 가능한 마스크 데이터: {len(available_masks)}개", "INFO")
        
        # 마스크 데이터가 있는 이미지만 선별
        original_count = len(image_paths)
        filtered_image_paths = []
        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            if img_basename in available_masks:
                filtered_image_paths.append(img_path)
        
        image_paths = filtered_image_paths
        
        log(f"전체 이미지: {original_count}개", "INFO")
        log(f"마스크 데이터가 있는 이미지: {len(image_paths)}개", "INFO")
        if original_count > 0:
            log(f"필터링된 이미지 비율: {len(image_paths)/original_count*100:.1f}%", "INFO")
        
        if len(image_paths) == 0:
            log("마스크 데이터가 있는 이미지가 없습니다. 처리를 건너뜁니다.", "WARNING")
            log("먼저 이미지 검출을 실행하여 마스크 데이터를 생성하세요.", "INFO")
            sys.exit(0)

    # 이미지가 없는 경우 처리
    if not image_paths:
        log(f"경고: {image_directory} 디렉토리에 이미지가 없습니다.", "WARNING")
        log("이미지를 해당 디렉토리에 추가한 후 다시 실행해 주세요.", "WARNING")
        log("종료합니다.")
        sys.exit(0)

    log(f"처리할 이미지 수: {len(image_paths)}")
    
    # 테스트 이미지 수 제한 적용
    if args.test_images:
        original_count = len(image_paths)
        image_paths = image_paths[:args.test_images]
        log(f"테스트 모드: {original_count}개 중 {len(image_paths)}개 이미지만 처리합니다.", "INFO")
    
    # 메모리 최적화: 처리 전 메모리 사용량 로깅
    if MEMORY_MONITORING:
        log_memory_usage("이미지 처리 전")

    # 모든 탐지 결과를 저장할 딕셔너리
    all_detections = {}

    # 모든 이미지에 대해 예측 수행
    max_images = min(len(image_paths), MAX_IMAGES)
    
    # 배치 처리 크기 설정
    batch_size = BATCH_SIZE
    
    # 프로그레스 바 사용
    for batch_idx in tqdm(range(0, max_images, batch_size), desc="이미지 배치 처리 중"):
        # 현재 배치의 이미지 경로 선택
        batch_end = min(batch_idx + batch_size, max_images)
        current_batch = image_paths[batch_idx:batch_end]
        
        # 각 배치 후 메모리 정리를 위한 로컬 변수 사용
        batch_detections = {}
        
        for cnt, image_path in enumerate(current_batch):
            try:
                overall_idx = batch_idx + cnt
                log(f"\n처리 중: {overall_idx+1}/{max_images} - {image_path}", "VERBOSE")
                image = cv2.imread(image_path)
                if image is None:
                    log(f"이미지 로드 실패: {image_path}", "WARNING")
                    continue
                else:
                    log(f"이미지 로드 성공: {image_path}", "DEBUG")
                
                log("GroundedSAM2 예측 시작...", "VERBOSE")
                results = base_model.predict(image_path)
                
                if results is None:
                    log(f"결과 없음: {image_path}", "WARNING")
                    continue
                
                log("GroundedSAM2 예측 완료", "VERBOSE")
                log(f"결과 객체 유형: {type(results)}", "DEBUG")
                log(f"결과 객체 속성: {dir(results)}", "DEBUG")
                
                if not hasattr(results, 'xyxy'):
                    log(f"결과에 xyxy 속성이 없습니다: {image_path}", "WARNING")
                    continue
                    
                log(f"박스 수: {len(results.xyxy)}", "INFO")
                log(f"클래스 ID: {results.class_id if hasattr(results, 'class_id') else 'None'}", "DEBUG")
                
                # 파일 이름만 추출 (경로 제외)
                img_filename = os.path.basename(image_path)
                img_basename = os.path.splitext(img_filename)[0]
                
                # Box 정보 저장
                try:
                    box_data = {
                        "image_path": image_path,
                        "boxes": results.xyxy.tolist() if hasattr(results.xyxy, 'tolist') else [],
                        "class_ids": results.class_id.tolist() if hasattr(results, 'class_id') and hasattr(results.class_id, 'tolist') else [],
                        "classes": [base_model.ontology.classes()[cls_id] for cls_id in results.class_id] if hasattr(results, 'class_id') else []
                    }
                    
                    if hasattr(results, 'confidence') and results.confidence is not None:
                        box_data["confidence"] = results.confidence.tolist() if hasattr(results.confidence, 'tolist') else []
                    
                    with open(os.path.join(box_directory, f"{img_basename}_box.json"), 'w') as f:
                        json.dump(box_data, f, indent=2)
                        
                    log(f"박스 정보 저장 완료: {os.path.join(box_directory, img_basename+'_box.json')}", "DEBUG")
                except Exception as e:
                    log(f"박스 정보 저장 중 오류: {e}", "ERROR")
                    if DEBUG:
                        traceback.print_exc()
                
                # Mask 정보 저장 (있는 경우)
                try:
                    if hasattr(results, 'mask') and results.mask is not None:
                        log("결과에 mask 속성이 있습니다.", "VERBOSE")
                        log(f"마스크 유형: {type(results.mask)}", "DEBUG")
                        if isinstance(results.mask, np.ndarray):
                            log(f"마스크 형태: {results.mask.shape}", "DEBUG")
                        
                        # 추가 마스크 세부 정보 로깅
                        if hasattr(results, 'segments') and results.segments is not None:
                            log("결과에 segments 속성이 있습니다!", "INFO")
                            log(f"segments 유형: {type(results.segments)}", "DEBUG")
                            if isinstance(results.segments, list):
                                log(f"segments 개수: {len(results.segments)}", "INFO")
                                for i, segment in enumerate(results.segments[:2]):  # 처음 2개만 로깅
                                    log(f"segment {i} 유형: {type(segment)}, 데이터: {segment[:10]}...", "DEBUG")
                            elif isinstance(results.segments, np.ndarray):
                                log(f"segments 배열 형태: {results.segments.shape}", "DEBUG")
                        
                        if hasattr(results, 'polygons') and results.polygons is not None:
                            log("결과에 polygons 속성이 있습니다!", "INFO")
                            log(f"polygons 유형: {type(results.polygons)}", "DEBUG")
                            if isinstance(results.polygons, list):
                                log(f"polygons 개수: {len(results.polygons)}", "INFO")
                                for i, polygon in enumerate(results.polygons[:2]):  # 처음 2개만 로깅
                                    log(f"polygon {i} 유형: {type(polygon)}, 데이터: {polygon[:10]}...", "DEBUG")
                        
                        # 새로운 코드: 마스크를 그대로 저장
                        log("마스크 배열을 원본 그대로 저장 중...", "INFO")
                        
                        # 이미지 크기 정의
                        if image is not None:
                            image_shape = (image.shape[1], image.shape[0])  # (너비, 높이)
                        else:
                            image_shape = (640, 480)  # 기본 이미지 크기
                        log(f"이미지 크기: {image_shape}", "DEBUG")
                        
                        # 마스크가 3D 배열인 경우 (여러 객체)
                        if len(results.mask.shape) == 3 and results.mask.shape[0] > 1:
                            log(f"여러 객체 마스크 감지: {results.mask.shape[0]}개", "INFO")
                            
                            success = save_mask_as_raw(
                                results.mask,
                                mask_directory,
                                img_basename,
                                results.class_id.tolist() if hasattr(results, 'class_id') else None,
                                image_shape,
                                save_png=False
                            )
                            
                            if success:
                                log(f"마스크 배열 저장 완료: {success.get('npy')}", "INFO")
                                log(f"마스크 PNG 이미지 저장 완료: {len(success.get('png', [])) if isinstance(success.get('png'), list) else 1}개 파일", "INFO")
                        # 단일 마스크인 경우
                        else:
                            if len(results.mask.shape) == 3:
                                single_mask = results.mask[0]
                            else:
                                single_mask = results.mask
                                
                            class_id = results.class_id[0] if hasattr(results, 'class_id') and len(results.class_id) > 0 else 0
                            
                            success = save_mask_as_raw(
                                single_mask,
                                mask_directory,
                                img_basename,
                                class_id,
                                image_shape,
                                save_png=False
                            )
                            
                            if success:
                                log(f"마스크 배열 저장 완료: {success.get('npy')}", "INFO")
                                log(f"마스크 PNG 이미지 저장 완료: {success.get('png')}", "INFO")
                        
                        # 마스크 시각화 (선택적)
                        if args.visualize:
                            try:
                                # 원본 이미지 불러오기
                                vis_img = cv2.imread(image_path)
                                
                                # 마스크 오버레이
                                if len(results.mask.shape) == 3:
                                    for i in range(results.mask.shape[0]):
                                        mask = results.mask[i].astype(np.uint8) * 255
                                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        
                                        # 클래스별 색상 설정
                                        class_name = base_model.ontology.classes()[i] if i < len(base_model.ontology.classes()) else "unknown"
                                        color = (0, 255, 0) if class_name == "fence_person" else (0, 0, 255) if class_name == "sidewalk" else (255, 0, 0) if class_name == "car" else (255, 255, 0)
                                        
                                        # 윤곽선 그리기
                                        cv2.drawContours(vis_img, contours, -1, color, 2)
                                        
                                        # 클래스 이름 표시
                                        if contours:
                                            M = cv2.moments(contours[0])
                                            if M["m00"] != 0:
                                                cx = int(M["m10"] / M["m00"])
                                                cy = int(M["m01"] / M["m00"])
                                                cv2.putText(vis_img, class_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                else:
                                    mask = results.mask.astype(np.uint8) * 255
                                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2)
                                
                                # 시각화 이미지 저장
                                vis_path = os.path.join(mask_directory, f"{img_basename}_mask_vis.jpg")
                                cv2.imwrite(vis_path, vis_img)
                                log(f"마스크 시각화 저장: {vis_path}", "DEBUG")
                            except Exception as e:
                                log(f"마스크 시각화 중 오류: {e}", "WARNING")
                        
                        # 기존 디버그 형식 저장 코드는 유지 (옵션에 따라)
                        if args.debug_format:
                            if hasattr(results, 'masks'):
                                log("결과에 'masks' 속성이 있습니다 (복수형).", "DEBUG")
                            else:
                                log("결과에 마스크 속성이 없습니다.", "DEBUG")
                                
                                # 마스크가 없는 경우, 경계 상자만으로 좌표 데이터 생성 및 저장
                                if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                                    log("경계 상자 정보만 사용하여 좌표 데이터 생성", "DEBUG")
                                    
                                    # 원본 이미지 크기 확인
                                    try:
                                        image_shape = (image.shape[1], image.shape[0])  # (너비, 높이)
                                        log(f"이미지 크기: {image_shape}", "DEBUG")
                                    except:
                                        # 기본 이미지 크기 설정
                                        image_shape = (640, 480)
                                        log(f"이미지 크기를 확인할 수 없어 기본값 사용: {image_shape}", "WARNING")
                                    
                                    # 클래스 이름 리스트 생성
                                    if hasattr(results, 'class_id'):
                                        class_names = [base_model.ontology.classes()[cls_id] for cls_id in results.class_id]
                                    else:
                                        class_names = None
                                    
                                    # 신뢰도 점수 가져오기
                                    confidence_scores = results.confidence.tolist() if hasattr(results, 'confidence') and results.confidence is not None else [1.0] * len(results.xyxy)
                                    
                                    # YOLO 형식 텍스트 파일 생성 (경계 상자 기반)
                                    try:
                                        yolo_txt_path = os.path.join(mask_directory, f"{img_basename}.txt")
                                        with open(yolo_txt_path, 'w') as f:
                                            for i, box in enumerate(results.xyxy.tolist()):
                                                if len(box) == 4:
                                                    x1, y1, x2, y2 = box
                                                    class_id = results.class_id[i] if hasattr(results, 'class_id') and i < len(results.class_id) else 0
                                                    
                                                    # 박스 중심 계산
                                                    center_x = (x1 + x2) / 2
                                                    center_y = (y1 + y2) / 2
                                                    
                                                    # 박스 크기
                                                    width = x2 - x1
                                                    height = y2 - y1
                                                    
                                                    # YOLO 형식의 사각형 생성 (단순 사각형)
                                                    rect_points = [
                                                        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                                                    ]
                                                    
                                                    # 정규화된 좌표로 변환
                                                    line_parts = [str(class_id)]
                                                    for px, py in rect_points:
                                                        # 이미지 크기로 정규화 (0~1 범위)
                                                        norm_px = px / image_shape[0]
                                                        norm_py = py / image_shape[1]
                                                        line_parts.append(f"{norm_px:.5f}")
                                                        line_parts.append(f"{norm_py:.5f}")
                                                    
                                                    # 공백으로 구분하여 한 줄에 작성
                                                    f.write(" ".join(line_parts) + "\n")
                                    except Exception as e:
                                        log(f"YOLO 형식 경계 상자 파일 저장 중 오류: {e}", "ERROR")
                                    
                                    log(f"YOLO 형식 경계 상자 기반 좌표 파일 저장 완료: {yolo_txt_path}", "INFO")
                                    if args.debug_format:
                                        # 디버그 형식으로 저장 (마스크 없음)
                                        saved_files = save_coords_without_mask_debug_format(
                                            results.xyxy.tolist(),
                                            results.class_id.tolist() if hasattr(results, 'class_id') else None,
                                            confidence_scores,
                                            mask_directory,  # Polygon 데이터용 디렉토리 (mask 폴더)
                                            box_directory,   # Box points 데이터용 디렉토리 (box 폴더)
                                            img_basename,
                                            class_names,
                                            image_shape
                                        )
                                        
                                        if saved_files:
                                            log(f"디버그 형식 경계 상자 기반 좌표 처리 완료:", "DEBUG")
                                            log(f"  - Polygon: {saved_files['polygon']}", "DEBUG")
                                            log(f"  - Box Points: {saved_files['box_points']}", "DEBUG")
                                    else:
                                        # 기존 형식으로 저장
                                        saved_files = save_coords_without_mask(
                                            results.xyxy.tolist(),
                                            mask_directory,
                                            img_basename,
                                            results.class_id.tolist() if hasattr(results, 'class_id') else None,
                                            class_names,
                                            None,  # 시각화 제거로 색상맵이 필요 없음
                                            image_shape
                                        )
                                        
                                        if saved_files:
                                            log(f"경계 상자 기반 좌표 처리 완료:", "DEBUG")
                                            log(f"  - JSON: {saved_files['json']}", "DEBUG")
                                            log(f"  - TXT: {saved_files['txt']}", "DEBUG")

                except Exception as e:
                    log(f"마스크 처리 중 오류: {e}", "ERROR")
                    if DEBUG:
                        traceback.print_exc()
                
                # 모든 탐지 결과 저장 (분석용) - 배치별 로컬 딕셔너리에 저장
                try:
                    batch_detections[image_path] = {
                        "boxes": results.xyxy.tolist() if hasattr(results.xyxy, 'tolist') else [],
                        "class_ids": results.class_id.tolist() if hasattr(results, 'class_id') and hasattr(results.class_id, 'tolist') else [],
                        "classes": [base_model.ontology.classes()[cls_id] for cls_id in results.class_id] if hasattr(results, 'class_id') else [],
                        "has_mask": hasattr(results, 'mask') and results.mask is not None
                    }
                except Exception as e:
                    log(f"탐지 결과 저장 중 오류: {e}", "ERROR")
                    if DEBUG:
                        traceback.print_exc()

                # 시각화
                try:
                    # 시각화 옵션이 활성화된 경우에만 실행
                    if PLOT:
                        # 시각화 전에 이미지가 유효한지 확인
                        if image is None or image.size == 0:
                            log(f"시각화를 위한 이미지가 유효하지 않습니다: {image_path}", "WARNING")
                        else:
                            plot_img = plot(
                                image=image,
                                classes=base_model.ontology.classes(),
                                detections=results,
                                raw=False
                            )
                            
                            # 반환된 이미지가 유효한지 확인
                            if plot_img is None or plot_img.size == 0:
                                log(f"시각화 결과 이미지가 유효하지 않습니다", "WARNING")
                            else:
                                # 출력 디렉토리 확인
                                results_dir = data_dir / category / "7.results"
                                os.makedirs(results_dir, exist_ok=True)
                                
                                # 시각화 이미지 저장
                                result_path = os.path.join(results_dir, f"{img_basename}_plot.png")
                                success = cv2.imwrite(result_path, plot_img)
                                
                                if success:
                                    log(f"결과 시각화 저장: {result_path}", "INFO")
                                else:
                                    log(f"이미지 저장 실패: {result_path}", "WARNING")
                except Exception as e:
                    log(f"시각화 중 오류: {e}", "ERROR")
                    if DEBUG:
                        traceback.print_exc()
                
                # 메모리 최적화: 불필요한 객체 즉시 해제
                results = None
                image = None
                if 'plot_img' in locals():
                    del plot_img
                
            except Exception as e:
                log(f"이미지 {image_path} 처리 중 오류: {e}", "ERROR")
                if DEBUG:
                    traceback.print_exc()
        
        # 각 배치가 끝날 때마다 전체 결과에 병합
        all_detections.update(batch_detections)
        
        # 메모리 최적화: 배치 처리 후 불필요한 객체 명시적 해제
        batch_detections = None
        
        # 메모리 최적화: 필요한 경우 가비지 컬렉션 강제 실행
        import gc
        gc.collect()
        
        # 메모리 사용량 로깅 (Python 메모리 프로파일러 설치된 경우)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            log(f"배치 {batch_idx//batch_size+1}/{(max_images+batch_size-1)//batch_size} 완료 후 메모리 사용량: {memory_info.rss / (1024 * 1024):.2f} MB", "INFO")
        except ImportError:
            log(f"배치 {batch_idx//batch_size+1}/{(max_images+batch_size-1)//batch_size} 완료", "INFO")

    # 모든 탐지 결과를 하나의 파일로 저장
    try:
        log("모든 탐지 결과 저장 중...", "VERBOSE")
        all_detections_file = os.path.join(str(data_dir / category / "7.results"), "all_detections.json")
        with open(all_detections_file, 'w') as f:
            # 경로를 상대 경로로 변환하여 이식성 확보
            portable_detections = {}
            for path, data in all_detections.items():
                rel_path = os.path.basename(path)
                data_copy = data.copy()
                data_copy["image_path"] = rel_path  # 절대 경로 대신 파일명만 저장
                portable_detections[rel_path] = data_copy
            
            json.dump(portable_detections, f, indent=2)
        log(f"모든 탐지 결과 저장 완료: {all_detections_file}")
    except Exception as e:
        log(f"모든 탐지 결과 저장 중 오류: {e}", "ERROR")
        if DEBUG:
            traceback.print_exc()

    log("탐지 완료")

    # YOLO 데이터셋 생성
    try:
        log("YOLO 데이터셋 생성 시작...", "VERBOSE")
        
        # 메모리 최적화: 기존 label() 메서드 대신 배치 처리 접근 방식 사용
        yolo_batch_size = 20  # 한 번에 처리할 이미지 수
        
        # 출력 디렉토리 구조 확인 및 생성
        train_dir = os.path.join(yolo_train_directory, "train")
        val_dir = os.path.join(yolo_train_directory, "val")
        train_images_dir = os.path.join(train_dir, "images")
        train_labels_dir = os.path.join(train_dir, "labels")
        val_images_dir = os.path.join(val_dir, "images")
        val_labels_dir = os.path.join(val_dir, "labels")
        
        for dir_path in [train_dir, val_dir, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 마스크 데이터가 있는 이미지만 필터링
        log("마스크 데이터가 있는 이미지 필터링 중...", "INFO")
        available_masks = set()
        for mask_file in os.listdir(mask_directory):
            # 디버그 형식에서는 .txt 파일로 저장되고, _box_points.txt는 제외
            if mask_file.endswith('.txt') and not mask_file.endswith('_box_points.txt'):
                # 파일명에서 이미지 이름 추출 (예: "image_name.txt" -> "image_name")
                image_name = os.path.splitext(mask_file)[0]
                available_masks.add(image_name)
        
        log(f"사용 가능한 마스크 데이터: {len(available_masks)}개", "INFO")
        
        # 마스크 데이터가 있는 이미지만 선별
        filtered_image_paths = []
        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            if img_basename in available_masks:
                filtered_image_paths.append(img_path)
        
        log(f"전체 이미지: {len(image_paths)}개", "INFO")
        log(f"마스크 데이터가 있는 이미지: {len(filtered_image_paths)}개", "INFO")
        log(f"필터링된 이미지 비율: {len(filtered_image_paths)/len(image_paths)*100:.1f}%", "INFO")
        
        if len(filtered_image_paths) == 0:
            log("마스크 데이터가 있는 이미지가 없습니다. YOLO 데이터셋 생성을 건너뜁니다.", "WARNING")
            sys.exit(0)
        
        # 이미지 경로 목록 섞기 (랜덤화) - 필터링된 이미지만 사용
        import random
        random.seed(42)  # 재현성을 위한 시드 설정
        shuffled_paths = filtered_image_paths.copy()
        random.shuffle(shuffled_paths)
        
        # 학습/검증 세트 분할 (80/20)
        split_idx = int(len(shuffled_paths) * 0.8)
        train_images = shuffled_paths[:split_idx]
        val_images = shuffled_paths[split_idx:]
        
        log(f"학습 세트 이미지 수: {len(train_images)}", "INFO")
        log(f"검증 세트 이미지 수: {len(val_images)}", "INFO")
        
        # 배치별로 학습 이미지 처리
        for batch_idx in tqdm(range(0, len(train_images), yolo_batch_size), desc="학습 데이터 생성"):
            batch_end = min(batch_idx + yolo_batch_size, len(train_images))
            current_batch = train_images[batch_idx:batch_end]
            
            for img_path in current_batch:
                # 이미지 파일 이름
                img_filename = os.path.basename(img_path)
                img_basename = os.path.splitext(img_filename)[0]
                
                # 원본 이미지 복사
                img_dest = os.path.join(train_images_dir, img_filename)
                try:
                    import shutil
                    shutil.copy2(img_path, img_dest)
                except Exception as e:
                    log(f"이미지 복사 중 오류: {e}", "ERROR")
                    continue
                
                # 레이블 파일 생성 (미리 생성된 mask_directory의 텍스트 파일 사용)
                txt_src = os.path.join(mask_directory, f"{img_basename}.txt")
                txt_dest = os.path.join(train_labels_dir, f"{img_basename}.txt")
                
                # 필터링된 이미지이므로 마스크 파일이 반드시 존재해야 함
                if os.path.exists(txt_src):
                    try:
                        shutil.copy2(txt_src, txt_dest)
                    except Exception as e:
                        log(f"레이블 파일 복사 중 오류: {e}", "ERROR")
                else:
                    # 이 경우는 발생하지 않아야 하지만 안전장치
                    log(f"예상치 못한 오류: 마스크 파일이 없음 - {txt_src}", "ERROR")
            
            # 배치 처리 후 메모리 정리
            import gc
            gc.collect()
        
        # 배치별로 검증 이미지 처리
        for batch_idx in tqdm(range(0, len(val_images), yolo_batch_size), desc="검증 데이터 생성"):
            batch_end = min(batch_idx + yolo_batch_size, len(val_images))
            current_batch = val_images[batch_idx:batch_end]
            
            for img_path in current_batch:
                # 이미지 파일 이름
                img_filename = os.path.basename(img_path)
                img_basename = os.path.splitext(img_filename)[0]
                
                # 원본 이미지 복사
                img_dest = os.path.join(val_images_dir, img_filename)
                try:
                    import shutil
                    shutil.copy2(img_path, img_dest)
                except Exception as e:
                    log(f"이미지 복사 중 오류: {e}", "ERROR")
                    continue
                
                # 레이블 파일 생성 (미리 생성된 mask_directory의 텍스트 파일 사용)
                txt_src = os.path.join(mask_directory, f"{img_basename}.txt")
                txt_dest = os.path.join(val_labels_dir, f"{img_basename}.txt")
                
                # 필터링된 이미지이므로 마스크 파일이 반드시 존재해야 함
                if os.path.exists(txt_src):
                    try:
                        shutil.copy2(txt_src, txt_dest)
                    except Exception as e:
                        log(f"레이블 파일 복사 중 오류: {e}", "ERROR")
                else:
                    # 이 경우는 발생하지 않아야 하지만 안전장치
                    log(f"예상치 못한 오류: 마스크 파일이 없음 - {txt_src}", "ERROR")
            
            # 배치 처리 후 메모리 정리
            import gc
            gc.collect()
        
        # data.yaml 파일 생성
        yaml_content = {
            'train': './train/images',
            'val': './val/images',
            'nc': len(base_model.ontology.classes()),
            'names': list(base_model.ontology.classes())
        }
        
        try:
            import yaml
            with open(yolo_yaml_directory, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
            
            log(f"YAML 설정 파일 생성 완료: {yolo_yaml_directory}", "INFO")
        except Exception as e:
            log(f"YAML 파일 생성 중 오류: {e}", "ERROR")
            
        log("YOLO 데이터셋 생성 완료", "INFO")
        
    except Exception as e:
        log(f"YOLO 데이터셋 생성 중 오류: {e}", "ERROR")
        if DEBUG:
            traceback.print_exc()

except Exception as e:
    log(f"프로그램 실행 중 오류: {e}", "ERROR")
    if DEBUG:
        traceback.print_exc()

# 실행 시간 계산 전에 명시적인 메모리 정리 코드 추가
log("\n리소스 정리 중...", "INFO")

# 대용량 객체 해제
all_detections = None
base_model = None

# 메모리 이상 상태 확인
if MEMORY_MONITORING:
    try:
        final_memory = log_memory_usage("종료")
        log(f"최종 메모리 사용량: {final_memory:.2f} MB", "INFO")
    except Exception as e:
        log(f"메모리 모니터링 오류: {e}", "ERROR")
        
# 가비지 컬렉션 강제 수행
try:
    import gc
    collected = gc.collect(generation=2)
    log(f"가비지 컬렉션으로 {collected}개 객체 정리됨", "INFO")
except Exception as e:
    log(f"가비지 컬렉션 오류: {e}", "ERROR")

# 실행 시간 계산
sec = time.time()-start # 종료 - 시작 (걸린 시간)
times = str(datetime.timedelta(seconds=sec)) # 걸린시간 보기좋게 바꾸기
short = times.split(".")[0] # 초 단위 까지만
log(f"총 실행 시간: {times} 초")
log(f"단축 시간: {short}")

# 마지막에 전처리 코드 추가
if args.preprocess:
    try:
        log("\n=== 이미지 전처리 시작 ===", "INFO")
        # 전처리 유틸리티 로드
        from scripts.preprocess_utils import batch_process_with_coords
        from scripts.data_utils import get_subfolder_path

        # 전처리 옵션 설정
        crop_enabled = not args.no_crop
        mask_enabled = not args.no_mask
        
        # 경로 설정
        mask_coords_dir = str(data_dir / category / "4.mask")
        preprocessed_dir = str(data_dir / category / "6.preprocessed")
        
        log(f"좌표 디렉토리: {mask_coords_dir}", "INFO")
        log(f"전처리 출력 디렉토리: {preprocessed_dir}", "INFO")
        log(f"객체 크롭: {'활성화' if crop_enabled else '비활성화'}", "INFO")
        log(f"마스크 적용: {'활성화' if mask_enabled else '비활성화'}", "INFO")
        
        # 좌표 파일 확인
        coords_files = [f for f in os.listdir(mask_coords_dir) if f.endswith('_coords.json')]
        
        if not coords_files:
            log("좌표 파일을 찾을 수 없습니다. 먼저 이미지 검출을 실행하세요.", "ERROR")
        else:
            log(f"좌표 파일 {len(coords_files)}개를 찾았습니다.", "INFO")
            
            # 메모리 모니터링
            if MEMORY_MONITORING:
                log_memory_usage("전처리 전")
            
            # 배치 크기 설정
            preprocess_batch_size = min(100, BATCH_SIZE)  # 전처리용 배치 크기 설정
            
            # 전체 이미지 목록 섞기
            import random
            random.seed(42)  # 재현성을 위한 시드 설정
            shuffled_images = image_paths.copy()
            random.shuffle(shuffled_images)
            
            # 최대 처리 이미지 수 제한
            max_preprocess_images = min(len(shuffled_images), MAX_IMAGES)
            preprocess_images = shuffled_images[:max_preprocess_images]
            
            log(f"전처리할 이미지 수: {len(preprocess_images)}/{len(image_paths)}", "INFO")
            
            # 배치 처리 결과 저장
            all_results = {}
            total_objects = 0
            all_skipped = []
            all_errors = []
            
            # 배치 단위로 처리
            for batch_idx in tqdm(range(0, len(preprocess_images), preprocess_batch_size), desc="전처리 배치 진행 중"):
                batch_end = min(batch_idx + preprocess_batch_size, len(preprocess_images))
                current_batch = preprocess_images[batch_idx:batch_end]
                
                # 배치 전처리 수행
                batch_result = batch_process_with_coords(
                    image_paths=current_batch, 
                    coords_dir=mask_coords_dir,
                    output_dir=preprocessed_dir,
                    crop=crop_enabled,
                    apply_mask=mask_enabled,
                    resize=target_size
                )
                
                # 결과 누적
                all_results.update(batch_result.get("result", {}))
                total_objects += batch_result.get("total_objects", 0)
                if "skipped_images" in batch_result:
                    all_skipped.extend(batch_result["skipped_images"])
                if "error_images" in batch_result:
                    all_errors.extend(batch_result["error_images"])
                
                # 배치 처리 후 메모리 정리
                batch_result = None
                import gc
                gc.collect()
                
                # 메모리 모니터링
                if MEMORY_MONITORING:
                    log_memory_usage(f"전처리 배치 {batch_idx//preprocess_batch_size+1}")
            
            # 전처리 결과 요약 출력
            log(f"총 {len(all_results)}개 이미지에서 {total_objects}개 객체 전처리 완료", "INFO")
            if all_skipped:
                log(f"{len(all_skipped)}개 이미지가 건너뛰어졌습니다.", "WARNING")
            if all_errors:
                log(f"{len(all_errors)}개 이미지에서 오류가 발생했습니다.", "ERROR")
            
            log(f"전처리된 이미지는 {preprocessed_dir}에 저장되었습니다.", "INFO")
            
    except Exception as e:
        log(f"전처리 중 오류 발생: {e}", "ERROR")
        if DEBUG:
            traceback.print_exc()

# 고품질 전처리 코드 추가
if args.advanced_preprocess:
    try:
        log("\n=== 고품질 이미지 전처리 시작 (Polygon-Mask-First 방식) ===", "INFO")
        log("처리 방식: _coords.txt의 폴리곤으로 마스크 생성 -> 원본 이미지에 적용 -> _box.json의 박스로 크롭 -> 리사이즈", "INFO")
        log("결과물은 To-be 스타일 (이미지 왼쪽 하단)을 목표로 합니다.", "INFO")
        
        # 고품질 전처리 유틸리티 로드
        import logging
        if DEBUG:
            # 디버그 모드에서는 고품질 전처리 로거의 레벨을 DEBUG로 설정
            logging.getLogger("advanced_preprocessor").setLevel(logging.DEBUG)
        
        from scripts.advanced_preprocessor import AdvancedPreprocessor
        
        log(f"카테고리: {category}", "INFO")
        log(f"타겟 크기: {target_size}", "INFO")
        if args.preprocess_class_id is not None:
            log(f"처리할 특정 클래스 ID: {args.preprocess_class_id}", "INFO")
        else:
            log("모든 클래스 처리 (또는 _coords.txt 내 모든 객체)", "INFO")
        
        # 테스트 모드 설정
        if args.test_mode:
            effective_max_images = 30 * 4 # 예시: 클래스별 30장 (총 4클래스 가정)
            log(f"테스트 모드 활성화: 최대 이미지 처리 수를 {effective_max_images} (MAX_IMAGES는 {MAX_IMAGES})로 제한하여 전달 시도", "INFO")
        else:
            effective_max_images = MAX_IMAGES # MAX_IMAGES는 argparse로 받은 값
        
        if MEMORY_MONITORING:
            log_memory_usage("고품질 전처리 전")
        
        preprocessor = AdvancedPreprocessor(
            category_name=category,
            target_size=target_size
        )
        
        log("Polygon-Mask-First 전처리 방식으로 객체 추출을 시작합니다...", "INFO")
        
        # 고품질 전처리 수행
        results = preprocessor.process_all_images(
            selected_class_id=args.preprocess_class_id,
            max_images=effective_max_images 
        )
        
        report = preprocessor.create_summary_report(results)
        log("\n" + report, "INFO")
        
        report_path = os.path.join(str(data_dir / category / "7.results"), "advanced_preprocessing_polygon_mask_report.txt")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            log(f"상세 보고서 저장: {report_path}", "INFO")
        except Exception as e:
            log(f"보고서 저장 실패 {report_path}: {e}", "ERROR")

        # 결과 요약 및 테스트 모드 결과 확인
        log(f"\n=== Polygon-Mask-First 전처리 완료 ===", "INFO")
        log(f"총 스캔된 이미지: {results.get('total_images_scanned', 0)}개", "INFO")
        log(f"객체가 처리된 이미지: {results.get('processed_images_with_objects', 0)}개", "INFO")
        log(f"총 추출된 객체: {results.get('total_objects_extracted', 0)}개", "INFO")
        log(f"총 저장된 파일: {results.get('total_files_saved', 0)}개", "INFO")
        
        class_counts_report = results.get('class_counts', {})
        if class_counts_report:
            log("클래스별 저장된 파일 수:", "INFO")
            for class_name_rep, count_rep in class_counts_report.items():
                log(f"  - {class_name_rep}: {count_rep}개 파일", "INFO")
        
        errors_report = results.get('errors', [])
        if errors_report:
            log(f"처리 중 발생한 오류: {len(errors_report)}개", "WARNING")
            # 상세 오류는 로그 파일이나 보고서에서 확인하도록 유도
            # for err_idx, error_item_rep in enumerate(errors_report[:5]):
            #     log(f"  오류 {err_idx+1}: {error_item_rep}", "WARNING")
            # if len(errors_report) > 5:
            #     log(f"  ... 외 {len(errors_report)-5}개 추가 오류 (상세 내용은 로그 또는 보고서 확인)", "WARNING")
        
        if MEMORY_MONITORING:
            log_memory_usage("고품질 전처리 후")
            
    except ImportError as ie:
        log(f"고품질 전처리 중 필요한 모듈 임포트 실패: {ie}", "ERROR")
        log("scripts.advanced_preprocessor 모듈이 올바른 위치에 있는지, 필요한 의존성이 설치되었는지 확인하세요.", "ERROR")
        if DEBUG:
            traceback.print_exc()
    except Exception as e:
        log(f"고품질 전처리 중 예외 발생: {e}", "ERROR")
        if DEBUG:
            traceback.print_exc()

# 분류 디렉토리 구조 생성 (선택적)
if args.prepare_classify:
    try:
        log("\n=== 분류 디렉토리 구조 생성 ===", "INFO")
        from scripts.preprocess_utils import prepare_classification_structure
        
        # 분류 방법 목록 파싱
        classification_methods = args.classification_methods.split(',')
        log(f"분류 방법 목록: {classification_methods}", "INFO")
        
        # 분류 디렉토리 구조 생성
        result = prepare_classification_structure(
            base_dir=data_dir,
            category_name=category,
            classification_methods=classification_methods
        )
        
        log(f"분류 디렉토리 구조 생성 완료:", "INFO")
        for method, path in result["paths"].items():
            log(f"  - {method}: {path}", "INFO")
        log(f"메타데이터 파일: {result['metadata_file']}", "INFO")
        
    except Exception as e:
        log(f"분류 디렉토리 구조 생성 중 오류 발생: {e}", "ERROR")
        if DEBUG:
            traceback.print_exc()

end = time.time()
elapsed = end - start
log(f"\n실행 완료! 총 소요 시간: {str(datetime.timedelta(seconds=int(elapsed)))}")

# YOLO 학습 (필요한 경우 주석 해제)
# target_model = YOLOv8("yolov8x-seg.pt")
# target_model.train(yolo_yaml_directory, epochs=200, device='0')

def run_main_launcher(category="test_category", det_model="grounding-dino-2", seg_model="sam-2", 
                     save_mask_png=False, batch_size=100, max_images=10000, 
                     test_mode=False, memory_monitor=False, cmd_args=None):
    """
    main_launcher를 외부에서 호출할 수 있는 API 함수
    
    Args:
        category (str): 처리할 카테고리 이름
        det_model (str): 사용할 detection 모델 ("grounding-dino-2", "grounding-dino", "yolo", "yolov8")
        seg_model (str): 사용할 segmentation 모델 ("sam-2", "sam", "mobile-sam")
        save_mask_png (bool): 마스크 PNG 이미지 저장 여부
        batch_size (int): 이미지 처리 배치 크기
        max_images (int): 처리할 최대 이미지 수
        test_mode (bool): 테스트 모드 활성화 여부 (클래스별 이미지 제한)
        memory_monitor (bool): 메모리 모니터링 활성화 여부
        cmd_args (list): 추가 명령행 인수 목록
        
    Returns:
        tuple: (성공 여부, 메시지)
    """
    global DEBUG, VERBOSE, BATCH_SIZE, MAX_IMAGES, MEMORY_MONITORING
    
    start = time.time()  # 시작 시간 기록
    
    # 설정 업데이트
    BATCH_SIZE = batch_size
    MAX_IMAGES = max_images
    MEMORY_MONITORING = memory_monitor
    
    try:
        log(f"실행 카테고리: {category}")
        
        # 메모리 모니터링 활성화 시 초기 사용량 기록
        if MEMORY_MONITORING:
            initial_memory = log_memory_usage("시작")
        
        # 디렉토리 생성
        directories = [
            data_dir / category / "1.images",
            data_dir / category / "2.support-set",
            data_dir / category / "3.box",
            data_dir / category / "4.mask",
            data_dir / category / "5.dataset",
            data_dir / category / "6.preprocessed",
            data_dir / category / "7.results",
            data_dir / category / "8.refine-dataset"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            log(f"디렉토리 생성: {directory}", "DEBUG")
        
        # 사용자 정의 헬퍼 모듈 import 및 패치 적용 (GroundedSAM2 import 전에 수행)
        try:
            from scripts.custom_helpers import patch_grounded_sam2
            patch_successful = patch_grounded_sam2()
            if not patch_successful:
                log("WARNING: 프로젝트 내 디렉토리를 사용한 SAM 모델 로딩 패치 실패", "WARNING")
                log("기본 ~/.cache 경로가 사용될 수 있습니다.", "WARNING")
        except Exception as e:
            log(f"패치 적용 중 오류: {e}", "ERROR")
            if DEBUG:
                traceback.print_exc()
        
        # 라이브러리 로드
        log("필요한 라이브러리 로드 중...", "INFO")
        
        # 메모리 최적화: 모델 로드 전 메모리 사용량 로깅
        if MEMORY_MONITORING:
            log_memory_usage("모델 로드 전")
        
        # 샘플 추론 배치 크기 조정 (메모리 사용량을 줄이기 위해)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 큰 텐서를 더 작은 청크로 분할
        
        # Detection 모델 선택
        if det_model == "grounding-dino-2":
            log("GroundedSAM2 모델 로드 중...", "INFO")
            from autodistill_grounded_sam_2 import GroundedSAM2
            from autodistill.detection import CaptionOntology
            
            # 온톨로지 정의
            ontology = CaptionOntology({
                "What blue, fabric-like barrier, explicitly designed to shield or guide pedestrians near construction zones, is visible in this image?": "fence_person",
                "What type of pedestrian pathway, found alongside roads or within construction zones, composed of materials like nonwoven fabric, sand, bricks, or asphalt, is visible in this image?": "sidewalk",
                "What motorized vehicle, designed for passenger or cargo transport, commonly seen in urban or road environments, is visible in this image?": "car",
                "What small, brightly colored cone-shaped object, specifically designed to redirect traffic or highlight construction hazards, is visible in this image?": "traffic cone",
            })
            
            # 모델 로드
            base_model = GroundedSAM2(ontology=ontology)
            
        elif det_model == "grounding-dino":
            log("GroundedSAM 모델 로드 중...", "INFO")
            from autodistill.detection import GroundedSAM, CaptionOntology
            
            # 온톨로지 정의
            ontology = CaptionOntology({
                "barrier designed to shield pedestrians": "fence_person",
                "pedestrian pathway": "sidewalk",
                "motor vehicle": "car",
                "traffic cone": "traffic cone",
            })
            
            # 모델 로드
            base_model = GroundedSAM(ontology=ontology)
            
        elif det_model == "yolov8" or det_model == "yolo":
            log("YOLOv8 모델 로드 중...", "INFO")
            from autodistill_yolov8 import YOLOv8
            
            # 모델 로드
            base_model = YOLOv8("yolov8x.pt")
            
        else:
            raise ValueError(f"지원하지 않는 detection 모델: {det_model}")
        
        # 메모리 사용량 로깅
        if MEMORY_MONITORING:
            log_memory_usage("모델 로드 후")
        
        # 경로 설정
        image_directory = str(data_dir / category / "1.images")
        box_directory = str(data_dir / category / "3.box")
        mask_directory = str(data_dir / category / "4.mask")
        yolo_train_directory = str(data_dir / category / "5.dataset")
        yolo_yaml_directory = str(data_dir / category / "5.dataset" / "data.yaml")
        
        log(f"이미지 디렉토리: {image_directory}")
        log(f"박스 디렉토리: {box_directory}")
        log(f"마스크 디렉토리: {mask_directory}")
        log(f"YOLO 학습 디렉토리: {yolo_train_directory}")
        log(f"YOLO YAML 경로: {yolo_yaml_directory}")
        log(f"PNG 마스크 저장: {'활성화' if save_mask_png else '비활성화'}")
        
        # 이미지 경로 가져오기
        image_paths = get_image_paths(image_directory)
        
        # 테스트 모드가 활성화된 경우 이미지 수 제한
        if test_mode:
            log("테스트 모드 활성화: 이미지 수 제한", "INFO")
            original_count = len(image_paths)
            image_paths = image_paths[:min(30, len(image_paths))]
            log(f"테스트 모드: {original_count}개 중 {len(image_paths)}개 이미지만 처리합니다.", "INFO")
        
        # 이미지가 없는 경우 처리
        if not image_paths:
            log(f"경고: {image_directory} 디렉토리에 이미지가 없습니다.", "WARNING")
            return False, f"{image_directory} 디렉토리에 이미지가 없습니다."
        
        log(f"처리할 이미지 수: {len(image_paths)}")
        
        # 메모리 최적화: 처리 전 메모리 사용량 로깅
        if MEMORY_MONITORING:
            log_memory_usage("이미지 처리 전")
        
        # 모든 탐지 결과를 저장할 딕셔너리
        all_detections = {}
        
        # 모든 이미지에 대해 예측 수행
        max_images = min(len(image_paths), MAX_IMAGES)
        
        # 배치 처리
        for batch_idx in tqdm(range(0, max_images, BATCH_SIZE), desc="이미지 배치 처리 중"):
            # 현재 배치의 이미지 경로 선택
            batch_end = min(batch_idx + BATCH_SIZE, max_images)
            current_batch = image_paths[batch_idx:batch_end]
            
            # 각 이미지 처리
            for cnt, image_path in enumerate(current_batch):
                try:
                    overall_idx = batch_idx + cnt
                    log(f"\n처리 중: {overall_idx+1}/{max_images} - {image_path}", "VERBOSE")
                    
                    # 이미지 로드
                    image = cv2.imread(image_path)
                    if image is None:
                        log(f"이미지 로드 실패: {image_path}", "WARNING")
                        continue
                    
                    # 모델 예측
                    log("모델 예측 시작...", "VERBOSE")
                    results = base_model.predict(image_path)
                    
                    if results is None:
                        log(f"결과 없음: {image_path}", "WARNING")
                        continue
                    
                    log("모델 예측 완료", "VERBOSE")
                    
                    # 파일 이름만 추출 (경로 제외)
                    img_filename = os.path.basename(image_path)
                    img_basename = os.path.splitext(img_filename)[0]
                    
                    # Box 정보 저장
                    try:
                        box_data = {
                            "image_path": image_path,
                            "boxes": results.xyxy.tolist() if hasattr(results.xyxy, 'tolist') else [],
                            "class_ids": results.class_id.tolist() if hasattr(results, 'class_id') and hasattr(results.class_id, 'tolist') else [],
                            "classes": [base_model.ontology.classes()[cls_id] for cls_id in results.class_id] if hasattr(results, 'class_id') else []
                        }
                        
                        if hasattr(results, 'confidence') and results.confidence is not None:
                            box_data["confidence"] = results.confidence.tolist() if hasattr(results.confidence, 'tolist') else []
                        
                        with open(os.path.join(box_directory, f"{img_basename}_box.json"), 'w') as f:
                            json.dump(box_data, f, indent=2)
                            
                        log(f"박스 정보 저장 완료: {os.path.join(box_directory, img_basename+'_box.json')}", "DEBUG")
                    except Exception as e:
                        log(f"박스 정보 저장 중 오류: {e}", "ERROR")
                        if DEBUG:
                            traceback.print_exc()
                
                    # Mask 정보 저장 (있는 경우)
                    try:
                        if hasattr(results, 'mask') and results.mask is not None:
                            # 마스크 저장
                            save_mask_as_raw(
                                results.mask, 
                                mask_directory, 
                                img_basename, 
                                class_id=results.class_id if hasattr(results, 'class_id') else None,
                                save_png=save_mask_png
                            )
                            
                            # 좌표 파일도 저장
                            convert_and_save_mask(
                                results.mask, 
                                mask_directory, 
                                img_basename, 
                                class_names=[base_model.ontology.classes()[cls_id] for cls_id in results.class_id] if hasattr(results, 'class_id') else None
                            )
                            
                            log(f"마스크 정보 저장 완료: {os.path.join(mask_directory, img_basename)}", "DEBUG")
                    except Exception as e:
                        log(f"마스크 정보 저장 중 오류: {e}", "ERROR")
                        if DEBUG:
                            traceback.print_exc()
                
                except Exception as e:
                    log(f"이미지 처리 중 오류: {image_path} - {e}", "ERROR")
                    if DEBUG:
                        traceback.print_exc()
        
        # 총 실행 시간
        elapsed = time.time() - start
        log(f"총 실행 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")
        
        # 메모리 모니터링
        if MEMORY_MONITORING:
            final_memory = log_memory_usage("완료")
            log(f"메모리 사용량 변화: {final_memory - initial_memory:.2f} MB", "INFO")
        
        return True, f"카테고리 {category} 처리 완료 ({len(image_paths)}개 이미지, {elapsed:.2f}초)"
        
    except Exception as e:
        log(f"실행 중 오류 발생: {e}", "ERROR")
        if DEBUG:
            traceback.print_exc()
        
        # 총 실행 시간
        elapsed = time.time() - start
        log(f"오류로 종료됨. 실행 시간: {elapsed:.2f}초")
        
        return False, f"오류 발생: {str(e)}"

start = time.time() # 시작

# 명령줄 인자 파싱으로 main 함수 실행
def main(category=None, det_model=None, seg_model=None, save_mask_png=None):
    """
    메인 함수 - 명령줄 인자로 실행하거나 다른 모듈에서 호출 가능
    
    Args:
        category (str): 처리할 카테고리 이름 (None인 경우 명령줄 인자 사용)
        det_model (str): 사용할 detection 모델 (None인 경우 기본값 사용)
        seg_model (str): 사용할 segmentation 모델 (None인 경우 기본값 사용)
        save_mask_png (bool): 마스크 PNG 이미지 저장 여부 (None인 경우 명령줄 인자 사용)
    """
    global args
    
    # 인자 설정 - 함수 인자가 None이면 명령줄 인자 사용
    if category is not None:
        args.category = category
    
    # 모델 설정
    if det_model is not None:
        # 모델 명칭 변환 (필요한 경우)
        pass
    
    if seg_model is not None:
        # 모델 명칭 변환 (필요한 경우)
        pass
    
    # 마스크 PNG 저장 설정
    if save_mask_png is not None:
        args.no_mask_png = not save_mask_png
    
    # 메인 처리 함수 호출
    return run_main_launcher(
        category=args.category,
        det_model=det_model or "grounding-dino-2",
        seg_model=seg_model or "sam-2",
        save_mask_png=not args.no_mask_png if save_mask_png is None else save_mask_png,
        batch_size=args.batch_size,
        max_images=args.max_images,
        test_mode=args.test_mode,
        memory_monitor=args.memory_monitor
    )

if __name__ == "__main__":
    success, message = main()
    log(message)
    sys.exit(0 if success else 1)