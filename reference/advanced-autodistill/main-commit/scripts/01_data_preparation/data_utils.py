"""
데이터 관리를 위한 유틸리티 모듈

이 모듈은 카테고리 기반 폴더 구조의 생성 및 관리를 지원합니다.
"""

import sys
import os
import json
import shutil
import yaml
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_utils')

# 기본 경로 설정
# 상대 경로를 사용하여 프로젝트의 이식성 보장
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MATERIAL_DIR = PROJECT_ROOT / "material"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# MCP 관련 설정
MCP_CONFIG = {
    "port": 5678,
    "webhook_url": os.environ.get("WEBHOOK_URL", "https://localhost:5678"),
    "cloudflare": {
        "enabled": os.environ.get("USE_CLOUDFLARE", "false").lower() == "true",
        "tunnel_command": "cloudflared tunnel --url http://localhost:5678"
    }
}

# 표준 카테고리 하위 폴더 구조
STANDARD_SUBFOLDERS = [
    "1.images",
    "2.support-set",
    "3.box",
    "4.mask",
    "5.preprocessed",
    "6.results",
    "7.dataset"
]

def create_category_structure(category_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    지정된 카테고리에 대한 표준화된 폴더 구조를 생성합니다.
    
    Args:
        category_name: 생성할 카테고리의 이름
        base_dir: 카테고리 폴더를 생성할 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        생성된 카테고리 폴더의 경로
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR
    
    # 기본 디렉토리가 존재하지 않으면 생성
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"기본 데이터 디렉토리 생성: {base_dir}")
    
    # 카테고리 경로
    category_path = base_dir / category_name
    
    # 이미 존재하는 카테고리인지 확인
    if category_path.exists():
        logger.warning(f"카테고리 '{category_name}'은(는) 이미 존재합니다: {category_path}")
    else:
        category_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"카테고리 '{category_name}' 생성됨: {category_path}")
    
    # 표준 하위 폴더 생성
    for subfolder in STANDARD_SUBFOLDERS:
        subfolder_path = category_path / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"하위 폴더 생성됨: {subfolder_path}")
    
    return category_path

def get_all_categories(base_dir: Optional[Path] = None) -> List[str]:
    """
    사용 가능한 모든 카테고리 이름의 목록을 반환합니다.
    
    Args:
        base_dir: 카테고리를 검색할 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        카테고리 이름 목록
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR
    
    if not base_dir.exists():
        logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {base_dir}")
        return []
    
    # 카테고리 목록 (각 폴더에 '1.images' 하위 폴더가 있는지 확인하여 유효한 카테고리만 포함)
    categories = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "1.images").exists():
            categories.append(item.name)
    
    return sorted(categories)

def get_categories(base_dir: Optional[Path] = None) -> List[str]:
    """
    get_all_categories의 별칭 함수 - 사용 가능한 모든 카테고리 이름의 목록을 반환합니다.
    
    Args:
        base_dir: 카테고리를 검색할 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        카테고리 이름 목록
    """
    return get_all_categories(base_dir)

def get_files_in_category(category_name: str, subfolder: str, file_ext: Optional[Union[str, List[str]]] = None) -> List[str]:
    """
    특정 카테고리의 하위 폴더에서 파일 목록을 반환합니다.
    
    Args:
        category_name: 카테고리 이름
        subfolder: 하위 폴더 이름 (예: "1.images", "4.mask" 등)
        file_ext: 파일 확장자 또는 확장자 목록 (기본값: None - 모든 파일)
    
    Returns:
        파일 경로 목록
    """
    # 카테고리 경로 가져오기
    category_path = get_category_path(category_name)
    folder_path = category_path / subfolder
    
    # 폴더가 존재하는지 확인
    if not folder_path.exists():
        logger.warning(f"폴더가 존재하지 않습니다: {folder_path}")
        return []
    
    # 확장자 목록 준비
    if file_ext is None:
        file_patterns = ["*.*"]
    elif isinstance(file_ext, str):
        file_patterns = [f"*.{file_ext.lstrip('.')}"]
    else:
        file_patterns = [f"*.{ext.lstrip('.')}" for ext in file_ext]
    
    # 파일 검색
    files = []
    for pattern in file_patterns:
        files.extend([str(f) for f in folder_path.glob(pattern) if f.is_file()])
    
    return sorted(files)

def get_material_categories() -> List[str]:
    """
    Material 폴더 내의 모든 카테고리 이름의 목록을 반환합니다.
    
    Returns:
        카테고리 이름 목록
    """
    material_dir = DEFAULT_MATERIAL_DIR
    
    if not material_dir.exists():
        logger.warning(f"Material 디렉토리가 존재하지 않습니다: {material_dir}")
        return []
    
    # Material 폴더 내 모든 하위 폴더를 카테고리로 간주
    categories = []
    for item in material_dir.iterdir():
        if item.is_dir():
            categories.append(item.name)
    
    return sorted(categories)

def get_category_path(category_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    지정된 카테고리의 경로를 반환합니다.
    
    Args:
        category_name: 카테고리 이름
        base_dir: 카테고리 폴더가 있는 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        카테고리 경로
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_DIR
    
    return base_dir / category_name

def get_material_category_path(category_name: str) -> Path:
    """
    Material 폴더 내 지정된 카테고리의 경로를 반환합니다.
    
    Args:
        category_name: 카테고리 이름
    
    Returns:
        카테고리 경로
    """
    return DEFAULT_MATERIAL_DIR / category_name

def get_subfolder_path(category_name: str, subfolder_type: str, base_dir: Optional[Path] = None) -> Path:
    """
    지정된 카테고리의 특정 하위 폴더 경로를 반환합니다.
    
    Args:
        category_name: 카테고리 이름
        subfolder_type: 하위 폴더 유형 ('1.images', '2.support-set' 등)
        base_dir: 카테고리 폴더가 있는 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        하위 폴더 경로
    """
    category_path = get_category_path(category_name, base_dir)
    return category_path / subfolder_type

def get_images_path(category_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    지정된 카테고리의 이미지 경로를 반환합니다.
    
    Args:
        category_name: 카테고리 이름
        base_dir: 카테고리 폴더가 있는 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        이미지 경로
    """
    return get_subfolder_path(category_name, "1.images", base_dir)

def get_dataset_path(category_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    지정된 카테고리의 YOLO 데이터셋 경로를 반환합니다.
    
    Args:
        category_name: 카테고리 이름
        base_dir: 카테고리 폴더가 있는 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        데이터셋 경로
    """
    return get_subfolder_path(category_name, "7.dataset", base_dir)

def load_class_mapping(category_name: str, base_dir: Optional[Path] = None) -> Dict[str, int]:
    """
    지정된 카테고리의 클래스 매핑을 로드합니다.
    
    Args:
        category_name: 카테고리 이름
        base_dir: 카테고리 폴더가 있는 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        클래스 이름을 클래스 ID에 매핑하는 딕셔너리
    """
    category_path = get_category_path(category_name, base_dir)
    mapping_file = category_path / "class_mapping.json"
    
    if not mapping_file.exists():
        # 지원 세트 디렉토리에서 클래스 이름 추출
        support_dir = category_path / "2.support-set"
        if not support_dir.exists():
            logger.warning(f"지원 세트 디렉토리가 존재하지 않습니다: {support_dir}")
            return {}
        
        # 지원 세트 디렉토리의 하위 폴더를 클래스로 간주
        class_names = [item.name for item in support_dir.iterdir() if item.is_dir()]
        class_mapping = {name: idx for idx, name in enumerate(sorted(class_names))}
        
        # 매핑 파일 저장
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"클래스 매핑 파일 생성됨: {mapping_file}")
    else:
        # 기존 매핑 파일 로드
        with open(mapping_file, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
        
        logger.debug(f"클래스 매핑 파일 로드됨: {mapping_file}")
    
    return class_mapping

def create_data_yaml(category_name: str, class_names: List[str], base_dir: Optional[Path] = None) -> Path:
    """
    YOLO 형식 data.yaml 파일을 생성합니다.
    
    Args:
        category_name: 카테고리 이름
        class_names: 클래스 이름 목록
        base_dir: 카테고리 폴더가 있는 기본 디렉토리 (기본값: 프로젝트 루트의 'data' 폴더)
    
    Returns:
        생성된 data.yaml 파일의 경로
    """
    dataset_path = get_dataset_path(category_name, base_dir)
    yaml_path = dataset_path / "data.yaml"
    
    # data.yaml 구성
    data_config = {
        "path": str(dataset_path),
        "train": str(dataset_path / "train"),
        "val": str(dataset_path / "val"),
        "test": str(dataset_path / "test"),
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    # 파일 작성
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
    
    logger.info(f"YOLO data.yaml 생성됨: {yaml_path}")
    return yaml_path

def update_mcp_config(webhook_url: Optional[str] = None, port: Optional[int] = None) -> Dict:
    """
    MCP 구성을 업데이트합니다.
    
    Args:
        webhook_url: 새 웹훅 URL (Cloudflare 터널 URL)
        port: 새 MCP 포트
    
    Returns:
        업데이트된 MCP 구성
    """
    global MCP_CONFIG
    
    if webhook_url:
        MCP_CONFIG["webhook_url"] = webhook_url
    
    if port:
        MCP_CONFIG["port"] = port
    
    # 구성 파일에 저장
    config_path = PROJECT_ROOT / "config" / "mcp_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(MCP_CONFIG, f, indent=2)
    
    logger.info(f"MCP 구성 업데이트됨: {config_path}")
    return MCP_CONFIG

def get_cloudflare_command() -> str:
    """
    현재 Cloudflare 터널 명령을 반환합니다.
    
    Returns:
        Cloudflare 터널 명령
    """
    port = MCP_CONFIG.get("port", 5678)
    return f"cloudflared tunnel --url http://localhost:{port}"

def generate_docker_command(webhook_url: str, volume_path: str = "E:/Ubuntu_AGI/n8n") -> str:
    """
    Docker 실행 명령을 생성합니다.
    
    Args:
        webhook_url: Cloudflare 터널 URL
        volume_path: 볼륨 마운트 경로
    
    Returns:
        Docker 실행 명령
    """
    return f"""docker run --rm `
  --name n8n-docker1 `
  -p {MCP_CONFIG.get('port', 5678)}:{MCP_CONFIG.get('port', 5678)} `
  -v {volume_path}:/home/node/.n8n `
  -e WEBHOOK_URL={webhook_url} `
  n8nio/n8n:latest"""

def copy_material_to_data(category_name: str, subset: str = None) -> Tuple[int, List[str]]:
    """
    Material 폴더에서 data 폴더로 이미지를 복사합니다.
    
    Args:
        category_name: 카테고리 이름
        subset: 사용하지 않음 (하위 호환성을 위해 유지)
        
    Returns:
        복사된 파일 수와 복사된 파일 경로 목록
    """
    material_category_path = get_material_category_path(category_name)
    target_path = get_images_path(category_name)
    
    # 소스 폴더가 존재하는지 확인
    if not material_category_path.exists():
        logger.error(f"소스 카테고리 폴더가 존재하지 않습니다: {material_category_path}")
        return 0, []
    
    # 대상 폴더 생성 (필요한 경우)
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 복사
    copied_files = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    for file_path in material_category_path.glob('**/*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            target_file = target_path / file_path.name
            shutil.copy2(file_path, target_file)
            copied_files.append(str(target_file))
            logger.debug(f"파일 복사됨: {file_path} -> {target_file}")
    
    logger.info(f"{len(copied_files)}개 파일이 '{material_category_path}'에서 '{target_path}'로 복사되었습니다.")
    return len(copied_files), copied_files

def convert_autodistill_to_yolo(
    category_name: str, 
    class_mapping: Dict[str, int],
    annotations_path: Path,
    img_width: int,
    img_height: int
) -> Tuple[int, List[str]]:
    """
    Autodistill 어노테이션을 YOLO 형식으로 변환합니다.
    
    Args:
        category_name: 카테고리 이름
        class_mapping: 클래스 이름에서 인덱스로의 매핑 딕셔너리
        annotations_path: 어노테이션 파일 경로
        img_width: 이미지 너비
        img_height: 이미지 높이
        
    Returns:
        변환된 어노테이션 파일 수와 생성된 파일 경로 목록
    """
    # 출력 파일 저장 경로 (YOLO 형식 어노테이션)
    box_path = get_subfolder_path(category_name, "3.box")
    if not box_path.exists():
        box_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    
    # JSON 파일 읽기
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # 각 이미지에 대한 어노테이션 처리
    for image_filename, objects in annotations.items():
        # YOLO 형식 출력 파일 경로 (확장자를 .txt로 변경)
        output_filename = Path(image_filename).stem + ".txt"
        output_path = box_path / output_filename
        
        with open(output_path, 'w') as f:
            for obj in objects:
                # 클래스 이름을 클래스 ID로 변환
                class_name = obj.get("class")
                if class_name not in class_mapping:
                    logger.warning(f"알 수 없는 클래스: {class_name}. 건너뜁니다.")
                    continue
                
                class_id = class_mapping[class_name]
                
                # 바운딩 박스 좌표 추출
                bbox = obj.get("bbox")
                if not bbox:
                    continue
                
                # [x1, y1, x2, y2] 형식에서 YOLO 형식으로 변환
                # YOLO 형식: [class_id x_center y_center width height] (정규화된 값)
                x1, y1, x2, y2 = bbox
                
                # 정규화된 중심점 및 크기 계산
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # YOLO 형식으로 기록
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        processed_files.append(str(output_path))
        
    logger.info(f"{len(processed_files)}개 어노테이션 파일이 YOLO 형식으로 변환되었습니다.")
    return len(processed_files), processed_files

def process_classification_results(
    category_name: str,
    classification_results_path: Path,
    confidence_threshold: float = 0.7
) -> Dict[str, List[str]]:
    """
    분류 결과를 처리하고 신뢰도 임계값을 기준으로 필터링합니다.
    
    Args:
        category_name: 카테고리 이름
        classification_results_path: 분류 결과가 포함된 JSON 파일 경로
        confidence_threshold: 신뢰도 임계값 (기본값: 0.7)
        
    Returns:
        클래스별로 분류된 이미지 파일 목록
    """
    # 분류 결과 파일 읽기
    with open(classification_results_path, 'r') as f:
        results = json.load(f)
    
    # 클래스별로 이미지 파일 분류
    class_images = {}
    
    for image_path, classifications in results.items():
        for cls, confidence in classifications.items():
            if confidence >= confidence_threshold:
                if cls not in class_images:
                    class_images[cls] = []
                class_images[cls].append(image_path)
    
    # 결과 저장 및 요약
    summary_path = get_subfolder_path(category_name, "6.results") / "classification_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(class_images, f, indent=2)
    
    logger.info(f"분류 결과 처리 완료 (임계값: {confidence_threshold})")
    for cls, images in class_images.items():
        logger.info(f"  - {cls}: {len(images)}개 이미지")
    
    return class_images

# 기본 함수: 데모 및 테스트용
def demo():
    """데모 함수: 기본 사용법을 보여줍니다."""
    # 표준 폴더 구조 생성
    category_name = "demo_category"
    category_path = create_category_structure(category_name)
    print(f"카테고리 경로: {category_path}")
    
    # 경로 접근 예시
    images_path = get_images_path(category_name)
    print(f"이미지 경로: {images_path}")
    
    # YOLO 데이터셋 구성 예시
    classes = ["car", "person", "bicycle"]
    yaml_path = create_data_yaml(category_name, classes)
    print(f"YOLO 구성 파일: {yaml_path}")
    
    # MCP 구성 업데이트 예시
    update_mcp_config("https://example-tunnel.trycloudflare.com")
    print(f"Cloudflare 명령: {get_cloudflare_command()}")
    print(f"Docker 명령: {generate_docker_command('https://example-tunnel.trycloudflare.com')}")

if __name__ == "__main__":
    demo() 