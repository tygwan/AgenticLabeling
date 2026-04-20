"""
Custom helpers for SAM2 model

This module provides a modified version of the SAM2 model loader
that uses a project-local directory instead of user cache.
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. GroundingDINO and SAM will run very slowly.")


def load_SAM_local():
    """
    Load SAM2 model from the project's models directory instead of ~/.cache.
    
    Returns:
        SAM2ImagePredictor: The initialized predictor
    """
    # 프로젝트 루트 디렉토리 설정
    cur_dir = os.getcwd()
    project_root = Path(cur_dir)
    
    # 프로젝트 내의 모델 디렉토리 지정
    SAM_DIR = project_root / "models" / "sam2"
    SAM_CODE_DIR = SAM_DIR / "segment-anything-2"
    SAM_CHECKPOINT_PATH = SAM_DIR / "sam2_hiera_base_plus.pth"
    
    # 모델 다운로드 URL
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

    # 디렉토리 생성
    os.makedirs(SAM_DIR, exist_ok=True)
    
    print(f"SAM2 모델 디렉토리: {SAM_DIR}")
    
    # 작업 디렉토리 변경
    os.chdir(SAM_DIR)
    
    # 저장소 클론
    if not SAM_CODE_DIR.exists():
        print(f"SAM2 코드 클론 중: {SAM_CODE_DIR}")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/segment-anything-2.git",
            ]
        )
        
        os.chdir(SAM_CODE_DIR)
        
        print("SAM2 패키지 설치 중...")
        subprocess.run(["pip", "install", "-e", "."])
    
    # 모듈 import를 위한 경로 추가
    sys.path.append(str(SAM_CODE_DIR))
    
    # 체크포인트 다운로드
    if not SAM_CHECKPOINT_PATH.exists():
        print(f"SAM2 체크포인트 다운로드 중: {SAM_CHECKPOINT_PATH}")
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        print(f"SAM2 모델 초기화: {SAM_CHECKPOINT_PATH}")
        model_cfg = "sam2_hiera_b+.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, str(SAM_CHECKPOINT_PATH)))
    except ImportError as e:
        print(f"SAM2 모듈 로드 실패: {e}")
        raise
    
    # 원래 작업 디렉토리로 복귀
    os.chdir(cur_dir)
    
    return predictor


# GroundedSAM2 클래스의 monkey patching을 위한 함수 (사용자 정의 모델 로더 사용)
def patch_grounded_sam2():
    """
    autodistill_grounded_sam_2 패키지의 GroundedSAM2 클래스를 패치하여
    프로젝트 내부 디렉토리에서 SAM 모델을 로드하도록 합니다.
    
    반드시 GroundedSAM2를 import하기 전에 호출해야 합니다.
    """
    try:
        import autodistill_grounded_sam_2.helpers
        
        # 기존 함수 백업
        original_load_SAM = autodistill_grounded_sam_2.helpers.load_SAM
        
        # 함수 대체
        autodistill_grounded_sam_2.helpers.load_SAM = load_SAM_local
        
        print("SAM2 모델 로더가 성공적으로 패치되었습니다!")
        return True
    except Exception as e:
        print(f"SAM2 모델 로더 패치 실패: {e}")
        return False 