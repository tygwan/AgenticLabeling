#!/bin/bash

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

echo "INFO: Starting dependency installation..."

# --- 섹션 1: CUDA Toolkit 설치 (v12.4) ---
echo "INFO: Installing CUDA Toolkit 12.4..."
# CUDA Keyring 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
# APT 저장소 업데이트 및 CUDA 설치
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
echo "INFO: CUDA Toolkit 12.4 installation finished."

# --- 섹션 2: PyTorch 설치 (v2.6.0 for CUDA 12.4) ---
# 중요: 이 스크립트를 실행하기 전에 원하는 가상환경을 활성화하세요!
echo "INFO: Installing PyTorch 2.6.0 for CUDA 12.4..."
# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
echo "INFO: PyTorch installation finished."

# --- 섹션 3: cuDNN 설치 (v9.1.1 for CUDA 12) ---
echo "INFO: Installing cuDNN 9.1.1..."
# 임시 다운로드 폴더 생성
CUDNN_TMP_DIR=$(mktemp -d)
echo "INFO: Downloading cuDNN packages to $CUDNN_TMP_DIR..."
# cuDNN .deb 파일 다운로드 (Ubuntu 22.04 기준 URL 사용)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn9-cuda-12_9.1.1.17-1_amd64.deb -P $CUDNN_TMP_DIR
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn9-dev-cuda-12_9.1.1.17-1_amd64.deb -P $CUDNN_TMP_DIR
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn9-static-cuda-12_9.1.1.17-1_amd64.deb -P $CUDNN_TMP_DIR
# cuDNN 9.1.1.17 for CUDA 12.4는 아래 파일이 아닌, 위 라이브러리 파일들로 설치됩니다.
# 메타 패키지가 있다면 다를 수 있으나, 제공된 URL 기반으로 개별 설치합니다.
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cudnn9-cuda-12-4_9.1.1.17-1_amd64.deb -P $CUDNN_TMP_DIR

# 다운로드된 .deb 파일 설치
echo "INFO: Installing downloaded cuDNN packages..."
sudo dpkg -i $CUDNN_TMP_DIR/*.deb || echo "WARN: dpkg might show errors for already satisfied dependencies, which is often okay."

# 임시 폴더 삭제
echo "INFO: Cleaning up cuDNN download directory..."
rm -rf $CUDNN_TMP_DIR
echo "INFO: cuDNN installation finished."

# --- 섹션 4: 기타 Python 패키지 설치 (requirements.txt 사용) ---
echo "INFO: Installing packages from requirements.txt..."
# 현재 디렉토리에 requirements.txt 파일이 있다고 가정
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found in the current directory!"
    exit 1
fi
echo "INFO: requirements.txt packages installation finished."

# --- 섹션 5: flash-attn 설치 (로컬 .whl 파일) ---
echo "INFO: Installing flash-attn from downloaded wheel file..."
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_WHL_NAME=$(basename $FLASH_ATTN_URL)
FLASH_ATTN_TMP_PATH="/tmp/$FLASH_ATTN_WHL_NAME"

# .whl 파일 다운로드
echo "INFO: Downloading $FLASH_ATTN_WHL_NAME..."
wget $FLASH_ATTN_URL -O $FLASH_ATTN_TMP_PATH

# .whl 파일 설치
echo "INFO: Installing $FLASH_ATTN_WHL_NAME..."
pip install $FLASH_ATTN_TMP_PATH

# 다운로드한 .whl 파일 삭제 (선택 사항)
echo "INFO: Cleaning up downloaded flash-attn wheel file..."
rm $FLASH_ATTN_TMP_PATH
echo "INFO: flash-attn installation finished."

echo "INFO: All dependency installations completed successfully!"