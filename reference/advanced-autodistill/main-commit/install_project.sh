#!/bin/bash

# Exit on error
set -e

echo "=== PROJECT-AGI Installation Script ==="
echo "This script will install all dependencies for the project."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Check if Python version is at least 3.10
required_version="3.10"
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python version must be at least 3.10"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies
echo "Installing basic dependencies..."
pip install --upgrade pip wheel setuptools

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Check for CUDA availability
echo "Checking for CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Install flash-attention if CUDA is available
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA is available. Installing flash-attention..."
    
    # Determine appropriate flash-attention version based on CUDA and torch versions
    cuda_version=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else '')")
    torch_version=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    python_version_short=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    # Construct URL based on versions
    flash_attn_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    
    echo "Downloading and installing flash-attention from: $flash_attn_url"
    pip install "$flash_attn_url" || echo "Warning: Failed to install flash-attention. The project will still work but may be slower."
else
    echo "CUDA is not available. Skipping flash-attention installation."
    echo "Note: The project will run on CPU, but performance may be limited."
fi

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{test_category/{1.images,2.support-set,3.box,4.mask,5.dataset,6.preprocessed,7.results,8.refine-dataset}}
mkdir -p models/{sam2,yolo}
mkdir -p logs

# Clone segment-anything-2 if needed
if [ ! -d "segment-anything-2" ]; then
    echo "Cloning segment-anything-2 repository..."
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e ".[demo]"
    cd ..
fi

echo "=== Installation Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the project, use:"
echo "  python scripts/main_launcher.py --category test_category"
echo ""
echo "For Few-Shot Learning experiments, use:"
echo "  python scripts/03_classification/run_few_shot_platform.py --webapp"
echo "" 