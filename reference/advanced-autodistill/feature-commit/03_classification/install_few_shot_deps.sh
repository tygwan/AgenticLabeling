#!/bin/bash
# Few-Shot Learning Platform Dependencies Installation Script

echo "Installing Few-Shot Learning Platform dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH."
    echo "Please install pip and try again."
    exit 1
fi

# Install dependencies from requirements file
if [ -f "few_shot_requirements.txt" ]; then
    echo "Installing dependencies from few_shot_requirements.txt..."
    pip install -r few_shot_requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies from few_shot_requirements.txt"
        exit 1
    fi
else
    echo "Error: few_shot_requirements.txt not found in the current directory."
    exit 1
fi

# Verify key installations
echo "Verifying key installations..."

# Check for torch
python -c "import torch; print(f'PyTorch installed: {torch.__version__}')" || echo "Warning: Failed to import torch"

# Check for transformers
python -c "import transformers; print(f'Transformers installed: {transformers.__version__}')" || echo "Warning: Failed to import transformers"

# Check for CLIP (this will fail if not installed)
python -c "import clip; print('CLIP installed successfully')" || echo "Warning: Failed to import CLIP. Some models may not work."

# Check for gradio
python -c "import gradio; print(f'Gradio installed: {gradio.__version__}')" || echo "Warning: Failed to import gradio"

echo "Installation completed. You can now run the Few-Shot Learning Platform."
echo "Usage:"
echo "  - Web App: python run_few_shot_platform.py --webapp"
echo "  - CLI Mode: python run_few_shot_platform.py --cli --category <category_name> --model <model_name>" 