# Few-Shot Classifier Model Comparison

This document explains how to use the model comparison functionality to evaluate and compare different classifier models (ResNet-50, DINOv2, CLIP) for few-shot image classification.

## Overview

The model comparison system allows you to:

1. Run multiple classifier models on the same dataset
2. Store results separately for each model in model-specific directories
3. Compare classification performance across models
4. Analyze which model performs best for different shot/threshold combinations

## Available Models

- **ResNet-50**: A deep residual network pre-trained on ImageNet
- **DINOv2**: Self-supervised vision transformer (ViT-B/14) 
- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model

## Directory Structure

Results are organized as follows:

```
data/
└── test_category/
    └── 7.results/
        ├── resnet/                  # ResNet-50 model results
        │   ├── shot_1/
        │   │   ├── threshold_0.60/
        │   │   └── ...
        │   ├── shot_5/
        │   └── ...
        ├── dino/                    # DINOv2 model results
        │   ├── shot_1/
        │   └── ...
        ├── clip/                    # CLIP model results (if used)
        │   └── ...
        ├── ground_truth/            # Ground truth data (shared)
        │   ├── Class_0/
        │   ├── Class_1/
        │   └── ...
        ├── model_comparison/        # Comparative analysis
        │   ├── model_comparison_shot_1.png
        │   ├── model_comparison_shot_5.png
        │   ├── best_model_heatmap.png
        │   └── best_rate_heatmap.png
        └── all_experiment_summary.json  # Combined results from all models
```

## Running Model Comparison

### Using the Shell Script

The easiest way to run model comparison is using the provided shell script:

```bash
# Run with default settings (ResNet + DINOv2, shots 1,5,10,30, thresholds 0.6-0.9)
./scripts/run_classifier_comparison.sh

# Run with custom parameters
./scripts/run_classifier_comparison.sh --category=test_category --models=resnet,dino --shots=5,10 --thresholds=0.7,0.8
```

### Using Python Directly

For more control, you can run the Python script directly:

```bash
python scripts/run_model_comparison.py --category test_category --models resnet,dino --shots 1,5,10,30 --thresholds 0.6,0.7,0.8,0.9
```

### Command Line Options

Both the shell script and Python script support the following options:

- `--category`: Dataset category name (default: test_category)
- `--models`: Comma-separated list of models to compare (default: resnet,dino)
- `--shots`: Comma-separated list of shot values (default: 1,5,10,30)
- `--thresholds`: Comma-separated list of threshold values (default: 0.6,0.7,0.8,0.9)
- `--create-ground-truth`: Analyze and prepare ground truth data
- `--skip-evaluation`: Skip evaluation after experiments

## Analyzing Results

### Model Comparison Visualizations

After running the comparison, you can find visualizations in the `7.results/model_comparison/` directory:

1. **Classification Rate Plots**: `model_comparison_shot_X.png` shows how different models perform at each threshold for a specific shot value.

2. **Best Model Heatmap**: `best_model_heatmap.png` shows which model performs best for each shot/threshold combination.

3. **Best Classification Rate Heatmap**: `best_rate_heatmap.png` shows the highest classification rate achieved for each shot/threshold combination.

### Individual Model Results

Each model has its own directory with detailed results:

- `predictions.csv`: All classification results
- `evaluation_results.json`: Performance metrics
- Confusion matrices and visualization graphs
- Class-specific results

## Example Workflow

1. **Prepare the dataset**:
   - Ensure your dataset is organized in the standard structure
   - Prepare support sets (2.support-set/shotX/class_Y)
   - Ensure preprocessed images are available (6.preprocessed/Class_Y)

2. **Run model comparison**:
   ```bash
   ./scripts/run_classifier_comparison.sh --category=test_category --models=resnet,dino
   ```

3. **Set up ground truth** (optional):
   - Create a ground truth directory: `7.results/ground_truth/`
   - Add correctly labeled images to class folders: `Class_0`, `Class_1`, etc.
   - Run ground truth analysis:
   ```bash
   ./scripts/run_classifier_comparison.sh --category=test_category --create-ground-truth
   ```

4. **Analyze and compare models**:
   - Examine the visualizations in `7.results/model_comparison/`
   - Check individual model results for detailed performance metrics

## Tips

1. **Start with small experiments**: Begin with fewer shot/threshold combinations (e.g., `--shots=1,5 --thresholds=0.7,0.8`) to save time.

2. **Use consistent ground truth**: The ground truth data is shared between all models, ensuring fair comparison.

3. **Check model-specific requirements**: Some models may require additional dependencies:
   - CLIP: `pip install 'clip @ git+https://github.com/openai/CLIP.git'`
   - DINOv2: `pip install transformers`
   - ResNet: Requires PyTorch and torchvision

4. **Storage considerations**: Running many models with multiple shot/threshold combinations can generate significant data. Consider cleaning up old results if needed. 