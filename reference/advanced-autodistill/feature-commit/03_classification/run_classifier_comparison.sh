#!/bin/bash

# Run Classifier Model Comparison
# This script runs multiple classifier models (ResNet-50 and DINOv2) on the same dataset
# and compares their performance.

# Default settings
CATEGORY="test_category"
MODELS="resnet,dino"
SHOTS="1,5,10,30"
THRESHOLDS="0.6,0.7,0.8,0.9"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --category=*)
      CATEGORY="${1#*=}"
      shift
      ;;
    --models=*)
      MODELS="${1#*=}"
      shift
      ;;
    --shots=*)
      SHOTS="${1#*=}"
      shift
      ;;
    --thresholds=*)
      THRESHOLDS="${1#*=}"
      shift
      ;;
    --create-ground-truth)
      CREATE_GROUND_TRUTH="--create-ground-truth"
      shift
      ;;
    --skip-evaluation)
      SKIP_EVALUATION="--skip-evaluation"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --category=NAME       Dataset category name (default: test_category)"
      echo "  --models=LIST         Comma-separated list of models (default: resnet,dino)"
      echo "  --shots=LIST          Comma-separated list of shot values (default: 1,5,10,30)"
      echo "  --thresholds=LIST     Comma-separated list of threshold values (default: 0.6,0.7,0.8,0.9)"
      echo "  --create-ground-truth Analyze and prepare ground truth data"
      echo "  --skip-evaluation     Skip evaluation after experiments"
      echo "  --help                Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage information."
      exit 1
      ;;
  esac
done

# Display settings
echo "======= Classifier Model Comparison ======="
echo "Category:   $CATEGORY"
echo "Models:     $MODELS"
echo "Shots:      $SHOTS"
echo "Thresholds: $THRESHOLDS"
echo "=========================================="

# Run the Python script
python scripts/run_model_comparison.py \
  --category="$CATEGORY" \
  --models="$MODELS" \
  --shots="$SHOTS" \
  --thresholds="$THRESHOLDS" \
  $CREATE_GROUND_TRUTH $SKIP_EVALUATION

# Check if the command was successful
if [ $? -eq 0 ]; then
  echo "======= Comparison Completed Successfully ======="
  echo "Results saved in data/$CATEGORY/7.results/"
  echo "Model comparison visualizations in data/$CATEGORY/7.results/model_comparison/"
else
  echo "Error: Comparison failed. Check logs for details."
  exit 1
fi 