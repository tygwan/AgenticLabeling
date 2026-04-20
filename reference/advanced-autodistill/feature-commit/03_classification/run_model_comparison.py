#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Runner

This script runs multiple classifier models (ResNet-50 and DINOv2) on the same dataset
with identical shot and threshold settings, then compares their performance.

Usage:
    python run_model_comparison.py --category test_category --shots 1,5,10,30 --thresholds 0.6,0.7,0.8

Results will be saved separately for each model in the 7.results/{model_name} directory,
and comparative analysis will be stored in 7.results/model_comparison.
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_comparison")

# Import project utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.classifier_cosine_experiment import FewShotExperiment
from scripts.data_utils import get_category_path

def run_experiments(category: str, n_shots: List[int], thresholds: List[float], models: List[str], 
                   input_dir: str = None, run_evaluation: bool = True, create_ground_truth: bool = False):
    """
    Run all model experiments with the same shot and threshold settings
    
    Args:
        category: Dataset category name
        n_shots: List of shot values to test
        thresholds: List of threshold values to test
        models: List of models to compare (e.g., ["resnet", "dino"])
        input_dir: Custom input directory (default: category's preprocessed dir)
        run_evaluation: Whether to run evaluation after experiments
        create_ground_truth: Whether to analyze and prepare ground truth data
    """
    category_path = get_category_path(category)
    base_results_dir = os.path.join(category_path, "7.results")
    
    # Create ground truth directory if it doesn't exist
    ground_truth_dir = os.path.join(base_results_dir, "ground_truth")
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    # Ensure model comparison directory exists
    model_comparison_dir = os.path.join(base_results_dir, "model_comparison")
    os.makedirs(model_comparison_dir, exist_ok=True)
    
    # Run experiments for each model
    for model in models:
        logger.info(f"==== Running experiments for {model.upper()} model ====")
        experiment = FewShotExperiment(category, model)
        
        # Set custom shot and threshold values
        experiment.n_shots = n_shots
        experiment.thresholds = thresholds
        
        logger.info(f"Running {len(n_shots) * len(thresholds)} experiments with {model} model")
        logger.info(f"Shot values: {n_shots}")
        logger.info(f"Threshold values: {thresholds}")
        
        # Run all experiments
        start_time = time.time()
        experiment.run_all_experiments(input_dir)
        end_time = time.time()
        logger.info(f"{model} model experiments completed in {end_time - start_time:.2f} seconds")
    
    # Create and analyze ground truth data if requested
    if create_ground_truth:
        logger.info("Analyzing ground truth data...")
        # Use the last experiment instance to analyze ground truth
        experiment.analyze_ground_truth(ground_truth_dir)
    
    # Run comparison analysis to generate comparison visualizations
    logger.info("Generating model comparison analysis...")
    for model in models:
        experiment = FewShotExperiment(category, model)
        experiment._compare_with_other_models()
    
    # Run evaluation for each model-shot-threshold combination if requested
    if run_evaluation:
        logger.info("Running evaluation for all experiments...")
        for model in models:
            experiment = FewShotExperiment(category, model)
            for n_shot in n_shots:
                for threshold in thresholds:
                    experiment_id = f"shot_{n_shot}_threshold_{threshold:.2f}"
                    logger.info(f"Evaluating {model} model, {experiment_id}")
                    experiment.evaluate_experiment(experiment_id)
    
    logger.info(f"All experiments and comparisons completed. Results saved in {base_results_dir}")
    logger.info(f"Model comparison visualizations available in {model_comparison_dir}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run and compare multiple classifier models")
    
    parser.add_argument("--category", type=str, default="test_category",
                      help="Dataset category name (default: test_category)")
    parser.add_argument("--models", type=str, default="resnet,dino",
                      help="Comma-separated list of models to compare (default: 'resnet,dino')")
    parser.add_argument("--shots", type=str, default="1,5,10,30",
                      help="Comma-separated list of shot values (default: '1,5,10,30')")
    parser.add_argument("--thresholds", type=str, default="0.6,0.7,0.8,0.9",
                      help="Comma-separated list of threshold values (default: '0.6,0.7,0.8,0.9')")
    parser.add_argument("--input-dir", type=str,
                      help="Custom input directory for query images (default: category's preprocessed dir)")
    parser.add_argument("--skip-evaluation", action="store_true",
                      help="Skip evaluation after experiments")
    parser.add_argument("--create-ground-truth", action="store_true",
                      help="Analyze and prepare ground truth data")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Parse shot values
    n_shots = [int(n) for n in args.shots.split(',')]
    
    # Parse threshold values
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # Parse model list
    models = [m.strip().lower() for m in args.models.split(',')]
    
    # Validate models
    valid_models = ["resnet", "dino", "clip"]
    for model in models:
        if model not in valid_models:
            logger.error(f"Invalid model: {model}. Valid options are: {', '.join(valid_models)}")
            return
    
    # Run experiments
    run_experiments(
        category=args.category,
        n_shots=n_shots,
        thresholds=thresholds,
        models=models,
        input_dir=args.input_dir,
        run_evaluation=not args.skip_evaluation,
        create_ground_truth=args.create_ground_truth
    )

if __name__ == "__main__":
    main() 