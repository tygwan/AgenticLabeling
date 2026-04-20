#!/usr/bin/env python3
"""
Few-Shot Learning Results Converter

This script converts few-shot classification results from various experiment formats
into a standardized format that can be used by the Ground Truth Labeler.

Usage:
  python convert_few_shot_results.py --category=<category> --shot=<shot> --threshold=<threshold>
  
Example:
  python convert_few_shot_results.py --category=test_category --shot=5 --threshold=0.7
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("convert_few_shot_results.log")
    ]
)
logger = logging.getLogger(__name__)

# Add project directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert Few-Shot Learning Results')
    parser.add_argument('--category', type=str, required=True, help='Category name')
    parser.add_argument('--shot', type=str, required=True, help='Shot value (e.g., 1, 5, 10)')
    parser.add_argument('--threshold', type=str, required=True, help='Threshold value (e.g., 0.5, 0.7, 0.9)')
    parser.add_argument('--method', type=str, default='all', help='Method name (default: all)')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: auto-generated)')
    return parser.parse_args()

def find_results_files(category, shot, threshold, method='all'):
    """Find results files for the given parameters"""
    data_dir = Path(project_dir) / "data"
    category_dir = data_dir / category
    results_dir = category_dir / "7.results"
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []
    
    results_files = []
    
    # Find method directories
    method_dirs = []
    if method == 'all':
        # Find all method directories
        for item in results_dir.iterdir():
            if item.is_dir() and any(item.name.startswith(prefix) for prefix in ['method', 'clip', 'dino', 'florence']):
                method_dirs.append(item)
    else:
        # Use specified method
        method_dir = results_dir / method
        if method_dir.exists() and method_dir.is_dir():
            method_dirs.append(method_dir)
    
    logger.info(f"Found {len(method_dirs)} method directories: {[d.name for d in method_dirs]}")
    
    # Find results files for each method
    for method_dir in method_dirs:
        # Find shot directories
        shot_dirs = []
        for item in method_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{shot}shot"):
                shot_dirs.append(item)
        
        logger.info(f"Found {len(shot_dirs)} shot directories in {method_dir.name}: {[d.name for d in shot_dirs]}")
        
        # Find threshold files
        for shot_dir in shot_dirs:
            for item in shot_dir.iterdir():
                if item.is_file() and item.name.endswith(".json") and threshold in item.name:
                    results_files.append(item)
    
    logger.info(f"Found {len(results_files)} results files matching shot={shot}, threshold={threshold}")
    return results_files

def convert_results_file(results_file, output_dir=None):
    """Convert a results file to the standardized format"""
    # Default output directory
    if output_dir is None:
        method_name = results_file.parent.parent.name
        shot_name = results_file.parent.name
        threshold_value = results_file.name.split('_')[-1].replace('.json', '')
        output_name = f"{method_name}_{shot_name}_{threshold_value}"
        output_dir = results_file.parent.parent.parent / output_name
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read results file
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Error reading results file {results_file}: {e}")
        return None
    
    # Convert to standardized format
    standardized_results = {}
    
    for image_name, result in results.items():
        # Extract predicted class and confidence
        if isinstance(result, dict):
            # Format 1: {"predicted_class": "class_name", "confidence": 0.95, ...}
            if "predicted_class" in result and "confidence" in result:
                predicted_class = result["predicted_class"]
                confidence = result["confidence"]
            # Format 2: {"class": "class_name", "score": 0.95, ...}
            elif "class" in result and "score" in result:
                predicted_class = result["class"]
                confidence = result["score"]
            # Format 3: {"label": "class_name", "confidence": 0.95, ...}
            elif "label" in result and "confidence" in result:
                predicted_class = result["label"]
                confidence = result["confidence"]
            else:
                logger.warning(f"Unknown result format for {image_name}: {result}")
                continue
        elif isinstance(result, list) and len(result) >= 2:
            # Format 4: ["class_name", 0.95, ...]
            predicted_class = result[0]
            confidence = result[1]
        else:
            logger.warning(f"Unknown result format for {image_name}: {result}")
            continue
        
        # Skip if no predicted class
        if not predicted_class:
            predicted_class = "unknown"
        
        # Create standardized result
        standardized_results[image_name] = {
            "predicted_class": predicted_class,
            "confidence": float(confidence) if confidence else 0.0,
            "original_result": result
        }
    
    # Save standardized results
    output_file = output_dir / "results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(standardized_results, f, indent=2)
        logger.info(f"Saved standardized results to {output_file}")
    except Exception as e:
        logger.error(f"Error saving standardized results to {output_file}: {e}")
        return None
    
    return output_dir

def main():
    """Main function"""
    args = parse_args()
    
    # Find results files
    results_files = find_results_files(args.category, args.shot, args.threshold, args.method)
    
    if not results_files:
        logger.error(f"No results files found for category={args.category}, shot={args.shot}, threshold={args.threshold}")
        return
    
    # Convert each results file
    converted_dirs = []
    for results_file in results_files:
        output_dir = convert_results_file(results_file, args.output_dir)
        if output_dir:
            converted_dirs.append(output_dir)
    
    logger.info(f"Converted {len(converted_dirs)} results files")
    for output_dir in converted_dirs:
        logger.info(f"  - {output_dir}")
    
    logger.info("To use these results with the Ground Truth Labeler:")
    logger.info("1. Open the webapp and go to the 'Ground Truth Labeling' tab")
    logger.info(f"2. Select category '{args.category}'")
    logger.info("3. Select one of the converted experiments from the dropdown")
    logger.info("4. Click 'Load Experiment' to start labeling")

if __name__ == "__main__":
    main() 