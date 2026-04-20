#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Format Converter for Project-AGI

This utility converts data from the original 마스크전처리_class.py format 
to the project-agi format, enabling high-quality preprocessing.

Key features:
1. Convert box.txt files to project-agi box JSON format
2. Convert mask.json files to project-agi mask JSON format
3. Batch conversion with progress tracking
4. Data validation and error handling

Usage:
    python scripts/data_converter.py --input_dir /path/to/original/data --output_dir /path/to/project-agi/data --category test_category
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm
import shutil

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_utils import get_category_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_converter")

class DataFormatConverter:
    """Converter for transforming original data format to project-agi format."""
    
    def __init__(self, input_dir: str, output_category: str):
        """
        Initialize the data converter.
        
        Args:
            input_dir: Directory containing original format data
            output_category: Category name for project-agi structure
        """
        self.input_dir = Path(input_dir)
        self.output_category = output_category
        self.category_path = get_category_path(output_category)
        
        # Set up output directories
        self.output_box_dir = os.path.join(self.category_path, "3.box")
        self.output_mask_dir = os.path.join(self.category_path, "4.mask")
        
        # Create output directories
        os.makedirs(self.output_box_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output category: {self.output_category}")
        logger.info(f"Output box directory: {self.output_box_dir}")
        logger.info(f"Output mask directory: {self.output_mask_dir}")
    
    def parse_box_txt(self, box_file_path: str) -> Dict[str, Any]:
        """
        Parse original box.txt file format.
        
        Args:
            box_file_path: Path to the box.txt file
            
        Returns:
            Parsed box data in project-agi format
        """
        box_data = {
            "image_path": "",
            "boxes": [],
            "class_ids": [],
            "classes": [],
            "confidence": []
        }
        
        try:
            with open(box_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Box:"):
                        # Parse line: "Box: [x1, y1, x2, y2], Class ID: 0, Confidence: 1.0000"
                        parts = line.split(", ")
                        
                        # Extract box coordinates
                        box_str = parts[0].replace("Box: [", "").replace("]", "")
                        box_coords = [float(x) for x in box_str.split(", ")]
                        box_data["boxes"].append(box_coords)
                        
                        # Extract class ID
                        class_id_str = parts[1].replace("Class ID: ", "")
                        class_id = int(class_id_str)
                        box_data["class_ids"].append(class_id)
                        
                        # Extract confidence
                        confidence_str = parts[2].replace("Confidence: ", "")
                        confidence = float(confidence_str)
                        box_data["confidence"].append(confidence)
                        
                        # Map class ID to class name
                        class_names = {
                            0: "fence_person",
                            1: "sidewalk", 
                            2: "car",
                            3: "traffic cone"
                        }
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        box_data["classes"].append(class_name)
        
        except Exception as e:
            logger.error(f"Error parsing box file {box_file_path}: {e}")
        
        return box_data
    
    def convert_mask_json(self, mask_file_path: str) -> List[Dict[str, Any]]:
        """
        Convert original mask.json file format to project-agi format.
        
        Args:
            mask_file_path: Path to the mask.json file
            
        Returns:
            Converted mask data in project-agi format
        """
        try:
            with open(mask_file_path, 'r') as f:
                original_mask_data = json.load(f)
            
            # The original format is already quite compatible
            # Just ensure it has the right structure
            if isinstance(original_mask_data, list):
                return original_mask_data
            else:
                logger.warning(f"Unexpected mask data format in {mask_file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error converting mask file {mask_file_path}: {e}")
            return []
    
    def find_matching_files(self) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Find matching box.txt and mask.json files in the input directory.
        
        Returns:
            List of tuples (image_name, box_file_path, mask_file_path)
        """
        matching_files = []
        
        # Find all box files
        box_files = list(self.input_dir.glob("*_box.txt"))
        
        for box_file in box_files:
            # Extract image name from box file
            image_name = box_file.stem.replace("_box", "")
            
            # Look for corresponding mask file
            mask_file = self.input_dir / f"{image_name}_mask.json"
            mask_file_path = str(mask_file) if mask_file.exists() else None
            
            matching_files.append((image_name, str(box_file), mask_file_path))
        
        logger.info(f"Found {len(matching_files)} sets of matching files")
        return matching_files
    
    def convert_single_file_set(self, image_name: str, box_file_path: str, 
                               mask_file_path: Optional[str]) -> Dict[str, Any]:
        """
        Convert a single set of files (box + mask) to project-agi format.
        
        Args:
            image_name: Base name of the image
            box_file_path: Path to the box.txt file
            mask_file_path: Path to the mask.json file (optional)
            
        Returns:
            Conversion results
        """
        results = {
            "image_name": image_name,
            "box_converted": False,
            "mask_converted": False,
            "errors": []
        }
        
        # Convert box data
        try:
            box_data = self.parse_box_txt(box_file_path)
            box_data["image_path"] = f"{image_name}.jpg"  # Assume jpg extension
            
            # Save converted box data
            output_box_file = os.path.join(self.output_box_dir, f"{image_name}_box.json")
            with open(output_box_file, 'w') as f:
                json.dump(box_data, f, indent=2)
            
            results["box_converted"] = True
            logger.debug(f"Converted box data: {output_box_file}")
            
        except Exception as e:
            error_msg = f"Failed to convert box data for {image_name}: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Convert mask data if available
        if mask_file_path:
            try:
                mask_data = self.convert_mask_json(mask_file_path)
                
                # Save converted mask data
                output_mask_file = os.path.join(self.output_mask_dir, f"{image_name}_mask.json")
                with open(output_mask_file, 'w') as f:
                    json.dump(mask_data, f, indent=2)
                
                results["mask_converted"] = True
                logger.debug(f"Converted mask data: {output_mask_file}")
                
            except Exception as e:
                error_msg = f"Failed to convert mask data for {image_name}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def convert_all_files(self) -> Dict[str, Any]:
        """
        Convert all matching files in the input directory.
        
        Returns:
            Overall conversion results
        """
        matching_files = self.find_matching_files()
        
        overall_results = {
            "total_files": len(matching_files),
            "box_conversions": 0,
            "mask_conversions": 0,
            "errors": []
        }
        
        for image_name, box_file_path, mask_file_path in tqdm(matching_files, desc="Converting files"):
            try:
                result = self.convert_single_file_set(image_name, box_file_path, mask_file_path)
                
                if result["box_converted"]:
                    overall_results["box_conversions"] += 1
                
                if result["mask_converted"]:
                    overall_results["mask_conversions"] += 1
                
                if result["errors"]:
                    overall_results["errors"].extend(result["errors"])
                    
            except Exception as e:
                error_msg = f"Failed to process file set for {image_name}: {e}"
                overall_results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return overall_results
    
    def create_class_mapping(self) -> str:
        """
        Create class mapping file for the category.
        
        Returns:
            Path to the created class mapping file
        """
        class_mapping = {
            "0": "fence_person",
            "1": "sidewalk",
            "2": "car", 
            "3": "traffic cone"
        }
        
        mapping_file = os.path.join(self.category_path, "class_mapping.json")
        
        try:
            with open(mapping_file, 'w') as f:
                json.dump(class_mapping, f, indent=2)
            
            logger.info(f"Created class mapping file: {mapping_file}")
            return mapping_file
            
        except Exception as e:
            logger.error(f"Failed to create class mapping file: {e}")
            return ""
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Create a summary report of conversion results.
        
        Args:
            results: Conversion results
            
        Returns:
            Summary report as string
        """
        report = f"""
Data Format Conversion Summary Report
====================================

Input Directory: {self.input_dir}
Output Category: {self.output_category}

Conversion Results:
- Total File Sets: {results['total_files']}
- Box Files Converted: {results['box_conversions']}
- Mask Files Converted: {results['mask_conversions']}

Output Directories:
- Box Data: {self.output_box_dir}
- Mask Data: {self.output_mask_dir}
"""
        
        if results['errors']:
            report += f"\nErrors ({len(results['errors'])}):\n"
            for error in results['errors'][:10]:  # Show first 10 errors
                report += f"- {error}\n"
            if len(results['errors']) > 10:
                report += f"... and {len(results['errors']) - 10} more errors\n"
        
        return report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Format Converter for Project-AGI")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original format data (box.txt and mask.json files)")
    parser.add_argument("--category", type=str, required=True,
                        help="Category name for project-agi structure")
    parser.add_argument("--output_report", type=str, default=None,
                        help="Path to save conversion report")
    
    return parser.parse_args()

def main():
    """Main function to run the data format converter."""
    args = parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Initialize converter
    converter = DataFormatConverter(
        input_dir=args.input_dir,
        output_category=args.category
    )
    
    # Create class mapping
    converter.create_class_mapping()
    
    # Convert all files
    logger.info("Starting data format conversion...")
    results = converter.convert_all_files()
    
    # Create and display summary report
    report = converter.create_summary_report(results)
    print(report)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.output_report}")
    
    logger.info("Data format conversion completed!")

if __name__ == "__main__":
    main() 