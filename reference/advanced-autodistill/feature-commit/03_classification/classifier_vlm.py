#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM (Florence-2) Classification Module

This script implements image classification using Florence-2 VLM capabilities
directly on preprocessed images.

Key features:
1. Load preprocessed images and caption ontology
2. Use Florence-2 VLM to classify images into defined classes
3. Save classification results

Usage:
    python classifier_vlm.py --category <category_name> --confidence <threshold>
                            [--input_dir <input_directory>] [--output_dir <output_directory>]

# Florence-2 VLM 분류기 실행
python scripts/classifier_vlm.py --category test_category --confidence 0.5 --visualize

"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import project utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_utils import get_category_path, load_class_mapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vlm_classifier")

class Florence2Classifier:
    """Classification using Florence-2 Visual Language Model."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the Florence-2 classifier.
        
        Args:
            confidence_threshold: Minimum confidence threshold for classification
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.device = None
        self.class_mapping = None
        self.class_descriptions = {}
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Florence-2 model and processor."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            logger.info("Loading Florence-2 model...")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load model and processor
            self.processor = AutoProcessor.from_pretrained("microsoft/florence-2-base")
            self.model = AutoModelForVision2Seq.from_pretrained("microsoft/florence-2-base")
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Florence-2 model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Error importing required libraries: {e}")
            logger.error("Please install the required packages: pip install transformers torch")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading Florence-2 model: {e}")
            sys.exit(1)
    
    def load_class_definitions(self, category_name: str):
        """
        Load class definitions for the given category.
        
        Args:
            category_name: Name of the category
        """
        try:
            # Load the class mapping
            self.class_mapping = load_class_mapping(category_name)
            
            # Get class descriptions from class names
            for class_name in self.class_mapping.keys():
                # For demonstration, just use the class name as the description
                # In a real application, you might have more detailed descriptions
                self.class_descriptions[class_name] = class_name.replace('_', ' ')
                
            logger.info(f"Loaded {len(self.class_descriptions)} class definitions for category '{category_name}'")
            
        except Exception as e:
            logger.error(f"Error loading class definitions for category '{category_name}': {e}")
            raise
    
    def prepare_prompts(self) -> List[str]:
        """
        Prepare classification prompts for Florence-2.
        
        Returns:
            List of prompt templates
        """
        prompts = []
        
        # Prompt templates for classification tasks
        prompts.append("What is shown in this image?")
        prompts.append("Classify this image into one of the following categories: {categories}.")
        prompts.append("Does this image contain any of the following: {categories}?")
        prompts.append("This image shows:")
        
        return prompts
    
    def format_prompt_with_classes(self, prompt_template: str) -> str:
        """
        Format a prompt template with available classes.
        
        Args:
            prompt_template: Prompt template string
            
        Returns:
            Formatted prompt
        """
        if "{categories}" in prompt_template:
            categories_list = ", ".join(self.class_descriptions.values())
            return prompt_template.format(categories=categories_list)
        
        return prompt_template
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a single image using Florence-2 VLM.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Classification result with class and confidence
        """
        import torch
        
        try:
            # Load the image
            image = Image.open(image_path).convert("RGB")
            
            best_class = None
            best_confidence = 0.0
            all_results = {}
            
            # Try different prompt templates for better results
            prompts = self.prepare_prompts()
            
            for prompt_template in prompts:
                prompt = self.format_prompt_with_classes(prompt_template)
                
                # Process the image with the text prompt
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                
                # Generate output
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=30,
                        num_beams=5,
                        early_stopping=True
                    )
                
                # Decode the output
                generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Analyze the generated text to find matches with our classes
                # This is a simple implementation; in a real application, 
                # you might use more sophisticated text matching or classification
                class_matches = self._find_class_matches(generated_text)
                
                # Update the best class if needed
                for class_name, confidence in class_matches.items():
                    all_results[class_name] = max(confidence, all_results.get(class_name, 0))
                    
                    if confidence > best_confidence:
                        best_class = class_name
                        best_confidence = confidence
            
            # Apply threshold
            if best_confidence < self.confidence_threshold:
                best_class = None
                
            return {
                "image": image_path,
                "class": best_class,
                "score": float(best_confidence),
                "all_scores": {cls: float(score) for cls, score in all_results.items()},
                "raw_output": generated_text
            }
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            return {
                "image": image_path,
                "class": None,
                "score": 0.0,
                "all_scores": {},
                "error": str(e)
            }
    
    def _find_class_matches(self, generated_text: str) -> Dict[str, float]:
        """
        Find matches between generated text and class descriptions.
        
        Args:
            generated_text: Text generated by Florence-2
            
        Returns:
            Dictionary mapping class names to confidence scores
        """
        matches = {}
        
        # Split the text into tokens for better matching
        tokens = generated_text.lower().split()
        
        # Replace common separators with spaces
        cleaned_text = generated_text.lower().replace(',', ' ').replace('.', ' ').replace(':', ' ')
        
        # Check for each class name/description in the generated text
        for class_name, description in self.class_descriptions.items():
            # Check for exact match
            if description.lower() in cleaned_text or class_name.lower().replace('_', ' ') in cleaned_text:
                matches[class_name] = 0.95  # High confidence for exact match
                continue
                
            # Check for partial matches in tokens
            class_tokens = description.lower().split()
            matched_tokens = sum(1 for token in class_tokens if token in tokens)
            
            if matched_tokens > 0:
                # Calculate confidence based on how many tokens matched
                confidence = matched_tokens / len(class_tokens) * 0.8  # Max 0.8 for token matches
                matches[class_name] = confidence
        
        return matches
        
    def classify_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple images in batch.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of classification results
        """
        results = []
        for img_path in tqdm(image_paths, desc="Classifying images with Florence-2"):
            result = self.classify_image(img_path)
            results.append(result)
            
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save classification results to a JSON file.
        
        Args:
            results: List of classification results
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def visualize_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        Visualize classification results with images and confidence scores.
        
        Args:
            results: List of classification results
            output_dir: Directory to save visualization images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            image_path = result["image"]
            predicted_class = result["class"]
            confidence = result["score"]
            raw_output = result.get("raw_output", "")
            
            # Load and display the image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(img)
            
            # Add title with prediction and confidence
            if predicted_class:
                title = f"Predicted: {predicted_class} ({confidence:.2f})"
            else:
                title = f"No class assigned (max score: {confidence:.2f})"
                
            plt.title(title)
            plt.axis('off')
            
            # Add raw output and scores as text
            all_scores = result["all_scores"]
            text = f"VLM Output: {raw_output}\n\nScores:\n"
            for cls, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                text += f"{cls}: {score:.2f}\n"
            
            plt.figtext(0.02, 0.02, text, fontsize=8, wrap=True)
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"result_{i}.jpg")
            plt.savefig(output_path)
            plt.close()
            
        logger.info(f"Visualizations saved to {output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Florence-2 VLM Image Classification")
    
    parser.add_argument("--category", type=str, required=True,
                        help="Category name")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for classification")
    parser.add_argument("--input_dir", type=str,
                        help="Directory containing images to classify. If not specified, uses category's preprocessed images.")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save results. If not specified, uses category's results directory.")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of classification results")
    
    return parser.parse_args()

def main():
    """Main function to run the Florence-2 VLM classifier."""
    args = parse_args()
    
    # Get the category path
    category_path = get_category_path(args.category)
    
    # Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = os.path.join(category_path, "6.preprocessed")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(category_path, "7.results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files to classify
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(image_files)} images to classify")
    
    # Initialize the classifier
    classifier = Florence2Classifier(confidence_threshold=args.confidence)
    
    # Load class definitions
    classifier.load_class_definitions(args.category)
    
    # Classify images
    start_time = time.time()
    results = classifier.classify_batch(image_files)
    elapsed_time = time.time() - start_time
    
    # Save results
    output_file = os.path.join(output_dir, "florence2_vlm_results.json")
    classifier.save_results(results, output_file)
    
    # Visualize results if requested
    if args.visualize:
        viz_dir = os.path.join(output_dir, "visualizations")
        classifier.visualize_results(results, viz_dir)
    
    # Print summary
    classes = set()
    classified_count = 0
    
    for result in results:
        if result["class"]:
            classified_count += 1
            classes.add(result["class"])
    
    logger.info(f"Classification complete. Summary:")
    logger.info(f"  Total images: {len(results)}")
    logger.info(f"  Classified images: {classified_count}")
    logger.info(f"  Unclassified images: {len(results) - classified_count}")
    logger.info(f"  Classes found: {len(classes)}")
    logger.info(f"  Total time: {elapsed_time:.2f} seconds")
    logger.info(f"  Average time per image: {elapsed_time/len(results):.2f} seconds")
    logger.info(f"  Results saved to: {output_file}")

if __name__ == "__main__":
    main() 