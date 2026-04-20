#!/usr/bin/env python3
"""
AutoDistill Enhanced Setup Script

This script installs dependencies, configures the environment, and sets up
the enhanced autodistill modules for the project-agi pipeline.
"""

import os
import sys
import subprocess
import platform
import logging
import argparse
import shutil
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup")

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Model directories
SAM2_DIR = MODELS_DIR / "sam2"
FLORENCE_DIR = MODELS_DIR / "florence"
YOLO_DIR = MODELS_DIR / "yolo"

# Ensure directories exist
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAM2_DIR, exist_ok=True)
os.makedirs(FLORENCE_DIR, exist_ok=True)
os.makedirs(YOLO_DIR, exist_ok=True)

def run_command(cmd, description=None, capture_output=False):
    """
    Run a command and log output
    
    Args:
        cmd: Command to run (list of strings)
        description: Description of the command
        capture_output: Whether to capture and return output
        
    Returns:
        subprocess.CompletedProcess: Result of subprocess.run
    """
    if description:
        logger.info(description)
    
    try:
        if capture_output:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
        else:
            result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}")
            if capture_output and result.stderr:
                logger.error(f"Error: {result.stderr}")
        
        return result
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return None

def check_python_version():
    """
    Check if Python version is compatible
    
    Returns:
        bool: True if compatible, False otherwise
    """
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        logger.error(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    return True

def check_gpu_support():
    """
    Check if CUDA is available for GPU support
    
    Returns:
        bool: True if GPU support is available, False otherwise
    """
    logger.info("Checking GPU support...")
    
    try:
        # Try to import torch
        if importlib.util.find_spec("torch") is None:
            logger.warning("PyTorch not installed, skipping GPU check")
            return False
        
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU support available: {gpu_count} devices")
            logger.info(f"GPU 0: {gpu_name}")
            return True
        else:
            logger.warning("CUDA is not available, will use CPU (slower)")
            return False
    except Exception as e:
        logger.warning(f"Error checking GPU support: {e}")
        return False

def install_requirements(requirements_file="requirements.txt", upgrade=False):
    """
    Install requirements from requirements.txt
    
    Args:
        requirements_file: Path to requirements file
        upgrade: Whether to upgrade existing packages
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Installing requirements from {requirements_file}...")
    
    if not os.path.exists(requirements_file):
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    if upgrade:
        cmd.append("--upgrade")
    
    result = run_command(cmd, f"Installing from {requirements_file}")
    return result and result.returncode == 0

def install_autodistill_packages(upgrade=False):
    """
    Install autodistill packages directly
    
    Args:
        upgrade: Whether to upgrade existing packages
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Installing autodistill packages...")
    
    packages = [
        "autodistill",
        "autodistill-grounded-sam-2",
        "autodistill-yolov8",
        "autodistill-florence-2"
    ]
    
    for package in packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        if upgrade:
            cmd.append("--upgrade")
        
        result = run_command(cmd, f"Installing {package}")
        if not result or result.returncode != 0:
            logger.error(f"Failed to install {package}")
            return False
    
    return True

def validate_installation():
    """
    Validate that all required packages are installed
    
    Returns:
        bool: True if all packages are installed, False otherwise
    """
    logger.info("Validating installation...")
    
    required_packages = [
        "opencv-python",
        "numpy",
        "torch",
        "autodistill",
        "autodistill_grounded_sam_2",
        "autodistill_florence_2",
        "autodistill_yolov8"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package.replace('-', '_'))
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("The following packages are missing:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        return False
    
    return True

def setup_test_structure():
    """
    Set up the test category directory structure
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Setting up test category directory structure...")
    
    test_dir = PROJECT_ROOT / "data" / "test_category"
    
    # Create main directories
    directories = [
        test_dir / "1.images",
        test_dir / "2.support-set",
        test_dir / "3.box",
        test_dir / "4.mask",
        test_dir / "5.dataset",
        test_dir / "6.preprocessed",
        test_dir / "7.results",
        test_dir / "8.refine-dataset"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create support-set class directories
    classes = ["car", "fence_person", "sidewalk", "traffic cone"]
    for class_name in classes:
        os.makedirs(test_dir / "2.support-set" / class_name, exist_ok=True)
    
    logger.info("Test category directory structure set up successfully")
    return True

def check_or_run_patches():
    """
    Check if model patches are working correctly
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Checking model patches...")
    
    try:
        # Try to import the custom_helpers module
        sys.path.append(str(SCRIPTS_DIR))
        from custom_helpers import patch_all_models
        
        # Run the patches
        results = patch_all_models()
        
        # Check results
        if all(results.values()):
            logger.info("All model patches applied successfully")
            return True
        else:
            logger.warning("Some model patches failed:")
            for model, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                logger.warning(f"  {model}: {status}")
            return False
    except Exception as e:
        logger.error(f"Error checking model patches: {e}")
        return False

def run_test_detection(test_image=None):
    """
    Run a test detection to verify the setup
    
    Args:
        test_image: Path to test image (None to skip)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not test_image:
        logger.info("Skipping test detection (no test image provided)")
        return True
    
    logger.info(f"Running test detection on {test_image}...")
    
    try:
        # Try to import the advanced_autodistill module
        sys.path.append(str(SCRIPTS_DIR))
        from advanced_autodistill import AdvancedAutodistill
        
        # Create processor
        processor = AdvancedAutodistill(category="test_category", verbose=True)
        
        # Initialize models
        if not processor.initialize_models():
            logger.error("Failed to initialize models")
            return False
        
        # Copy test image to data directory if it exists
        if os.path.exists(test_image):
            dest_dir = PROJECT_ROOT / "data" / "test_category" / "1.images"
            dest_file = dest_dir / os.path.basename(test_image)
            shutil.copy2(test_image, dest_file)
            logger.info(f"Copied test image to {dest_file}")
            
            # Process single image
            result = processor.process_single_image(str(dest_file))
            
            if result["success"]:
                logger.info("Test detection successful")
                logger.info(f"Detected {result['detections']} objects")
                return True
            else:
                logger.error(f"Test detection failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            logger.warning(f"Test image not found: {test_image}")
            return False
    except Exception as e:
        logger.error(f"Error running test detection: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AutoDistill Enhanced Setup")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages")
    parser.add_argument("--test-image", type=str, help="Path to test image for verification")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")
    args = parser.parse_args()
    
    logger.info("Starting AutoDistill Enhanced Setup")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check GPU support
    check_gpu_support()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_requirements(upgrade=args.upgrade):
            logger.error("Failed to install requirements")
            return 1
        
        if not install_autodistill_packages(upgrade=args.upgrade):
            logger.error("Failed to install autodistill packages")
            return 1
    
    # Validate installation
    if not args.skip_validation and not validate_installation():
        logger.error("Installation validation failed")
        return 1
    
    # Set up test structure
    if not setup_test_structure():
        logger.error("Failed to set up test structure")
        return 1
    
    # Check model patches
    if not check_or_run_patches():
        logger.warning("Model patches check failed, models may use default paths")
    
    # Run test detection
    if args.test_image:
        if not run_test_detection(args.test_image):
            logger.error("Test detection failed")
            return 1
    
    logger.info("AutoDistill Enhanced Setup completed successfully")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Add images to the data/test_category/1.images directory")
    logger.info("2. Run the pipeline with: python scripts/enhanced_main_launcher.py")
    logger.info("3. Explore the results in the data/test_category directory")
    logger.info("")
    logger.info("For more options, run: python scripts/enhanced_main_launcher.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 