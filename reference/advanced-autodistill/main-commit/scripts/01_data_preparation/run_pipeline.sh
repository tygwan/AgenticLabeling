#!/bin/bash

# Project-AGI Pipeline Runner
# This script demonstrates running the pipeline with different categories

# Display help information
display_help() {
    echo "Usage: ./run_pipeline.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  -c, --category NAME   Process a specific category (default: test_category)"
    echo "  -a, --all             Process all categories in the data directory"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline.sh"
    echo "  ./run_pipeline.sh -c my_custom_category"
    echo "  ./run_pipeline.sh --all"
}

# Process a single category
process_category() {
    local category="$1"
    echo "========================================================"
    echo "Processing category: $category"
    echo "========================================================"
    
    # Check if the category directory exists
    if [ ! -d "data/$category" ]; then
        echo "Error: Category directory data/$category does not exist"
        echo "Creating basic directory structure..."
        mkdir -p "data/$category/1.images"
        mkdir -p "data/$category/2.support-set"
        mkdir -p "data/$category/4.dataset"
        echo "Please add images to data/$category/1.images and run again"
        return 1
    fi
    
    # Check if images exist
    if [ -z "$(ls -A data/$category/1.images 2>/dev/null)" ]; then
        echo "Warning: No images found in data/$category/1.images"
        echo "Please add images to data/$category/1.images before processing"
        return 1
    fi
    
    # Run the main launcher with the specified category
    python scripts/main_launcher.py --category "$category"
    
    echo "Completed processing category: $category"
    echo ""
    
    return 0
}

# Process all categories
process_all_categories() {
    echo "Processing all categories..."
    
    # Get all subdirectories in the data directory
    for category in $(ls -d data/*/ 2>/dev/null | cut -d'/' -f2); do
        process_category "$category"
    done
}

# Default values
CATEGORY="test_category"
ALL_CATEGORIES=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -a|--all)
            ALL_CATEGORIES=true
            shift
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Make sure the script is run from the project root directory
if [ ! -d "scripts" ] || [ ! -f "scripts/main_launcher.py" ]; then
    echo "Error: This script must be run from the project root directory"
    echo "Please run: cd /path/to/project-agi && ./run_pipeline.sh"
    exit 1
fi

# Run the appropriate process
if [ "$ALL_CATEGORIES" = true ]; then
    process_all_categories
else
    process_category "$CATEGORY"
fi

echo "Pipeline execution completed!" 