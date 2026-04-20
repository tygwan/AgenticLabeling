#!/bin/bash
# Script to track modified package files that would normally be ignored by gitignore

echo "This script helps track modified package files that would normally be ignored by .gitignore"
echo "==============================="

# Function to add a file with force
add_file_with_force() {
    local file_path=$1
    if [ -f "$file_path" ]; then
        git add -f "$file_path"
        echo "Added: $file_path"
    else
        echo "Error: File not found - $file_path"
    fi
}

# Check if path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path/to/modified/file.py>"
    echo "Example: $0 lib/python3.10/site-packages/autodistill/detection.py"
    exit 1
fi

# Add the specified file
add_file_with_force "$1"

echo "==============================="
echo "File has been force-added to git tracking."
echo "Make sure to commit your changes."
echo "To see all tracked files in the lib directory: git ls-files lib/" 