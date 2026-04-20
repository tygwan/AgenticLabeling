# Project Cleanup Instructions

Follow these instructions to clean up and organize the project for better structure and deployment readiness.

## Step 1: Create Necessary Directories

```bash
# Create few_shot directory for few-shot learning related files
mkdir -p few_shot/scripts

# Create tests directory for test files
mkdir -p tests
```

## Step 2: Move Few-Shot Learning Files

```bash
# Move main files
cp run_few_shot_platform.py few_shot/
cp few_shot_requirements.txt few_shot/requirements.txt
cp FEW_SHOT_README.md few_shot/README.md
cp install_few_shot_deps.sh few_shot/install_deps.sh

# Move script files
cp scripts/classifier_cosine.py few_shot/scripts/
cp scripts/classifier_cosine_experiment.py few_shot/scripts/
cp scripts/few_shot_webapp.py few_shot/scripts/
cp scripts/debug_model.py few_shot/scripts/

# Create __init__.py files for proper imports
echo "# Few-Shot Learning Package" > few_shot/__init__.py
echo "# Few-Shot Learning Scripts" > few_shot/scripts/__init__.py

# Make scripts executable
chmod +x few_shot/run_few_shot_platform.py
chmod +x few_shot/install_deps.sh
```

## Step 3: Move Test Files

```bash
# Move test files
cp test_model_imports.py tests/
cp simple_few_shot_test.py tests/
cp run_few_shot_platform_debug.py tests/
cp test_advanced_preprocessor.py tests/
cp test_mask_extraction.py tests/
cp test_mask_filtering.py tests/
cp test_debug_format.py tests/

# Create README for tests directory
cat > tests/README.md << EOF
# Test Files

This directory contains various test files for debugging and testing different components of the project.

## Files:
- test_model_imports.py: Tests importing various models used in the few-shot learning platform
- simple_few_shot_test.py: Simplified test script for diagnosing few-shot learning issues
- run_few_shot_platform_debug.py: Debug version of the few-shot platform with extended error reporting
- test_advanced_preprocessor.py: Tests for the advanced preprocessor
- test_mask_extraction.py: Tests for mask extraction functionality
- test_mask_filtering.py: Tests for mask filtering functionality
- test_debug_format.py: Tests for debug format functionality
EOF
```

## Step 4: Update .gitignore

Add the following to your .gitignore file:

```
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual Environment
venv/
ENV/
env/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log
logs/
mcp.log
eventerrorlog.txt

# Database
*.db
mcp.db

# OS specific
.DS_Store
Thumbs.db

# Library directories
lib/
lib64/
```

## Step 5: Clean Up Unnecessary Files

After you've moved all the files you need, you can safely remove the original copies and other temporary or unnecessary files:

```bash
# Remove unnecessary files
rm -f eventerrorlog.txt
rm -f cleanup_project.sh
rm -f organize_project.py
```

## Step 6: Git Setup (If Not Already Done)

If you're setting up Git for the first time, run:

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Organized project structure"
```

## Final Project Structure

After completing these steps, your project structure should look like this:

```
project-agi/
├── few_shot/                # Few-Shot Learning Platform
│   ├── scripts/
│   │   ├── classifier_cosine.py
│   │   ├── classifier_cosine_experiment.py
│   │   ├── few_shot_webapp.py
│   │   ├── debug_model.py
│   │   └── __init__.py
│   ├── run_few_shot_platform.py
│   ├── requirements.txt
│   ├── install_deps.sh
│   ├── README.md
│   └── __init__.py
├── tests/                   # Test files
│   ├── test_model_imports.py
│   ├── simple_few_shot_test.py
│   ├── run_few_shot_platform_debug.py
│   ├── test_advanced_preprocessor.py
│   ├── test_mask_extraction.py
│   ├── test_mask_filtering.py
│   ├── test_debug_format.py
│   └── README.md
├── scripts/                 # Other scripts
├── data/                    # Data directory
│   └── test_category/       # Example category
└── README.md                # Main README
```

## Running the Few-Shot Learning Platform

After reorganizing, you can run the platform with:

```bash
cd few_shot
python run_few_shot_platform.py --webapp  # For web interface
# OR
python run_few_shot_platform.py --cli --category test_category --model resnet  # For CLI
``` 