# Data Directory

This directory contains all data files for the project. The structure is as follows:

```
data/
└── <category_name>/             # Category name (e.g., test_category)
    ├── 1.images/                # Input images
    ├── 2.support-set/           # Example images for each class
    │   ├── class_0/             # Class 0 examples
    │   ├── class_1/             # Class 1 examples
    │   ├── class_2/             # Class 2 examples
    │   ├── class_3/             # Class 3 examples
    │   └── unknown_*/           # Unknown class examples
    ├── 3.box/                   # Bounding box annotations
    ├── 4.mask/                  # Mask and coordinate data
    ├── 5.dataset/               # YOLO dataset format
    ├── 6.preprocessed/          # Preprocessed images
    ├── 7.results/               # Visualization results
    │   └── ground_truth/        # Ground truth data for evaluation
    │       ├── class_0/         # Class 0 ground truth
    │       ├── class_1/         # Class 1 ground truth
    │       ├── class_2/         # Class 2 ground truth
    │       ├── class_3/         # Class 3 ground truth
    │       └── unknown/         # Unknown class ground truth
    ├── 8.refine-dataset/        # Refined dataset
    └── class_mapping.json       # Class mapping configuration
```

## How to Use

1. Create a new category directory under `data/` (e.g., `data/my_category/`)
2. Create the subdirectories listed above
3. Place your input images in the `1.images/` directory
4. Set up your support set in the `2.support-set/` directory
5. Run the processing scripts to generate the rest of the data

## Important Notes

- The actual data files are not included in the repository due to size constraints
- You need to provide your own data files following this structure
- See the project documentation for more details on data preparation
