# Project-Local Model Storage

This document explains how model files are stored locally within the project directory structure rather than in user-specific cache directories. This approach makes the project more portable and ensures consistent behavior across different environments and users.

## Default Behavior (Before)

By default, the SAM2 and other Autodistill models store their files in user-specific cache directories:

- `~/.cache/autodistill/` - Default cache location for models
- `~/.yoon/advanced-autodistill/` - Additional cache location used by some models

This approach had several drawbacks:
1. Different users/machines would download their own copies of models
2. Docker containers needed volume mounts to persistent storage
3. Models weren't versioned with the project code
4. Complex deployment required additional setup

## Improved Approach (Current)

Our implementation overrides the default behavior to store model files within the project structure:

```
project-agi/
└── models/
    └── sam2/                    # SAM2 model directory
        ├── segment-anything-2/  # Cloned SAM2 repository
        └── sam2_hiera_base_plus.pth  # Model weights
```

## Implementation Details

### 1. Custom Model Loader

The `custom_helpers.py` module defines a `load_SAM_local()` function that loads the SAM2 model from the project directory:

```python
def load_SAM_local():
    """
    Load SAM2 model from the project's models directory instead of ~/.cache.
    
    Returns:
        SAM2ImagePredictor: The initialized predictor
    """
    # Project root directory setup
    cur_dir = os.getcwd()
    project_root = Path(cur_dir)
    
    # Project model directory specification
    SAM_DIR = project_root / "models" / "sam2"
    SAM_CODE_DIR = SAM_DIR / "segment-anything-2"
    SAM_CHECKPOINT_PATH = SAM_DIR / "sam2_hiera_base_plus.pth"
    
    # ...implementation details...
    
    return predictor
```

Key improvements:
- Uses project-relative paths
- Creates directories if they don't exist
- Downloads the model if not present
- Clones the repository if needed

### 2. Monkey Patching

To avoid modifying the Autodistill library directly, we use monkey patching to replace the original function:

```python
def patch_grounded_sam2():
    """
    Patch the autodistill_grounded_sam_2 package's GroundedSAM2 class
    to load SAM models from the project internal directory.
    
    Must be called before importing GroundedSAM2.
    """
    try:
        import autodistill_grounded_sam_2.helpers
        
        # Backup original function
        original_load_SAM = autodistill_grounded_sam_2.helpers.load_SAM
        
        # Replace function
        autodistill_grounded_sam_2.helpers.load_SAM = load_SAM_local
        
        print("SAM2 model loader successfully patched!")
        return True
    except Exception as e:
        print(f"SAM2 model loader patch failed: {e}")
        return False
```

This approach has several advantages:
1. No modification of installed packages is required
2. The patching is done at runtime
3. If the patch fails, the system can fall back to the original behavior

### 3. Usage in Main Launcher

The patch is applied at the start of the `main_launcher.py` script:

```python
# Import and apply custom helpers patch (before GroundedSAM2 import)
from scripts.custom_helpers import patch_grounded_sam2
patch_successful = patch_grounded_sam2()
if not patch_successful:
    print("WARNING: Failed to patch SAM model loading to use project directory")
    print("Default ~/.cache path may be used.")
```

## Docker Integration

The Docker configuration mounts the entire project directory, making the models directory accessible to the container:

```yaml
volumes:
  - .:/app  # Mount entire project
  - ./data:/app/data  # Explicitly mount data directory
```

This ensures that:
1. Models downloaded inside the container are preserved between runs
2. Multiple containers can share the same model files
3. No volume mounting of user home directories is required

## Benefits of This Approach

1. **Portability**: The project can be moved between machines without losing model files
2. **Consistency**: All users/environments use the same model files
3. **Versioning**: Model files can be versioned alongside code if needed
4. **Simplified Deployment**: No need to pre-download models or configure custom paths
5. **Reduced Disk Usage**: Only one copy of each model is needed, regardless of users

## Customizing the Implementation

To modify the model storage locations:

1. Update the `SAM_DIR` path in `load_SAM_local()` function
2. Ensure the directory is included in Docker volumes if using containers
3. Update this documentation to reflect the changes 