# Advanced Preprocessor Fix for Debug Format Files

## Problem Description

The advanced preprocessor was not properly recognizing mask data in the debug format files. The issue occurred because:

1. The advanced preprocessor was looking for files with the format `{image_name}_coords.txt` while the debug format generates files with the format `{image_name}.txt`.
2. Similarly, the box data was being saved in `{image_name}_box_points.txt` files but the preprocessor was only looking for `{image_name}_box.json` files.

## Changes Made

The following modifications were made to fix the issue:

1. Updated the `load_image_data` method in `scripts/advanced_preprocessor.py` to:
   - Support debug format polygon files (`{image_name}.txt`)
   - Support debug format box files (`{image_name}_box_points.txt`)
   - Added a parser for the box points format to convert it to the same structure as the JSON format
   - Added additional logging to help with debugging

2. Added debug logging configuration in `scripts/main_launcher.py` to make it easier to track issues.

## Files Created for Testing

1. `test_advanced_preprocessor.py`: A test script to verify that the advanced preprocessor can correctly load and process debug format files.
2. `scripts/copy_test_images.py`: A utility script to create test images that match the mask files in the mask directory.

## Steps to Test the Fix

1. First, create test images that match the mask files:
   ```
   python scripts/copy_test_images.py --max_images 5
   ```

2. Run the advanced preprocessor with the debug flag:
   ```
   python scripts/main_launcher.py --category test_category --advanced-preprocess --debug --verbose
   ```

3. Verify that the advanced preprocessor correctly loads and processes the mask data.

## What to Look For

- Check that the advanced preprocessor logs show:
  - Successfully loading debug format files (`{image_name}.txt` and `{image_name}_box_points.txt`)
  - Processing the polygon data to create masks
  - Saving the processed images to the output directory

- Verify that the `data/test_category/6.preprocessed` directory contains processed images from the mask data.

## Additional Notes

- The fix maintains backward compatibility with the original file formats.
- The debug format files must be in the mask directory (`data/test_category/4.mask`).
- If both standard and debug format files exist for the same image, the standard format is prioritized. 