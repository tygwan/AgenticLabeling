import os
import json
from pathlib import Path
import argparse

def create_ground_truth_manifest(base_path, output_path):
    """
    ground_truth 디렉토리를 스캔하여 각 이미지 파일이 어떤 클래스에 속하는지에 대한
    매니페스트 JSON 파일을 생성합니다.
    """
    print(f"Scanning ground truth data from: {base_path}")
    
    gt_manifest = {}
    class_folders = [f for f in os.listdir(base_path) if f.startswith('Class_') and os.path.isdir(os.path.join(base_path, f))]
    
    if not class_folders:
        print(f"Error: No 'Class_*' directories found in {base_path}")
        return

    print(f"Found class folders: {class_folders}")
    total_files = 0

    for class_folder in class_folders:
        class_name = class_folder
        folder_path = os.path.join(base_path, class_folder)
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                gt_manifest[filename] = class_name
                total_files += 1
                
    # 출력 디렉토리 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt_manifest, f, indent=4)
        
    print(f"\nSuccessfully created manifest file at: {output_path}")
    print(f"Total files processed: {total_files}")
    print(f"Total classes found: {len(class_folders)}")

def main():
    parser = argparse.ArgumentParser(description="Create a ground truth manifest file for the analysis dashboard.")
    parser.add_argument(
        '--input-dir', 
        default='data/test_category/7.results/ground_truth',
        help="Path to the ground_truth directory."
    )
    parser.add_argument(
        '--output-file',
        default='analysis_dashboard/public/data/ground_truth_manifest.json',
        help="Path to save the output manifest.json file."
    )
    args = parser.parse_args()

    # 입력 디렉토리 존재 확인
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    create_ground_truth_manifest(args.input_dir, args.output_file)

if __name__ == '__main__':
    main() 