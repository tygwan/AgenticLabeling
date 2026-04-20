#!/usr/bin/env python3
import os
import shutil

# 백업 실행
mask_dir = "data/test_category/4.mask"
backup_dir = "data/test_category/4.mask_backup"

print("=== 마스크 파일 백업 실행 ===")

if os.path.exists(mask_dir):
    print(f"백업 디렉토리 생성: {backup_dir}")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # .txt 파일들만 백업
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.txt')]
    print(f"백업할 파일 수: {len(mask_files)}")
    
    for i, mask_file in enumerate(mask_files):
        src = os.path.join(mask_dir, mask_file)
        dst = os.path.join(backup_dir, mask_file)
        shutil.move(src, dst)
        if i < 5:  # 처음 5개만 출력
            print(f"  {i+1}. {mask_file} -> 백업 완료")
    
    if len(mask_files) > 5:
        print(f"  ... 외 {len(mask_files)-5}개 파일 백업 완료")
    
    print(f"총 {len(mask_files)}개 파일 백업 완료")
else:
    print(f"마스크 디렉토리가 없습니다: {mask_dir}")

# 현재 이미지 확인
images_dir = "data/test_category/1.images"
if os.path.exists(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n현재 이미지 파일 수: {len(image_files)}")
    print("처음 5개 이미지:")
    for i, img in enumerate(image_files[:5]):
        print(f"  {i+1}. {img}")

print("\n=== 백업 완료 ===")
print("이제 새로운 이미지들로 처리할 준비가 되었습니다.") 