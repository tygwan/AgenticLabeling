"""Dataset management implementation."""
import io
import json
import os
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetManager:
    """Manages datasets with COCO/YOLO format support."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.datasets_dir = self.data_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_dir(self, name: str) -> Path:
        """Get dataset directory."""
        return self.datasets_dir / name

    def create_dataset(
        self,
        name: str,
        description: str,
        classes: List[str],
    ) -> Dict[str, Any]:
        """Create a new dataset."""
        dataset_dir = self._get_dataset_dir(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure
        for split in ["train", "val", "test"]:
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "name": name,
            "description": description,
            "classes": classes,
            "created_at": datetime.utcnow().isoformat(),
            "image_count": 0,
        }

        metadata_path = dataset_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Create YOLO data.yaml
        self._create_yolo_yaml(dataset_dir, classes)

        return metadata

    def _create_yolo_yaml(self, dataset_dir: Path, classes: List[str]):
        """Create YOLO format data.yaml file."""
        yaml_content = f"""path: {dataset_dir}
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
        (dataset_dir / "data.yaml").write_text(yaml_content)

    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset information."""
        dataset_dir = self._get_dataset_dir(name)
        metadata_path = dataset_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text())

        # Count images per split
        for split in ["train", "val", "test"]:
            img_dir = dataset_dir / split / "images"
            if img_dir.exists():
                metadata[f"{split}_count"] = len(list(img_dir.glob("*")))

        return metadata

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets."""
        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                info = self.get_dataset_info(dataset_dir.name)
                if info:
                    datasets.append(info)
        return datasets

    def add_image(
        self,
        dataset_name: str,
        split: str,
        filename: str,
        content: bytes,
        class_name: Optional[str] = None,
    ) -> str:
        """Add image to dataset."""
        dataset_dir = self._get_dataset_dir(dataset_name)
        img_dir = dataset_dir / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        img_path = img_dir / filename
        img_path.write_bytes(content)

        return str(img_path)

    def list_images(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        class_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List images in dataset."""
        dataset_dir = self._get_dataset_dir(dataset_name)
        images = []

        splits = [split] if split else ["train", "val", "test"]

        for s in splits:
            img_dir = dataset_dir / s / "images"
            if not img_dir.exists():
                continue

            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    label_path = dataset_dir / s / "labels" / f"{img_path.stem}.txt"
                    images.append({
                        "path": str(img_path),
                        "filename": img_path.name,
                        "split": s,
                        "has_label": label_path.exists(),
                    })

        return images[offset : offset + limit]

    def split_dataset(
        self,
        dataset_name: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratified: bool = True,
    ) -> Dict[str, int]:
        """Split dataset into train/val/test sets."""
        dataset_dir = self._get_dataset_dir(dataset_name)

        # Collect all images
        all_images = []
        for split_dir in ["train", "val", "test"]:
            img_dir = dataset_dir / split_dir / "images"
            if img_dir.exists():
                all_images.extend(list(img_dir.glob("*")))

        random.shuffle(all_images)

        # Calculate split sizes
        total = len(all_images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        # Split images
        train_images = all_images[:train_count]
        val_images = all_images[train_count : train_count + val_count]
        test_images = all_images[train_count + val_count :]

        # Move images to appropriate splits
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split_name, images in splits.items():
            target_dir = dataset_dir / split_name / "images"
            target_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                if img.parent.name != split_name:
                    shutil.move(str(img), str(target_dir / img.name))

                    # Move label if exists
                    label_path = img.parent.parent / "labels" / f"{img.stem}.txt"
                    if label_path.exists():
                        target_label_dir = dataset_dir / split_name / "labels"
                        target_label_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(
                            str(label_path), str(target_label_dir / f"{img.stem}.txt")
                        )

        return {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

    def export_dataset(
        self,
        dataset_name: str,
        format: str,
        splits: List[str],
    ) -> str:
        """Export dataset as zip file."""
        dataset_dir = self._get_dataset_dir(dataset_name)
        export_path = self.data_dir / "exports" / f"{dataset_name}_{format}.zip"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(export_path, "w") as zf:
            # Add data.yaml
            yaml_path = dataset_dir / "data.yaml"
            if yaml_path.exists():
                zf.write(yaml_path, "data.yaml")

            # Add images and labels
            for split in splits:
                for subdir in ["images", "labels"]:
                    dir_path = dataset_dir / split / subdir
                    if dir_path.exists():
                        for file_path in dir_path.glob("*"):
                            zf.write(file_path, f"{split}/{subdir}/{file_path.name}")

        return str(export_path)

    def import_dataset(
        self,
        dataset_name: str,
        format: str,
        content: bytes,
    ) -> Dict[str, Any]:
        """Import dataset from zip file."""
        dataset_dir = self._get_dataset_dir(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        images_count = 0
        annotations_count = 0

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for file_info in zf.filelist:
                if file_info.is_dir():
                    continue

                # Extract to appropriate location
                target_path = dataset_dir / file_info.filename
                target_path.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(file_info) as src:
                    target_path.write_bytes(src.read())

                if "images" in file_info.filename:
                    images_count += 1
                elif "labels" in file_info.filename:
                    annotations_count += 1

        return {"images": images_count, "annotations": annotations_count}

    def delete_dataset(self, dataset_name: str):
        """Delete a dataset."""
        dataset_dir = self._get_dataset_dir(dataset_name)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

    def augment_dataset(
        self,
        dataset_name: str,
        augmentations: Dict[str, Any],
        target_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply data augmentation (placeholder)."""
        # TODO: Implement augmentation with albumentations
        return {"generated": 0, "message": "Augmentation not yet implemented"}
