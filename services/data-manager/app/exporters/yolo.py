"""YOLO format dataset exporter."""
import base64
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml


class YOLOExporter:
    """Export objects from Object Registry to YOLO format.

    YOLO format structure:
        dataset_name/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   │   └── image_001.jpg
        │   └── labels/
        │       └── image_001.txt
        ├── val/
        │   └── ...
        └── test/
            └── ...

    Label format: class_id center_x center_y width height (normalized 0-1)
    """

    def __init__(self, registry_url: str, output_dir: str):
        self.registry_url = registry_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def export_dataset(
        self,
        dataset_name: str,
        filter_config: Optional[Dict[str, Any]] = None,
        split_config: Optional[Dict[str, float]] = None,
        include_masks: bool = False,
    ) -> Dict[str, Any]:
        """Export dataset from Object Registry to YOLO format.

        Args:
            dataset_name: Name for the exported dataset
            filter_config: Filter criteria (categories, project_id, min_confidence, is_validated)
            split_config: Split ratios {"train": 0.8, "val": 0.1, "test": 0.1}
            include_masks: Include segmentation masks (for YOLO-seg)

        Returns:
            Export result with paths and statistics
        """
        filter_config = filter_config or {}
        split_config = split_config or {"train": 0.8, "val": 0.1, "test": 0.1}

        dataset_dir = self.output_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # Create directory structure
        for split in ["train", "val", "test"]:
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Step 1: Get categories for class mapping
            categories = await self._get_categories(client, filter_config.get("categories"))
            class_names = [c["name"] for c in categories]
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            # Step 2: Get all objects matching filter
            objects = await self._get_filtered_objects(client, filter_config)

            if not objects:
                return {
                    "success": True,
                    "dataset_name": dataset_name,
                    "path": str(dataset_dir),
                    "object_count": 0,
                    "image_count": 0,
                    "splits": {},
                    "message": "No objects found matching filter criteria",
                }

            # Step 3: Group objects by source (image)
            source_objects = self._group_by_source(objects)

            # Step 4: Split sources into train/val/test
            splits = self._split_sources(list(source_objects.keys()), split_config)

            # Step 5: Export each split
            stats = {"train": 0, "val": 0, "test": 0}
            image_count = 0

            for split_name, source_ids in splits.items():
                for source_id in source_ids:
                    source_objs = source_objects[source_id]
                    if not source_objs:
                        continue

                    # Get source info
                    source = await self._get_source(client, source_id)
                    if not source:
                        continue

                    # Get image dimensions
                    img_width = source.get("width")
                    img_height = source.get("height")

                    if not img_width or not img_height:
                        continue

                    # Generate image filename
                    file_name = source.get("file_name") or f"{source_id}.jpg"
                    base_name = Path(file_name).stem

                    # Write YOLO label file
                    label_lines = []
                    for obj in source_objs:
                        category_name = obj.get("category_name")
                        if category_name not in class_to_idx:
                            continue

                        class_idx = class_to_idx[category_name]

                        # Convert bbox to YOLO format (normalized center x, y, w, h)
                        bbox = self._convert_bbox_to_yolo(
                            obj["bbox_x"], obj["bbox_y"],
                            obj["bbox_w"], obj["bbox_h"],
                            img_width, img_height
                        )

                        if include_masks and obj.get("mask_path"):
                            # YOLO-seg format: class_id x1 y1 x2 y2 ... (polygon points)
                            # For simplicity, use bbox for now
                            pass

                        label_lines.append(
                            f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                        )
                        stats[split_name] += 1

                    if label_lines:
                        # Write label file
                        label_path = dataset_dir / split_name / "labels" / f"{base_name}.txt"
                        label_path.write_text("\n".join(label_lines))

                        # Copy/download image (placeholder - source에서 file_path 사용)
                        image_path = dataset_dir / split_name / "images" / file_name
                        await self._save_image_placeholder(image_path, source)
                        image_count += 1

            # Step 6: Create data.yaml
            self._create_data_yaml(dataset_dir, class_names)

            # Step 7: Create zip file
            zip_path = self._create_zip(dataset_dir)

            return {
                "success": True,
                "dataset_name": dataset_name,
                "path": str(dataset_dir),
                "zip_path": str(zip_path),
                "class_names": class_names,
                "object_count": sum(stats.values()),
                "image_count": image_count,
                "splits": stats,
            }

    async def _get_categories(
        self,
        client: httpx.AsyncClient,
        filter_categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get categories from registry."""
        resp = await client.get(f"{self.registry_url}/categories")
        data = resp.json()

        if not data.get("success"):
            return []

        categories = data.get("data", [])

        if filter_categories:
            categories = [c for c in categories if c["name"] in filter_categories]

        return categories

    async def _get_filtered_objects(
        self,
        client: httpx.AsyncClient,
        filter_config: Dict[str, Any]
    ) -> List[Dict]:
        """Get objects matching filter criteria."""
        params = {"limit": 1000}  # Max limit allowed by registry API

        if filter_config.get("project_id"):
            params["project_id"] = filter_config["project_id"]
        if filter_config.get("category"):
            params["category"] = filter_config["category"]
        if filter_config.get("min_confidence"):
            params["min_confidence"] = filter_config["min_confidence"]
        if filter_config.get("is_validated") is not None:
            params["is_validated"] = filter_config["is_validated"]

        resp = await client.get(f"{self.registry_url}/objects", params=params)
        data = resp.json()

        if not data.get("success"):
            return []

        objects = data.get("data", [])

        # Additional category filtering
        if filter_config.get("categories"):
            objects = [
                obj for obj in objects
                if obj.get("category_name") in filter_config["categories"]
            ]

        return objects

    async def _get_source(self, client: httpx.AsyncClient, source_id: str) -> Optional[Dict]:
        """Get source info from registry."""
        resp = await client.get(f"{self.registry_url}/sources/{source_id}")
        data = resp.json()
        return data.get("data") if data.get("success") else None

    def _group_by_source(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        """Group objects by source_id."""
        grouped = {}
        for obj in objects:
            source_id = obj.get("source_id")
            if source_id:
                if source_id not in grouped:
                    grouped[source_id] = []
                grouped[source_id].append(obj)
        return grouped

    def _split_sources(
        self,
        source_ids: List[str],
        split_config: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Split source IDs into train/val/test."""
        import random
        random.shuffle(source_ids)

        total = len(source_ids)
        train_end = int(total * split_config.get("train", 0.8))
        val_end = train_end + int(total * split_config.get("val", 0.1))

        return {
            "train": source_ids[:train_end],
            "val": source_ids[train_end:val_end],
            "test": source_ids[val_end:],
        }

    def _convert_bbox_to_yolo(
        self,
        x: float, y: float, w: float, h: float,
        img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        """Convert bbox from (x, y, w, h) to YOLO format (cx, cy, w, h) normalized."""
        # Center coordinates
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        # Normalized width and height
        nw = w / img_width
        nh = h / img_height

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        return (cx, cy, nw, nh)

    async def _save_image_placeholder(self, image_path: Path, source: Dict):
        """Save image or create placeholder.

        In production, this would:
        1. Copy from local file_path if available
        2. Or download from a storage service

        For now, creates a placeholder text file.
        """
        file_path = source.get("file_path")

        if file_path:
            src_path = Path(file_path)
            if src_path.exists():
                shutil.copy(src_path, image_path)
                return

        # Create placeholder (in production, handle missing images differently)
        placeholder = image_path.with_suffix(".txt")
        placeholder.write_text(f"Image placeholder for source: {source.get('source_id')}\n"
                               f"Original path: {file_path}")

    def _create_data_yaml(self, dataset_dir: Path, class_names: List[str]):
        """Create YOLO data.yaml file."""
        data_config = {
            "path": str(dataset_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }

        yaml_path = dataset_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data_config, default_flow_style=False, allow_unicode=True))

    def _create_zip(self, dataset_dir: Path) -> Path:
        """Create zip archive of dataset."""
        zip_path = dataset_dir.with_suffix(".zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_dir)
                    zf.write(file_path, arcname)

        return zip_path


async def export_yolo_dataset(
    registry_url: str,
    output_dir: str,
    dataset_name: str,
    filter_config: Optional[Dict[str, Any]] = None,
    split_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Convenience function to export YOLO dataset."""
    exporter = YOLOExporter(registry_url, output_dir)
    return await exporter.export_dataset(dataset_name, filter_config, split_config)
