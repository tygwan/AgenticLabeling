"""COCO format dataset exporter."""
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class COCOExporter:
    """Export objects from Object Registry to COCO format.

    COCO format structure:
        dataset_name/
        ├── annotations/
        │   ├── instances_train.json
        │   ├── instances_val.json
        │   └── instances_test.json
        ├── train/
        │   └── image_001.jpg
        ├── val/
        │   └── ...
        └── test/
            └── ...

    COCO JSON structure:
    {
        "info": {...},
        "licenses": [...],
        "images": [{"id": 1, "file_name": ..., "width": ..., "height": ...}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], ...}],
        "categories": [{"id": 1, "name": ..., "supercategory": ...}]
    }
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
        """Export dataset from Object Registry to COCO format.

        Args:
            dataset_name: Name for the exported dataset
            filter_config: Filter criteria (categories, project_id, min_confidence, is_validated)
            split_config: Split ratios {"train": 0.8, "val": 0.1, "test": 0.1}
            include_masks: Include segmentation masks

        Returns:
            Export result with paths and statistics
        """
        filter_config = filter_config or {}
        split_config = split_config or {"train": 0.8, "val": 0.1, "test": 0.1}

        dataset_dir = self.output_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # Create directory structure
        (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (dataset_dir / split).mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Step 1: Get categories for class mapping
            categories = await self._get_categories(client, filter_config.get("categories"))
            coco_categories = self._build_coco_categories(categories)
            category_name_to_id = {c["name"]: c["id"] for c in coco_categories}

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

            # Step 5: Build COCO annotations for each split
            stats = {"train": 0, "val": 0, "test": 0}
            image_count = 0
            annotation_id = 1

            for split_name, source_ids in splits.items():
                coco_images = []
                coco_annotations = []
                image_id = 1

                for source_id in source_ids:
                    source_objs = source_objects.get(source_id, [])
                    if not source_objs:
                        continue

                    # Get source info
                    source = await self._get_source(client, source_id)
                    if not source:
                        continue

                    img_width = source.get("width")
                    img_height = source.get("height")
                    if not img_width or not img_height:
                        continue

                    # Image filename
                    file_name = source.get("file_name") or f"{source_id}.jpg"

                    # Add COCO image entry
                    coco_images.append({
                        "id": image_id,
                        "file_name": file_name,
                        "width": img_width,
                        "height": img_height,
                        "source_id": source_id,
                    })

                    # Add annotations for this image
                    for obj in source_objs:
                        category_name = obj.get("category_name")
                        if category_name not in category_name_to_id:
                            continue

                        category_id = category_name_to_id[category_name]

                        # COCO bbox format: [x, y, width, height]
                        bbox = [
                            obj["bbox_x"],
                            obj["bbox_y"],
                            obj["bbox_w"],
                            obj["bbox_h"],
                        ]

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": obj.get("area") or (bbox[2] * bbox[3]),
                            "iscrowd": 0,
                            "object_id": obj.get("object_id"),
                        }

                        # Add confidence if available
                        if obj.get("confidence"):
                            annotation["score"] = obj["confidence"]

                        # Add flags
                        if obj.get("is_occluded"):
                            annotation["attributes"] = annotation.get("attributes", {})
                            annotation["attributes"]["occluded"] = True
                        if obj.get("is_truncated"):
                            annotation["attributes"] = annotation.get("attributes", {})
                            annotation["attributes"]["truncated"] = True

                        # TODO: Add segmentation if include_masks and mask available

                        coco_annotations.append(annotation)
                        annotation_id += 1
                        stats[split_name] += 1

                    # Copy/download image
                    image_path = dataset_dir / split_name / file_name
                    await self._save_image_placeholder(image_path, source)
                    image_id += 1
                    image_count += 1

                # Write COCO JSON for this split
                coco_data = self._build_coco_json(
                    dataset_name, split_name, coco_images, coco_annotations, coco_categories
                )
                json_path = dataset_dir / "annotations" / f"instances_{split_name}.json"
                json_path.write_text(json.dumps(coco_data, indent=2))

            # Step 6: Create zip file
            zip_path = self._create_zip(dataset_dir)

            return {
                "success": True,
                "dataset_name": dataset_name,
                "path": str(dataset_dir),
                "zip_path": str(zip_path),
                "categories": [c["name"] for c in coco_categories],
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

    def _build_coco_categories(self, categories: List[Dict]) -> List[Dict]:
        """Build COCO categories list."""
        coco_categories = []
        for idx, cat in enumerate(categories, start=1):
            coco_categories.append({
                "id": idx,
                "name": cat["name"],
                "supercategory": cat.get("supercategory") or cat["name"],
            })
        return coco_categories

    def _build_coco_json(
        self,
        dataset_name: str,
        split_name: str,
        images: List[Dict],
        annotations: List[Dict],
        categories: List[Dict],
    ) -> Dict:
        """Build complete COCO JSON structure."""
        return {
            "info": {
                "description": f"{dataset_name} - {split_name} split",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "AgenticLabeling",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": "",
                }
            ],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    async def _save_image_placeholder(self, image_path: Path, source: Dict):
        """Save image or create placeholder."""
        file_path = source.get("file_path")

        if file_path:
            src_path = Path(file_path)
            if src_path.exists():
                shutil.copy(src_path, image_path)
                return

        # Create placeholder
        placeholder = image_path.with_suffix(".txt")
        placeholder.write_text(f"Image placeholder for source: {source.get('source_id')}\n"
                               f"Original path: {file_path}")

    def _create_zip(self, dataset_dir: Path) -> Path:
        """Create zip archive of dataset."""
        zip_path = dataset_dir.with_suffix(".zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_dir)
                    zf.write(file_path, arcname)

        return zip_path


async def export_coco_dataset(
    registry_url: str,
    output_dir: str,
    dataset_name: str,
    filter_config: Optional[Dict[str, Any]] = None,
    split_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Convenience function to export COCO dataset."""
    exporter = COCOExporter(registry_url, output_dir)
    return await exporter.export_dataset(dataset_name, filter_config, split_config)
