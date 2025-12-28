"""Label management implementation."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class LabelManager:
    """Manages labels and annotations for projects."""

    def __init__(self, data_dir: str = "/data/labels"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_project_dir(self, project_id: str) -> Path:
        """Get project directory path."""
        project_dir = self.data_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def _get_label_path(self, project_id: str, image_id: str) -> Path:
        """Get label file path."""
        return self._get_project_dir(project_id) / f"{image_id}.json"

    def save_label(
        self,
        project_id: str,
        image_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save label data for an image."""
        label_path = self._get_label_path(project_id, image_id)

        label = {
            "image_id": image_id,
            "project_id": project_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            **data,
        }

        # Update timestamp if label exists
        if label_path.exists():
            existing = json.loads(label_path.read_text())
            label["created_at"] = existing.get("created_at", label["created_at"])

        label_path.write_text(json.dumps(label, indent=2))
        return label

    def get_label(self, project_id: str, image_id: str) -> Optional[Dict[str, Any]]:
        """Get label for an image."""
        label_path = self._get_label_path(project_id, image_id)
        if not label_path.exists():
            return None
        return json.loads(label_path.read_text())

    def list_labels(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all labels for a project."""
        project_dir = self._get_project_dir(project_id)
        labels = []

        for label_file in sorted(project_dir.glob("*.json")):
            if label_file.name == "project.json":
                continue
            try:
                label = json.loads(label_file.read_text())
                labels.append(label)
            except Exception:
                continue

        return labels[offset : offset + limit]

    def delete_label(self, project_id: str, image_id: str) -> bool:
        """Delete label for an image."""
        label_path = self._get_label_path(project_id, image_id)
        if label_path.exists():
            label_path.unlink()
            return True
        return False

    def import_labels(
        self,
        project_id: str,
        format: str,
        content: bytes,
    ) -> Dict[str, Any]:
        """Import labels from file."""
        data = json.loads(content.decode("utf-8"))
        count = 0

        if format == "coco":
            # COCO format: annotations list with image_id
            images = {img["id"]: img for img in data.get("images", [])}
            categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

            for ann in data.get("annotations", []):
                image_id = str(ann["image_id"])
                image_info = images.get(ann["image_id"], {})

                label_data = {
                    "boxes": [ann.get("bbox", [])],
                    "classes": [categories.get(ann["category_id"], "unknown")],
                    "image_info": image_info,
                }

                if "segmentation" in ann:
                    label_data["segmentation"] = ann["segmentation"]

                self.save_label(project_id, image_id, label_data)
                count += 1

        elif format == "yolo":
            # YOLO format: one label file per image
            # Expects a JSON with {filename: labels_text} mapping
            for filename, labels_text in data.items():
                image_id = Path(filename).stem
                boxes = []
                classes = []

                for line in labels_text.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5:
                        classes.append(int(parts[0]))
                        boxes.append([float(x) for x in parts[1:5]])

                self.save_label(
                    project_id,
                    image_id,
                    {"boxes": boxes, "classes": classes, "format": "yolo"},
                )
                count += 1

        return {"count": count, "format": format}

    def export_labels(
        self,
        project_id: str,
        format: str,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """Export labels in specified format."""
        labels = self.list_labels(project_id, limit=10000)

        if format == "coco":
            # Convert to COCO format
            coco_data = {
                "images": [],
                "annotations": [],
                "categories": [],
            }

            categories_set = set()
            ann_id = 1

            for idx, label in enumerate(labels):
                image_id = idx + 1
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": f"{label['image_id']}.jpg",
                    "width": label.get("width", 0),
                    "height": label.get("height", 0),
                })

                for i, (box, cls) in enumerate(
                    zip(label.get("boxes", []), label.get("classes", []))
                ):
                    categories_set.add(cls)
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls if isinstance(cls, int) else hash(cls) % 1000,
                        "bbox": box,
                        "area": box[2] * box[3] if len(box) >= 4 else 0,
                        "iscrowd": 0,
                    })
                    ann_id += 1

            for cat in categories_set:
                coco_data["categories"].append({
                    "id": cat if isinstance(cat, int) else hash(cat) % 1000,
                    "name": str(cat),
                })

            export_path = self._get_project_dir(project_id) / "export_coco.json"
            export_path.write_text(json.dumps(coco_data, indent=2))

            return {
                "count": len(labels),
                "format": format,
                "download_url": str(export_path),
            }

        return {"count": len(labels), "format": format}

    def validate_labels(self, project_id: str) -> Dict[str, Any]:
        """Validate all labels in a project."""
        labels = self.list_labels(project_id, limit=10000)
        errors = []
        valid_count = 0

        for label in labels:
            label_errors = []

            if "boxes" not in label and "masks" not in label:
                label_errors.append("No boxes or masks found")

            if "classes" not in label:
                label_errors.append("No classes found")

            if label_errors:
                errors.append({
                    "image_id": label.get("image_id"),
                    "errors": label_errors,
                })
            else:
                valid_count += 1

        return {
            "valid": len(errors) == 0,
            "total": len(labels),
            "valid_count": valid_count,
            "errors": errors,
        }

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        projects = []
        for project_dir in self.data_dir.iterdir():
            if project_dir.is_dir():
                project_file = project_dir / "project.json"
                if project_file.exists():
                    projects.append(json.loads(project_file.read_text()))
                else:
                    # Count labels
                    label_count = len(list(project_dir.glob("*.json")))
                    projects.append({
                        "project_id": project_dir.name,
                        "name": project_dir.name,
                        "label_count": label_count,
                    })
        return projects

    def create_project(
        self,
        project_id: str,
        name: str,
        classes: List[str],
    ) -> Dict[str, Any]:
        """Create a new project."""
        project_dir = self._get_project_dir(project_id)
        project_data = {
            "project_id": project_id,
            "name": name,
            "classes": classes,
            "created_at": datetime.utcnow().isoformat(),
        }

        project_file = project_dir / "project.json"
        project_file.write_text(json.dumps(project_data, indent=2))
        return project_data
