"""Minimal object registry for the MVP app."""

from __future__ import annotations

import base64
import io
import sqlite3
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import get_settings
from .storage import get_conn, remove_mask_file, reset_export_dir, store_mask_bytes


@dataclass
class ExportResult:
    """Metadata for a completed export."""

    dataset_name: str
    dataset_dir: Path
    zip_path: Path
    image_count: int
    object_count: int


class Registry:
    """SQLite-backed registry for sources, categories and objects."""

    def register_source(
        self,
        *,
        project_id: str,
        file_path: str,
        file_name: str,
        width: int,
        height: int,
        source_type: str = "image",
        status: str = "pending",
        error: Optional[str] = None,
    ) -> str:
        source_id = f"src_{uuid.uuid4().hex[:12]}"
        conn = get_conn()
        conn.execute(
            """INSERT INTO sources
               (source_id, project_id, source_type, file_path, file_name, width, height, status, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, project_id, source_type, file_path, file_name, width, height, status, error),
        )
        conn.commit()
        conn.close()
        return source_id

    def set_source_status(
        self,
        source_id: str,
        *,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update a source's status and optional error message."""
        conn = get_conn()
        conn.execute(
            "UPDATE sources SET status = ?, error = ? WHERE source_id = ?",
            (status, error, source_id),
        )
        conn.commit()
        conn.close()

    def list_sources(self, limit: int = 100) -> list[dict]:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM sources ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_source(self, source_id: str) -> Optional[dict]:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_or_create_category(self, name: str, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT category_id FROM categories WHERE name = ?",
            (name,),
        ).fetchone()
        if row:
            return row["category_id"]
        cursor = conn.execute(
            "INSERT INTO categories (name) VALUES (?)",
            (name,),
        )
        return int(cursor.lastrowid)

    def register_objects_batch(self, source_id: str, objects_data: list[dict]) -> list[str]:
        object_ids: list[str] = []
        conn = get_conn()
        try:
            category_cache: dict[str, int] = {}
            for obj in objects_data:
                if obj["category"] not in category_cache:
                    category_cache[obj["category"]] = self.get_or_create_category(obj["category"], conn)

            for obj in objects_data:
                object_id = f"obj_{uuid.uuid4().hex[:12]}"
                object_ids.append(object_id)
                category_id = category_cache[obj["category"]]
                mask_path = None
                if obj.get("mask_base64"):
                    mask_bytes = base64.b64decode(obj["mask_base64"])
                    mask_path = str(store_mask_bytes(object_id, mask_bytes))
                bbox_x, bbox_y, bbox_w, bbox_h = obj["bbox"]
                area = bbox_w * bbox_h
                conn.execute(
                    """INSERT INTO objects (
                           object_id, source_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h,
                           area, confidence, detection_model, mask_path
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        object_id,
                        source_id,
                        category_id,
                        bbox_x,
                        bbox_y,
                        bbox_w,
                        bbox_h,
                        area,
                        obj.get("confidence"),
                        obj.get("detection_model"),
                        mask_path,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return object_ids

    def list_objects(
        self,
        *,
        source_id: Optional[str] = None,
        is_validated: Optional[bool] = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """List objects.

        - `is_validated=True` narrows to `validation_status = 'approved'`.
        - `include_deleted=False` (default) hides soft-deleted rows so callers
          that treat objects as "live data" continue to work unchanged.
        """
        conn = get_conn()
        query = (
            "SELECT o.*, c.name AS category_name FROM objects o "
            "LEFT JOIN categories c ON o.category_id = c.category_id"
        )
        conditions = []
        params: list[object] = []
        if source_id:
            conditions.append("o.source_id = ?")
            params.append(source_id)
        if is_validated is True:
            conditions.append("o.validation_status = 'approved'")
        elif is_validated is False:
            conditions.append("(o.validation_status IS NULL OR o.validation_status = 'pending')")
        if not include_deleted:
            conditions.append("(o.validation_status IS NULL OR o.validation_status != 'deleted')")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY o.created_at DESC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_object(self, object_id: str) -> Optional[dict]:
        conn = get_conn()
        row = conn.execute(
            "SELECT o.*, c.name AS category_name FROM objects o "
            "LEFT JOIN categories c ON o.category_id = c.category_id "
            "WHERE o.object_id = ?",
            (object_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def validate_object(self, object_id: str, reviewer: str = "reviewer", quality_score: float = 0.9) -> bool:
        conn = get_conn()
        cursor = conn.execute(
            "UPDATE objects SET is_validated = 1, validation_status = 'approved', "
            "validated_by = ?, quality_score = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE object_id = ?",
            (reviewer, quality_score, object_id),
        )
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    def delete_object(self, object_id: str) -> bool:
        """Soft-delete: mark the row as 'deleted' but keep it (and its mask
        on disk) so a restore is possible. Use `purge_object` for hard
        removal."""
        current = self.get_object(object_id)
        if not current:
            return False
        conn = get_conn()
        cursor = conn.execute(
            "UPDATE objects SET validation_status = 'deleted', is_validated = 0, "
            "updated_at = CURRENT_TIMESTAMP WHERE object_id = ?",
            (object_id,),
        )
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    def restore_object(self, object_id: str) -> bool:
        """Reverse a soft-delete. Row returns to pending (validation_status NULL)."""
        conn = get_conn()
        cursor = conn.execute(
            "UPDATE objects SET validation_status = NULL, is_validated = 0, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE object_id = ? AND validation_status = 'deleted'",
            (object_id,),
        )
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    def purge_object(self, object_id: str) -> bool:
        """Hard delete: remove the row and its mask file permanently.
        Intended for 'empty trash' workflows, not the normal delete path."""
        current = self.get_object(object_id)
        if not current:
            return False
        conn = get_conn()
        cursor = conn.execute("DELETE FROM objects WHERE object_id = ?", (object_id,))
        conn.commit()
        conn.close()
        remove_mask_file(current.get("mask_path"))
        return cursor.rowcount > 0

    def list_categories(self) -> list[dict]:
        conn = get_conn()
        rows = conn.execute("SELECT * FROM categories ORDER BY name").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        conn = get_conn()
        stats = {
            "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "objects": conn.execute(
                "SELECT COUNT(*) FROM objects "
                "WHERE validation_status IS NULL OR validation_status != 'deleted'"
            ).fetchone()[0],
            "validated_objects": conn.execute(
                "SELECT COUNT(*) FROM objects WHERE validation_status = 'approved'"
            ).fetchone()[0],
            "deleted_objects": conn.execute(
                "SELECT COUNT(*) FROM objects WHERE validation_status = 'deleted'"
            ).fetchone()[0],
            "categories": conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0],
            "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        }
        conn.close()
        return stats

    def start_run(
        self,
        *,
        project_id: str,
        classes: list[str],
        detection_backend: str = "florence2",
        segmentation_backend: str = "sam3",
    ) -> str:
        """Record the start of an auto-label pipeline run. Returns run_id."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        conn = get_conn()
        conn.execute(
            """INSERT INTO runs
               (run_id, project_id, status, classes, detection_backend, segmentation_backend)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (run_id, project_id, ",".join(classes or []), detection_backend, segmentation_backend),
        )
        conn.commit()
        conn.close()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        source_id: Optional[str] = None,
        detections: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a run finished (status='completed' | 'failed') with metadata."""
        conn = get_conn()
        conn.execute(
            """UPDATE runs
               SET status = ?,
                   source_id = COALESCE(?, source_id),
                   detections = COALESCE(?, detections),
                   error = ?,
                   finished_at = CURRENT_TIMESTAMP,
                   duration_ms = CAST((julianday('now') - julianday(started_at)) * 86400000 AS INTEGER)
               WHERE run_id = ?""",
            (status, source_id, detections, error, run_id),
        )
        conn.commit()
        conn.close()

    def list_runs(self, limit: int = 20, project_id: Optional[str] = None) -> list[dict]:
        conn = get_conn()
        if project_id:
            rows = conn.execute(
                "SELECT * FROM runs WHERE project_id = ? ORDER BY started_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def export_dataset(
        self,
        *,
        dataset_name: str,
        export_format: str,
        only_validated: bool = True,
        split_ratios: Optional[dict] = None,
    ) -> ExportResult:
        settings = get_settings()
        objects = self.list_objects(is_validated=True if only_validated else None)
        grouped: dict[str, list[dict]] = {}
        for obj in objects:
            grouped.setdefault(obj["source_id"], []).append(obj)

        categories = self.list_categories()
        cat_to_idx = {cat["name"]: idx for idx, cat in enumerate(categories)}

        dataset_dir = settings.exports_dir / dataset_name
        reset_export_dir(dataset_dir)
        if export_format == "yolo":
            for split in ("train", "val", "test"):
                (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
                (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        else:
            (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
            for split in ("train", "val", "test"):
                (dataset_dir / split).mkdir(parents=True, exist_ok=True)

        source_ids = list(grouped.keys())
        splits = self._split_sources(source_ids, ratios=split_ratios)
        image_count = 0
        object_count = 0

        if export_format == "yolo":
            self._export_yolo(dataset_dir, grouped, splits, cat_to_idx)
            image_count = len(source_ids)
            object_count = len(objects)
        else:
            self._export_coco(dataset_dir, grouped, splits, cat_to_idx)
            image_count = len(source_ids)
            object_count = len(objects)

        zip_path = dataset_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in dataset_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(dataset_dir.parent))

        return ExportResult(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            zip_path=zip_path,
            image_count=image_count,
            object_count=object_count,
        )

    def _split_sources(
        self,
        source_ids: list[str],
        *,
        ratios: Optional[dict] = None,
    ) -> dict[str, list[str]]:
        """Split source_ids into train/val/test buckets.

        `ratios` accepts values in either percent (summing to ~100) or fraction
        (summing to ~1.0). Missing keys default to train=80, val=15, test=5.
        Ordering is deterministic (insertion order of grouped[*]). val+test
        floors are computed first so train absorbs rounding leftover.
        """
        total = len(source_ids)
        if total == 0:
            return {"train": [], "val": [], "test": []}

        raw = ratios or {}
        train_r = float(raw.get("train", 80))
        val_r = float(raw.get("val", 15))
        test_r = float(raw.get("test", 5))
        # Normalize so they sum to 1.0 regardless of percent/fraction input
        tot_r = train_r + val_r + test_r
        if tot_r <= 0:
            train_r, val_r, test_r, tot_r = 80.0, 15.0, 5.0, 100.0
        train_f = train_r / tot_r
        val_f = val_r / tot_r

        val_count = int(total * val_f)
        train_count = max(1, int(total * train_f))
        # Ensure sum never exceeds total.
        if train_count + val_count > total:
            train_count = max(1, total - val_count)
        test_count = total - train_count - val_count
        if test_count < 0:
            test_count = 0
            train_count = total - val_count

        return {
            "train": source_ids[:train_count],
            "val": source_ids[train_count:train_count + val_count],
            "test": source_ids[train_count + val_count:train_count + val_count + test_count],
        }

    def _copy_source_asset(self, source: dict, target: Path) -> None:
        source_path = Path(source["file_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            target.write_bytes(source_path.read_bytes())
        else:
            # Fallback placeholder so export still completes.
            Image.new("RGB", (source.get("width") or 640, source.get("height") or 480), color="white").save(target)

    def _export_yolo(
        self,
        dataset_dir: Path,
        grouped: dict[str, list[dict]],
        splits: dict[str, list[str]],
        cat_to_idx: dict[str, int],
    ) -> None:
        yaml_lines = [
            f"path: {dataset_dir.resolve()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            f"nc: {len(cat_to_idx)}",
            f"names: {list(cat_to_idx.keys())}",
        ]
        (dataset_dir / "data.yaml").write_text("\n".join(yaml_lines))

        for split, source_ids in splits.items():
            for source_id in source_ids:
                source = self.get_source(source_id)
                if not source:
                    continue
                file_name = source["file_name"]
                stem = Path(file_name).stem
                image_target = dataset_dir / split / "images" / file_name
                self._copy_source_asset(source, image_target)

                label_lines = []
                for obj in grouped[source_id]:
                    class_idx = cat_to_idx[obj["category_name"]]
                    cx = (obj["bbox_x"] + obj["bbox_w"] / 2) / source["width"]
                    cy = (obj["bbox_y"] + obj["bbox_h"] / 2) / source["height"]
                    nw = obj["bbox_w"] / source["width"]
                    nh = obj["bbox_h"] / source["height"]
                    label_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                (dataset_dir / split / "labels" / f"{stem}.txt").write_text("\n".join(label_lines))

    def _export_coco(
        self,
        dataset_dir: Path,
        grouped: dict[str, list[dict]],
        splits: dict[str, list[str]],
        cat_to_idx: dict[str, int],
    ) -> None:
        for split, source_ids in splits.items():
            images = []
            annotations = []
            ann_id = 1
            for image_id, source_id in enumerate(source_ids, start=1):
                source = self.get_source(source_id)
                if not source:
                    continue
                file_name = source["file_name"]
                self._copy_source_asset(source, dataset_dir / split / file_name)
                images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": source["width"],
                        "height": source["height"],
                    }
                )
                for obj in grouped[source_id]:
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cat_to_idx[obj["category_name"]] + 1,
                            "bbox": [obj["bbox_x"], obj["bbox_y"], obj["bbox_w"], obj["bbox_h"]],
                            "area": obj["area"],
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
            coco = {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": idx + 1, "name": name, "supercategory": name}
                    for name, idx in cat_to_idx.items()
                ],
            }
            (dataset_dir / "annotations" / f"instances_{split}.json").write_text(
                __import__("json").dumps(coco, indent=2)
            )
