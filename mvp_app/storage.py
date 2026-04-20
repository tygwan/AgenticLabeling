"""SQLite storage and filesystem helpers for the MVP app."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from .config import get_settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    project_id TEXT,
    source_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_w REAL NOT NULL,
    bbox_h REAL NOT NULL,
    area REAL,
    confidence REAL,
    detection_model TEXT,
    mask_path TEXT,
    is_validated INTEGER NOT NULL DEFAULT 0,
    validated_by TEXT,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE INDEX IF NOT EXISTS idx_objects_source_id ON objects(source_id);
CREATE INDEX IF NOT EXISTS idx_objects_category_id ON objects(category_id);
CREATE INDEX IF NOT EXISTS idx_objects_is_validated ON objects(is_validated);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT,
    source_id TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    source_count INTEGER DEFAULT 1,
    classes TEXT,
    detection_backend TEXT,
    segmentation_backend TEXT,
    detections INTEGER,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    duration_ms INTEGER,
    error TEXT,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
"""


def init_storage() -> None:
    """Initialize runtime directories and SQLite schema."""
    settings = get_settings()
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.sqlite_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


def get_conn() -> sqlite3.Connection:
    """Open a SQLite connection with pragmatic concurrency settings."""
    settings = get_settings()
    conn = sqlite3.connect(settings.sqlite_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def store_image_bytes(filename: str, image_bytes: bytes) -> Path:
    """Store an uploaded asset and return its absolute path."""
    settings = get_settings()
    suffix = Path(filename).suffix or ".jpg"
    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    target = settings.assets_dir / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(image_bytes)
    return target


def store_mask_bytes(object_id: str, mask_bytes: bytes) -> Path:
    """Store a mask under a sharded filesystem path."""
    settings = get_settings()
    settings.masks_dir.mkdir(parents=True, exist_ok=True)
    shard_a = object_id[4:6] if len(object_id) >= 6 else "00"
    shard_b = object_id[6:8] if len(object_id) >= 8 else "00"
    target_dir = settings.masks_dir / shard_a / shard_b
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{object_id}.png"
    target.write_bytes(mask_bytes)
    return target


def remove_mask_file(mask_path: Optional[str]) -> None:
    """Best-effort removal for a stored mask."""
    if not mask_path:
        return
    path = Path(mask_path)
    if path.exists():
        path.unlink()


def reset_export_dir(dataset_dir: Path) -> None:
    """Recreate a dataset export directory."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
