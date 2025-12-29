"""Object Registry - Core class for managing object data."""
import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LRUCache:
    """Simple LRU cache with TTL support for embedding search results."""

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}

    def _make_key(self, embedding: List[float], top_k: int, category: str, min_conf: float) -> str:
        """Create cache key from search parameters."""
        # Use first 16 values and last 16 values for faster hashing
        emb_sample = embedding[:16] + embedding[-16:] if len(embedding) > 32 else embedding
        key_data = f"{emb_sample}:{top_k}:{category}:{min_conf}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, embedding: List[float], top_k: int, category: str, min_conf: float):
        """Get cached result if exists and not expired."""
        key = self._make_key(embedding, top_k, category or "", min_conf or 0)
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
        return None

    def set(self, embedding: List[float], top_k: int, category: str, min_conf: float, value):
        """Cache a search result."""
        key = self._make_key(embedding, top_k, category or "", min_conf or 0)
        if len(self._cache) >= self.maxsize:
            # Remove oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._timestamps[oldest]
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._timestamps.clear()


class SearchMetrics:
    """Track embedding search metrics for performance monitoring."""

    def __init__(self):
        self.total_searches = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_search_time_ms = 0.0
        self.avg_results_per_search = 0.0
        self._results_sum = 0

    def record_search(self, duration_ms: float, num_results: int, cache_hit: bool):
        """Record a search event."""
        self.total_searches += 1
        self.total_search_time_ms += duration_ms
        self._results_sum += num_results
        self.avg_results_per_search = self._results_sum / self.total_searches

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            "total_searches": self.total_searches,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, self.total_searches),
            "avg_search_time_ms": self.total_search_time_ms / max(1, self.total_searches),
            "avg_results_per_search": self.avg_results_per_search,
        }

    def reset(self):
        """Reset all metrics."""
        self.__init__()


class ObjectRegistry:
    """Central registry for managing detected objects with SQLite + ChromaDB hybrid storage."""

    def __init__(self, data_dir: str, cache_size: int = 100, cache_ttl: int = 300):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage paths
        self.db_path = self.data_dir / "registry.db"
        self.chroma_path = self.data_dir / "chroma"
        self.masks_path = self.data_dir / "masks"
        self.masks_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_db()

        # ChromaDB client (lazy loaded)
        self._chroma_client = None
        self._collections = {}

        # Search optimization
        self._search_cache = LRUCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        self._search_metrics = SearchMetrics()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r") as f:
            schema = f.read()

        conn = sqlite3.connect(self.db_path)
        conn.executescript(schema)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get SQLite connection with row factory and WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def chroma(self):
        """Lazy load ChromaDB client."""
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        return self._chroma_client

    def _get_collection(self, version_id: str = "dinov2_base_v1"):
        """Get or create ChromaDB collection for embedding version."""
        if version_id not in self._collections:
            collection_name = f"objects_{version_id}"
            self._collections[version_id] = self.chroma.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[version_id]

    # ==================== Source Management ====================

    def register_source(
        self,
        source_type: str,
        file_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        frame_count: Optional[int] = None,
        fps: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Register an image/video source."""
        source_id = f"src_{uuid.uuid4().hex[:12]}"
        file_name = Path(file_path).name if file_path else None

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO sources (source_id, source_type, file_path, file_name,
               width, height, frame_count, fps, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, source_type, file_path, file_name, width, height,
             frame_count, fps, json.dumps(metadata) if metadata else None)
        )
        conn.commit()
        conn.close()
        return source_id

    def get_source(self, source_id: str) -> Optional[dict]:
        """Get source by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    # ==================== Category Management ====================

    def get_or_create_category(
        self,
        name: str,
        supercategory: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Get existing category or create new one."""
        should_close = conn is None
        if conn is None:
            conn = self._get_conn()

        # Try to get existing
        row = conn.execute(
            "SELECT category_id FROM categories WHERE name = ?", (name,)
        ).fetchone()

        if row:
            if should_close:
                conn.close()
            return row["category_id"]

        # Create new
        cursor = conn.execute(
            "INSERT INTO categories (name, supercategory) VALUES (?, ?)",
            (name, supercategory)
        )
        category_id = cursor.lastrowid
        if should_close:
            conn.commit()
            conn.close()
        return category_id

    def list_categories(self) -> List[dict]:
        """List all categories."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM categories ORDER BY name").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ==================== Object Management ====================

    def register_object(
        self,
        source_id: str,
        category_name: str,
        bbox: List[float],  # [x, y, w, h]
        confidence: Optional[float] = None,
        detection_model: Optional[str] = None,
        mask_data: Optional[bytes] = None,
        embedding: Optional[List[float]] = None,
        project_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Register a new detected object."""
        object_id = f"obj_{uuid.uuid4().hex[:12]}"

        # Save mask if provided (outside transaction - filesystem)
        mask_path = None
        if mask_data:
            mask_path = self._save_mask(object_id, mask_data)

        # Calculate area
        area = bbox[2] * bbox[3] if len(bbox) >= 4 else None

        conn = self._get_conn()
        try:
            # Get or create category within same connection
            category_id = self.get_or_create_category(category_name, conn=conn)

            conn.execute(
                """INSERT INTO objects (object_id, category_id, source_id, frame_id, project_id,
                   bbox_x, bbox_y, bbox_w, bbox_h, mask_path, area, confidence, detection_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (object_id, category_id, source_id, frame_id, project_id,
                 bbox[0], bbox[1], bbox[2], bbox[3], mask_path, area, confidence, detection_model)
            )

            # Queue embedding for sync if provided
            if embedding:
                self._queue_embedding(conn, object_id, embedding)

            conn.commit()
        finally:
            conn.close()

        # Sync embedding immediately (can be made async later)
        if embedding:
            self._sync_embedding(object_id, embedding, category_name, confidence)

        return object_id

    def register_objects_batch(
        self,
        source_id: str,
        objects_data: List[dict],
        project_id: Optional[str] = None,
    ) -> List[str]:
        """Register multiple objects in batch."""
        object_ids = []

        # Pre-save masks (filesystem operations - outside transaction)
        mask_paths = {}
        for i, obj in enumerate(objects_data):
            object_id = f"obj_{uuid.uuid4().hex[:12]}"
            object_ids.append(object_id)
            if obj.get("mask_data"):
                mask_paths[i] = self._save_mask(object_id, obj["mask_data"])

        conn = self._get_conn()
        try:
            # Pre-cache category IDs to minimize queries
            category_cache = {}
            for obj in objects_data:
                cat_name = obj["category"]
                if cat_name not in category_cache:
                    category_cache[cat_name] = self.get_or_create_category(cat_name, conn=conn)

            # Batch insert objects
            for i, obj in enumerate(objects_data):
                object_id = object_ids[i]
                category_id = category_cache[obj["category"]]
                mask_path = mask_paths.get(i)

                bbox = obj["bbox"]
                area = bbox[2] * bbox[3] if len(bbox) >= 4 else None

                conn.execute(
                    """INSERT INTO objects (object_id, category_id, source_id, frame_id, project_id,
                       bbox_x, bbox_y, bbox_w, bbox_h, mask_path, area, confidence, detection_model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (object_id, category_id, source_id, obj.get("frame_id"), project_id,
                     bbox[0], bbox[1], bbox[2], bbox[3], mask_path, area,
                     obj.get("confidence"), obj.get("detection_model"))
                )

                if obj.get("embedding"):
                    self._queue_embedding(conn, object_id, obj["embedding"])

            conn.commit()
        finally:
            conn.close()

        # Sync embeddings (after transaction committed)
        for obj_id, obj in zip(object_ids, objects_data):
            if obj.get("embedding"):
                self._sync_embedding(
                    obj_id, obj["embedding"], obj["category"], obj.get("confidence")
                )

        return object_ids

    def get_object(self, object_id: str) -> Optional[dict]:
        """Get object by ID with category info."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT o.*, c.name as category_name, c.supercategory
               FROM objects o
               LEFT JOIN categories c ON o.category_id = c.category_id
               WHERE o.object_id = ?""",
            (object_id,)
        ).fetchone()
        conn.close()

        if not row:
            return None

        obj = dict(row)
        obj["bbox"] = [obj["bbox_x"], obj["bbox_y"], obj["bbox_w"], obj["bbox_h"]]
        return obj

    def update_object(self, object_id: str, updates: dict) -> bool:
        """Update object fields."""
        allowed_fields = {
            "confidence", "is_validated", "validated_by", "quality_score",
            "is_occluded", "is_truncated", "is_difficult"
        }
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}

        if not filtered:
            return False

        set_clause = ", ".join(f"{k} = ?" for k in filtered.keys())
        values = list(filtered.values()) + [object_id]

        conn = self._get_conn()
        cursor = conn.execute(
            f"UPDATE objects SET {set_clause} WHERE object_id = ?", values
        )
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    def delete_object(self, object_id: str) -> bool:
        """Delete object and its associated data."""
        obj = self.get_object(object_id)
        if not obj:
            return False

        # Delete mask file
        if obj.get("mask_path"):
            mask_file = self.masks_path / obj["mask_path"]
            if mask_file.exists():
                mask_file.unlink()

        # Delete from ChromaDB
        try:
            collection = self._get_collection()
            collection.delete(ids=[object_id])
        except Exception:
            pass

        # Delete from SQLite
        conn = self._get_conn()
        conn.execute("DELETE FROM objects WHERE object_id = ?", (object_id,))
        conn.commit()
        conn.close()
        return True

    # ==================== Search & Query ====================

    def search_objects(
        self,
        source_id: Optional[str] = None,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        is_validated: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Search objects with filters."""
        conditions = []
        params = []

        if source_id:
            conditions.append("o.source_id = ?")
            params.append(source_id)
        if category:
            conditions.append("c.name = ?")
            params.append(category)
        if project_id:
            conditions.append("o.project_id = ?")
            params.append(project_id)
        if min_confidence is not None:
            conditions.append("o.confidence >= ?")
            params.append(min_confidence)
        if is_validated is not None:
            conditions.append("o.is_validated = ?")
            params.append(is_validated)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        conn = self._get_conn()
        rows = conn.execute(
            f"""SELECT o.*, c.name as category_name
                FROM objects o
                LEFT JOIN categories c ON o.category_id = c.category_id
                WHERE {where_clause}
                ORDER BY o.created_at DESC
                LIMIT ? OFFSET ?""",
            params
        ).fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10,
        category: Optional[str] = None,
        min_confidence: Optional[float] = None,
        use_cache: bool = True,
    ) -> List[dict]:
        """Search similar objects by embedding vector.

        Args:
            embedding: Query embedding vector (e.g., 768-dim for DINOv2)
            top_k: Number of results to return
            category: Filter by category name
            min_confidence: Minimum confidence threshold
            use_cache: Whether to use cached results

        Returns:
            List of object dicts with similarity scores
        """
        start_time = time.time()
        cache_hit = False

        # Check cache first
        if use_cache:
            cached = self._search_cache.get(embedding, top_k, category, min_confidence)
            if cached is not None:
                cache_hit = True
                duration_ms = (time.time() - start_time) * 1000
                self._search_metrics.record_search(duration_ms, len(cached), cache_hit)
                return cached

        collection = self._get_collection()

        # Build where filter
        where = {}
        if category:
            where["category"] = category
        if min_confidence is not None:
            where["confidence"] = {"$gte": min_confidence}

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where if where else None,
            include=["metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            duration_ms = (time.time() - start_time) * 1000
            self._search_metrics.record_search(duration_ms, 0, cache_hit)
            return []

        # Get full object data from SQLite
        objects = []
        for obj_id, metadata, distance in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            obj = self.get_object(obj_id)
            if obj:
                obj["similarity"] = 1 - distance  # Convert distance to similarity
                objects.append(obj)

        # Cache the results
        if use_cache:
            self._search_cache.set(embedding, top_k, category, min_confidence, objects)

        duration_ms = (time.time() - start_time) * 1000
        self._search_metrics.record_search(duration_ms, len(objects), cache_hit)

        return objects

    def search_by_embeddings_batch(
        self,
        embeddings: List[List[float]],
        top_k: int = 10,
        category: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[List[dict]]:
        """Batch search for multiple embeddings.

        More efficient than calling search_by_embedding multiple times
        as it batches the ChromaDB query.

        Args:
            embeddings: List of query embedding vectors
            top_k: Number of results per query
            category: Filter by category name
            min_confidence: Minimum confidence threshold

        Returns:
            List of result lists, one per query embedding
        """
        if not embeddings:
            return []

        start_time = time.time()
        collection = self._get_collection()

        # Build where filter
        where = {}
        if category:
            where["category"] = category
        if min_confidence is not None:
            where["confidence"] = {"$gte": min_confidence}

        # Batch query
        results = collection.query(
            query_embeddings=embeddings,
            n_results=top_k,
            where=where if where else None,
            include=["metadatas", "distances"]
        )

        # Process results for each query
        all_objects = []
        for query_idx in range(len(embeddings)):
            if not results["ids"] or query_idx >= len(results["ids"]):
                all_objects.append([])
                continue

            query_ids = results["ids"][query_idx]
            query_metadatas = results["metadatas"][query_idx]
            query_distances = results["distances"][query_idx]

            objects = []
            for obj_id, metadata, distance in zip(query_ids, query_metadatas, query_distances):
                obj = self.get_object(obj_id)
                if obj:
                    obj["similarity"] = 1 - distance
                    objects.append(obj)
            all_objects.append(objects)

        duration_ms = (time.time() - start_time) * 1000
        total_results = sum(len(objs) for objs in all_objects)
        self._search_metrics.record_search(duration_ms, total_results, False)

        return all_objects

    def get_search_metrics(self) -> dict:
        """Get embedding search performance metrics."""
        return self._search_metrics.get_stats()

    def reset_search_metrics(self):
        """Reset search metrics."""
        self._search_metrics.reset()

    def clear_search_cache(self):
        """Clear the embedding search cache."""
        self._search_cache.clear()

    def optimize_index(self) -> dict:
        """Optimize ChromaDB index for better search performance.

        Note: ChromaDB with HNSW automatically handles index optimization,
        but this method can be used to get index stats and trigger maintenance.
        """
        collection = self._get_collection()

        # Get collection stats
        count = collection.count()

        return {
            "collection_name": collection.name,
            "total_embeddings": count,
            "index_type": "HNSW",
            "space": "cosine",
            "optimized": True,
        }

    def get_embedding_stats(self) -> dict:
        """Get embedding storage statistics."""
        collection = self._get_collection()
        count = collection.count()

        # Get pending sync count from outbox
        conn = self._get_conn()
        pending = conn.execute("SELECT COUNT(*) FROM embedding_outbox").fetchone()[0]
        conn.close()

        return {
            "indexed_embeddings": count,
            "pending_sync": pending,
            "search_cache_size": len(self._search_cache._cache),
            "search_metrics": self._search_metrics.get_stats(),
        }

    def count_objects(
        self,
        source_id: Optional[str] = None,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """Count objects matching filters."""
        conditions = []
        params = []

        if source_id:
            conditions.append("o.source_id = ?")
            params.append(source_id)
        if category:
            conditions.append("c.name = ?")
            params.append(category)
        if project_id:
            conditions.append("o.project_id = ?")
            params.append(project_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        conn = self._get_conn()
        row = conn.execute(
            f"""SELECT COUNT(*) as count FROM objects o
                LEFT JOIN categories c ON o.category_id = c.category_id
                WHERE {where_clause}""",
            params
        ).fetchone()
        conn.close()
        return row["count"]

    # ==================== Track Management (Re-ID) ====================

    def create_track(
        self,
        source_id: str,
        object_ids: List[str],
        category_name: Optional[str] = None,
    ) -> str:
        """Create a track from a sequence of objects."""
        track_id = f"trk_{uuid.uuid4().hex[:12]}"
        category_id = self.get_or_create_category(category_name) if category_name else None

        conn = self._get_conn()

        # Get frame indices for start/end
        frame_indices = []
        confidences = []
        for obj_id in object_ids:
            row = conn.execute(
                """SELECT f.frame_idx, o.confidence FROM objects o
                   LEFT JOIN frames f ON o.frame_id = f.frame_id
                   WHERE o.object_id = ?""",
                (obj_id,)
            ).fetchone()
            if row:
                if row["frame_idx"] is not None:
                    frame_indices.append(row["frame_idx"])
                if row["confidence"] is not None:
                    confidences.append(row["confidence"])

        start_frame = min(frame_indices) if frame_indices else None
        end_frame = max(frame_indices) if frame_indices else None
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Create track
        conn.execute(
            """INSERT INTO tracks (track_id, source_id, category_id,
               start_frame_idx, end_frame_idx, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (track_id, source_id, category_id, start_frame, end_frame, avg_confidence)
        )

        # Link objects to track
        for idx, obj_id in enumerate(object_ids):
            conn.execute(
                "INSERT INTO track_objects (track_id, object_id, sequence_idx) VALUES (?, ?, ?)",
                (track_id, obj_id, idx)
            )

        conn.commit()
        conn.close()
        return track_id

    def get_track(self, track_id: str) -> Optional[dict]:
        """Get track with its objects."""
        conn = self._get_conn()
        track = conn.execute(
            """SELECT t.*, c.name as category_name FROM tracks t
               LEFT JOIN categories c ON t.category_id = c.category_id
               WHERE t.track_id = ?""",
            (track_id,)
        ).fetchone()

        if not track:
            conn.close()
            return None

        # Get objects in track
        objects = conn.execute(
            """SELECT o.*, to_.sequence_idx FROM track_objects to_
               JOIN objects o ON to_.object_id = o.object_id
               WHERE to_.track_id = ?
               ORDER BY to_.sequence_idx""",
            (track_id,)
        ).fetchall()

        conn.close()

        result = dict(track)
        result["objects"] = [dict(obj) for obj in objects]
        return result

    def list_tracks(
        self,
        source_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """List tracks with optional filters."""
        conditions = []
        params = []

        if source_id:
            conditions.append("t.source_id = ?")
            params.append(source_id)
        if category:
            conditions.append("c.name = ?")
            params.append(category)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        conn = self._get_conn()
        rows = conn.execute(
            f"""SELECT t.*, c.name as category_name,
                       (SELECT COUNT(*) FROM track_objects WHERE track_id = t.track_id) as object_count
                FROM tracks t
                LEFT JOIN categories c ON t.category_id = c.category_id
                WHERE {where_clause}
                ORDER BY t.created_at DESC
                LIMIT ? OFFSET ?""",
            params
        ).fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_track(
        self,
        track_id: str,
        category_name: Optional[str] = None,
        is_validated: Optional[bool] = None,
        validated_by: Optional[str] = None,
    ) -> bool:
        """Update track fields."""
        track = self.get_track(track_id)
        if not track:
            return False

        conn = self._get_conn()
        try:
            updates = []
            params = []

            if category_name is not None:
                category_id = self.get_or_create_category(category_name, conn=conn)
                updates.append("category_id = ?")
                params.append(category_id)

            if is_validated is not None:
                updates.append("is_validated = ?")
                params.append(is_validated)

            if validated_by is not None:
                updates.append("validated_by = ?")
                params.append(validated_by)

            if not updates:
                conn.close()
                return False

            params.append(track_id)
            conn.execute(
                f"UPDATE tracks SET {', '.join(updates)} WHERE track_id = ?",
                params
            )
            conn.commit()
        finally:
            conn.close()

        return True

    def delete_track(self, track_id: str) -> bool:
        """Delete a track and its object associations."""
        track = self.get_track(track_id)
        if not track:
            return False

        conn = self._get_conn()
        try:
            # Delete track-object associations
            conn.execute("DELETE FROM track_objects WHERE track_id = ?", (track_id,))
            # Delete track
            conn.execute("DELETE FROM tracks WHERE track_id = ?", (track_id,))
            conn.commit()
        finally:
            conn.close()

        return True

    def merge_tracks(
        self,
        track_ids: List[str],
        new_category_name: Optional[str] = None,
    ) -> Optional[str]:
        """Merge multiple tracks into a single track.

        Objects are ordered by their sequence_idx within each track,
        then by track order in track_ids list.

        Args:
            track_ids: List of track IDs to merge (at least 2)
            new_category_name: Category for merged track (uses first track's category if None)

        Returns:
            New track ID or None if merge fails
        """
        if len(track_ids) < 2:
            return None

        # Get all tracks and validate
        tracks = []
        for tid in track_ids:
            track = self.get_track(tid)
            if not track:
                return None
            tracks.append(track)

        # Determine source_id (must be same for all tracks)
        source_ids = set(t["source_id"] for t in tracks)
        if len(source_ids) > 1:
            return None  # Cannot merge tracks from different sources

        source_id = tracks[0]["source_id"]
        category_name = new_category_name or tracks[0].get("category_name")

        # Collect all objects in order
        all_objects = []
        for track in tracks:
            all_objects.extend(track.get("objects", []))

        if not all_objects:
            return None

        object_ids = [obj["object_id"] for obj in all_objects]

        # Create new merged track
        new_track_id = self.create_track(source_id, object_ids, category_name)

        # Delete old tracks
        for tid in track_ids:
            self.delete_track(tid)

        return new_track_id

    def split_track(
        self,
        track_id: str,
        split_index: int,
    ) -> Optional[Tuple[str, str]]:
        """Split a track at the specified index.

        Args:
            track_id: Track to split
            split_index: Index at which to split (objects at and after this index go to second track)

        Returns:
            Tuple of (first_track_id, second_track_id) or None if split fails
        """
        track = self.get_track(track_id)
        if not track:
            return None

        objects = track.get("objects", [])
        if split_index < 1 or split_index >= len(objects):
            return None  # Invalid split point

        # Split objects
        first_objects = [obj["object_id"] for obj in objects[:split_index]]
        second_objects = [obj["object_id"] for obj in objects[split_index:]]

        source_id = track["source_id"]
        category_name = track.get("category_name")

        # Create two new tracks
        first_track_id = self.create_track(source_id, first_objects, category_name)
        second_track_id = self.create_track(source_id, second_objects, category_name)

        # Delete original track
        self.delete_track(track_id)

        return (first_track_id, second_track_id)

    def add_objects_to_track(
        self,
        track_id: str,
        object_ids: List[str],
        insert_at: Optional[int] = None,
    ) -> bool:
        """Add objects to an existing track.

        Args:
            track_id: Target track
            object_ids: Objects to add
            insert_at: Position to insert (None = append at end)

        Returns:
            True if successful
        """
        track = self.get_track(track_id)
        if not track:
            return False

        conn = self._get_conn()
        try:
            # Get current max sequence
            existing_objects = track.get("objects", [])
            max_seq = len(existing_objects)

            if insert_at is None:
                # Append at end
                for i, obj_id in enumerate(object_ids):
                    conn.execute(
                        "INSERT INTO track_objects (track_id, object_id, sequence_idx) VALUES (?, ?, ?)",
                        (track_id, obj_id, max_seq + i)
                    )
            else:
                # Insert at position - shift existing objects
                insert_at = max(0, min(insert_at, max_seq))

                # Shift objects after insert point
                conn.execute(
                    """UPDATE track_objects
                       SET sequence_idx = sequence_idx + ?
                       WHERE track_id = ? AND sequence_idx >= ?""",
                    (len(object_ids), track_id, insert_at)
                )

                # Insert new objects
                for i, obj_id in enumerate(object_ids):
                    conn.execute(
                        "INSERT INTO track_objects (track_id, object_id, sequence_idx) VALUES (?, ?, ?)",
                        (track_id, obj_id, insert_at + i)
                    )

            # Update track metadata
            self._update_track_metadata(conn, track_id)
            conn.commit()
        finally:
            conn.close()

        return True

    def remove_objects_from_track(
        self,
        track_id: str,
        object_ids: List[str],
    ) -> bool:
        """Remove objects from a track.

        Args:
            track_id: Target track
            object_ids: Objects to remove

        Returns:
            True if successful
        """
        track = self.get_track(track_id)
        if not track:
            return False

        conn = self._get_conn()
        try:
            # Remove specified objects
            placeholders = ",".join("?" * len(object_ids))
            conn.execute(
                f"DELETE FROM track_objects WHERE track_id = ? AND object_id IN ({placeholders})",
                [track_id] + object_ids
            )

            # Re-sequence remaining objects
            remaining = conn.execute(
                """SELECT object_id FROM track_objects
                   WHERE track_id = ? ORDER BY sequence_idx""",
                (track_id,)
            ).fetchall()

            for i, row in enumerate(remaining):
                conn.execute(
                    "UPDATE track_objects SET sequence_idx = ? WHERE track_id = ? AND object_id = ?",
                    (i, track_id, row["object_id"])
                )

            # Update track metadata
            self._update_track_metadata(conn, track_id)
            conn.commit()
        finally:
            conn.close()

        return True

    def _update_track_metadata(self, conn: sqlite3.Connection, track_id: str):
        """Update track's start/end frame and confidence based on current objects."""
        # Get frame indices and confidences
        rows = conn.execute(
            """SELECT f.frame_idx, o.confidence
               FROM track_objects to_
               JOIN objects o ON to_.object_id = o.object_id
               LEFT JOIN frames f ON o.frame_id = f.frame_id
               WHERE to_.track_id = ?""",
            (track_id,)
        ).fetchall()

        frame_indices = [r["frame_idx"] for r in rows if r["frame_idx"] is not None]
        confidences = [r["confidence"] for r in rows if r["confidence"] is not None]

        start_frame = min(frame_indices) if frame_indices else None
        end_frame = max(frame_indices) if frame_indices else None
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        conn.execute(
            """UPDATE tracks SET start_frame_idx = ?, end_frame_idx = ?, avg_confidence = ?
               WHERE track_id = ?""",
            (start_frame, end_frame, avg_confidence, track_id)
        )

    # ==================== Dataset Export ====================

    def create_dataset(
        self,
        name: str,
        format: str = "yolo",
        filter_config: Optional[dict] = None,
        split_config: Optional[dict] = None,
    ) -> str:
        """Create a dataset configuration."""
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        filter_config = filter_config or {}
        split_config = split_config or {"train": 0.8, "val": 0.1, "test": 0.1}

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO datasets (dataset_id, name, format, filter_config, split_config)
               VALUES (?, ?, ?, ?, ?)""",
            (dataset_id, name, format, json.dumps(filter_config), json.dumps(split_config))
        )
        conn.commit()
        conn.close()
        return dataset_id

    def build_dataset(self, dataset_id: str) -> dict:
        """Populate dataset with objects based on filter config."""
        conn = self._get_conn()
        dataset = conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()

        if not dataset:
            conn.close()
            raise ValueError(f"Dataset {dataset_id} not found")

        filter_config = json.loads(dataset["filter_config"]) if dataset["filter_config"] else {}
        split_config = json.loads(dataset["split_config"]) if dataset["split_config"] else {"train": 0.8, "val": 0.1, "test": 0.1}

        # Ensure filter_config is a dict
        if filter_config is None:
            filter_config = {}

        # Build query based on filters
        conditions = ["1=1"]
        params = []

        if filter_config.get("categories"):
            placeholders = ",".join("?" * len(filter_config["categories"]))
            conditions.append(f"c.name IN ({placeholders})")
            params.extend(filter_config["categories"])

        if filter_config.get("min_confidence"):
            conditions.append("o.confidence >= ?")
            params.append(filter_config["min_confidence"])

        if filter_config.get("is_validated"):
            conditions.append("o.is_validated = ?")
            params.append(filter_config["is_validated"])

        if filter_config.get("project_id"):
            conditions.append("o.project_id = ?")
            params.append(filter_config["project_id"])

        # Get matching objects
        objects = conn.execute(
            f"""SELECT o.object_id FROM objects o
                LEFT JOIN categories c ON o.category_id = c.category_id
                WHERE {" AND ".join(conditions)}
                ORDER BY RANDOM()""",
            params
        ).fetchall()

        object_ids = [row["object_id"] for row in objects]
        total = len(object_ids)

        if total == 0:
            conn.close()
            return {"dataset_id": dataset_id, "object_count": 0, "splits": {}}

        # Split objects
        train_end = int(total * split_config.get("train", 0.8))
        val_end = train_end + int(total * split_config.get("val", 0.1))

        splits = {
            "train": object_ids[:train_end],
            "val": object_ids[train_end:val_end],
            "test": object_ids[val_end:]
        }

        # Clear existing and insert new
        conn.execute("DELETE FROM dataset_objects WHERE dataset_id = ?", (dataset_id,))

        for split_name, split_ids in splits.items():
            for obj_id in split_ids:
                conn.execute(
                    "INSERT INTO dataset_objects (dataset_id, object_id, split) VALUES (?, ?, ?)",
                    (dataset_id, obj_id, split_name)
                )

        conn.execute(
            "UPDATE datasets SET object_count = ? WHERE dataset_id = ?",
            (total, dataset_id)
        )

        conn.commit()
        conn.close()

        return {
            "dataset_id": dataset_id,
            "object_count": total,
            "splits": {k: len(v) for k, v in splits.items()}
        }

    # ==================== Mask Storage ====================

    def _save_mask(self, object_id: str, mask_data: bytes) -> str:
        """Save mask with sharded directory structure."""
        hash_val = hashlib.md5(object_id.encode()).hexdigest()
        shard_path = self.masks_path / hash_val[:2] / hash_val[2:4]
        shard_path.mkdir(parents=True, exist_ok=True)

        mask_file = shard_path / f"{object_id}.png"
        mask_file.write_bytes(mask_data)

        return str(mask_file.relative_to(self.masks_path))

    def get_mask(self, object_id: str) -> Optional[bytes]:
        """Get mask data for an object."""
        obj = self.get_object(object_id)
        if not obj or not obj.get("mask_path"):
            return None

        mask_file = self.masks_path / obj["mask_path"]
        if mask_file.exists():
            return mask_file.read_bytes()
        return None

    # ==================== Embedding Sync ====================

    def _queue_embedding(self, conn: sqlite3.Connection, object_id: str, embedding: List[float]):
        """Queue embedding for ChromaDB sync."""
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        conn.execute(
            """INSERT INTO embedding_outbox (object_id, version_id, operation, embedding)
               VALUES (?, 'dinov2_base_v1', 'upsert', ?)""",
            (object_id, embedding_bytes)
        )

    def _sync_embedding(
        self,
        object_id: str,
        embedding: List[float],
        category: str,
        confidence: Optional[float],
    ):
        """Sync embedding to ChromaDB."""
        try:
            collection = self._get_collection()
            collection.upsert(
                ids=[object_id],
                embeddings=[embedding],
                metadatas=[{
                    "object_id": object_id,
                    "category": category,
                    "confidence": confidence or 0.0,
                }]
            )

            # Mark as synced
            conn = self._get_conn()
            conn.execute(
                """UPDATE embedding_outbox SET status = 'synced', synced_at = CURRENT_TIMESTAMP
                   WHERE object_id = ? AND status = 'pending'""",
                (object_id,)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            # Mark as failed
            conn = self._get_conn()
            conn.execute(
                """UPDATE embedding_outbox SET status = 'failed',
                   error_message = ?, retry_count = retry_count + 1
                   WHERE object_id = ? AND status = 'pending'""",
                (str(e), object_id)
            )
            conn.commit()
            conn.close()

    def sync_pending_embeddings(self) -> dict:
        """Sync all pending embeddings to ChromaDB."""
        conn = self._get_conn()
        pending = conn.execute(
            """SELECT eo.*, o.confidence, c.name as category
               FROM embedding_outbox eo
               JOIN objects o ON eo.object_id = o.object_id
               LEFT JOIN categories c ON o.category_id = c.category_id
               WHERE eo.status = 'pending' AND eo.retry_count < 3
               LIMIT 100"""
        ).fetchall()
        conn.close()

        synced = 0
        failed = 0

        for row in pending:
            try:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                self._sync_embedding(
                    row["object_id"], embedding, row["category"], row["confidence"]
                )
                synced += 1
            except Exception:
                failed += 1

        return {"synced": synced, "failed": failed, "pending": len(pending) - synced - failed}

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get registry statistics."""
        conn = self._get_conn()

        stats = {
            "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "objects": conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
            "categories": conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0],
            "tracks": conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0],
            "datasets": conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "validated_objects": conn.execute(
                "SELECT COUNT(*) FROM objects WHERE is_validated = 1"
            ).fetchone()[0],
        }

        # Objects per category
        category_counts = conn.execute(
            """SELECT c.name, COUNT(o.object_id) as count
               FROM categories c
               LEFT JOIN objects o ON c.category_id = o.category_id
               GROUP BY c.category_id
               ORDER BY count DESC"""
        ).fetchall()
        stats["objects_per_category"] = {row["name"]: row["count"] for row in category_counts}

        conn.close()
        return stats
