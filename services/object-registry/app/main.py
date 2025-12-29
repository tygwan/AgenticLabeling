"""Object Registry Service - REST API."""
import base64
import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .registry import ObjectRegistry


# Initialize registry
DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~/dev/AgenticLabeling/data/registry"))
registry = ObjectRegistry(data_dir=DATA_DIR)

app = FastAPI(
    title="Object Registry",
    description="Central registry for managing detected objects with hybrid storage",
    version="0.1.0",
)


# ==================== Schemas ====================

class SourceCreate(BaseModel):
    source_type: str  # 'image', 'video', 'text'
    file_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    metadata: Optional[dict] = None


class ObjectCreate(BaseModel):
    source_id: str
    category: str
    bbox: List[float]  # [x, y, w, h]
    confidence: Optional[float] = None
    detection_model: Optional[str] = None
    mask_base64: Optional[str] = None
    embedding: Optional[List[float]] = None
    project_id: Optional[str] = None
    frame_id: Optional[str] = None


class ObjectBatchCreate(BaseModel):
    source_id: str
    project_id: Optional[str] = None
    objects: List[dict]  # Each: {category, bbox, confidence, mask_base64, embedding}


class ObjectUpdate(BaseModel):
    confidence: Optional[float] = None
    is_validated: Optional[bool] = None
    validated_by: Optional[str] = None
    quality_score: Optional[float] = None
    is_occluded: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_difficult: Optional[bool] = None


class TrackCreate(BaseModel):
    source_id: str
    object_ids: List[str]
    category: Optional[str] = None


class TrackUpdate(BaseModel):
    category: Optional[str] = None
    is_validated: Optional[bool] = None
    validated_by: Optional[str] = None


class TrackMerge(BaseModel):
    track_ids: List[str]
    category: Optional[str] = None


class TrackSplit(BaseModel):
    split_index: int


class TrackAddObjects(BaseModel):
    object_ids: List[str]
    insert_at: Optional[int] = None


class TrackRemoveObjects(BaseModel):
    object_ids: List[str]


class DatasetCreate(BaseModel):
    name: str
    format: str = "yolo"
    filter_config: Optional[dict] = None
    split_config: Optional[dict] = None


class EmbeddingSearch(BaseModel):
    embedding: List[float]
    top_k: int = 10
    category: Optional[str] = None
    min_confidence: Optional[float] = None


# ==================== Health ====================

@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = registry.get_stats()
    return {
        "status": "healthy",
        "service": "object-registry",
        "stats": {
            "objects": stats["objects"],
            "sources": stats["sources"],
            "categories": stats["categories"],
        }
    }


# ==================== Sources ====================

@app.post("/sources")
async def create_source(source: SourceCreate):
    """Register a new source (image/video)."""
    try:
        source_id = registry.register_source(
            source_type=source.source_type,
            file_path=source.file_path,
            width=source.width,
            height=source.height,
            frame_count=source.frame_count,
            fps=source.fps,
            metadata=source.metadata,
        )
        return {"success": True, "source_id": source_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/sources/{source_id}")
async def get_source(source_id: str):
    """Get source by ID."""
    source = registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"success": True, "data": source}


# ==================== Categories ====================

@app.get("/categories")
async def list_categories():
    """List all categories."""
    categories = registry.list_categories()
    return {"success": True, "data": categories}


@app.post("/categories")
async def create_category(name: str = Form(...), supercategory: Optional[str] = Form(None)):
    """Create or get a category."""
    category_id = registry.get_or_create_category(name, supercategory)
    return {"success": True, "category_id": category_id, "name": name}


# ==================== Objects ====================

@app.post("/objects")
async def create_object(obj: ObjectCreate):
    """Register a new detected object."""
    try:
        mask_data = None
        if obj.mask_base64:
            mask_data = base64.b64decode(obj.mask_base64)

        object_id = registry.register_object(
            source_id=obj.source_id,
            category_name=obj.category,
            bbox=obj.bbox,
            confidence=obj.confidence,
            detection_model=obj.detection_model,
            mask_data=mask_data,
            embedding=obj.embedding,
            project_id=obj.project_id,
            frame_id=obj.frame_id,
        )
        return {"success": True, "object_id": object_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/objects/batch")
async def create_objects_batch(batch: ObjectBatchCreate):
    """Register multiple objects in batch."""
    try:
        objects_data = []
        for obj in batch.objects:
            data = {
                "category": obj["category"],
                "bbox": obj["bbox"],
                "confidence": obj.get("confidence"),
                "detection_model": obj.get("detection_model"),
                "embedding": obj.get("embedding"),
                "frame_id": obj.get("frame_id"),
            }
            if obj.get("mask_base64"):
                data["mask_data"] = base64.b64decode(obj["mask_base64"])
            objects_data.append(data)

        object_ids = registry.register_objects_batch(
            source_id=batch.source_id,
            objects_data=objects_data,
            project_id=batch.project_id,
        )
        return {"success": True, "object_ids": object_ids, "count": len(object_ids)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/objects/{object_id}")
async def get_object(object_id: str):
    """Get object by ID."""
    obj = registry.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")
    return {"success": True, "data": obj}


@app.patch("/objects/{object_id}")
async def update_object(object_id: str, updates: ObjectUpdate):
    """Update object fields."""
    success = registry.update_object(object_id, updates.model_dump(exclude_none=True))
    if not success:
        raise HTTPException(status_code=404, detail="Object not found or no valid updates")
    return {"success": True, "object_id": object_id}


@app.delete("/objects/{object_id}")
async def delete_object(object_id: str):
    """Delete an object."""
    success = registry.delete_object(object_id)
    if not success:
        raise HTTPException(status_code=404, detail="Object not found")
    return {"success": True, "object_id": object_id}


@app.get("/objects/{object_id}/mask")
async def get_object_mask(object_id: str):
    """Get mask image for an object."""
    mask_data = registry.get_mask(object_id)
    if not mask_data:
        raise HTTPException(status_code=404, detail="Mask not found")
    return Response(content=mask_data, media_type="image/png")


# ==================== Search ====================

@app.get("/objects")
async def search_objects(
    source_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None),
    is_validated: Optional[bool] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
):
    """Search objects with filters."""
    objects = registry.search_objects(
        source_id=source_id,
        category=category,
        project_id=project_id,
        min_confidence=min_confidence,
        is_validated=is_validated,
        limit=limit,
        offset=offset,
    )
    total = registry.count_objects(source_id, category, project_id)
    return {"success": True, "data": objects, "total": total, "limit": limit, "offset": offset}


@app.post("/objects/search/embedding")
async def search_by_embedding(search: EmbeddingSearch):
    """Search similar objects by embedding vector."""
    try:
        results = registry.search_by_embedding(
            embedding=search.embedding,
            top_k=search.top_k,
            category=search.category,
            min_confidence=search.min_confidence,
        )
        return {"success": True, "data": results, "count": len(results)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ==================== Tracks ====================

@app.post("/tracks")
async def create_track(track: TrackCreate):
    """Create a track from a sequence of objects."""
    try:
        track_id = registry.create_track(
            source_id=track.source_id,
            object_ids=track.object_ids,
            category_name=track.category,
        )
        return {"success": True, "track_id": track_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/tracks")
async def list_tracks(
    source_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
):
    """List tracks with optional filters."""
    tracks = registry.list_tracks(
        source_id=source_id,
        category=category,
        limit=limit,
        offset=offset,
    )
    return {"success": True, "data": tracks, "count": len(tracks)}


@app.get("/tracks/{track_id}")
async def get_track(track_id: str):
    """Get track with its objects."""
    track = registry.get_track(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return {"success": True, "data": track}


@app.patch("/tracks/{track_id}")
async def update_track(track_id: str, updates: TrackUpdate):
    """Update track fields."""
    success = registry.update_track(
        track_id=track_id,
        category_name=updates.category,
        is_validated=updates.is_validated,
        validated_by=updates.validated_by,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Track not found or no valid updates")
    return {"success": True, "track_id": track_id}


@app.delete("/tracks/{track_id}")
async def delete_track(track_id: str):
    """Delete a track."""
    success = registry.delete_track(track_id)
    if not success:
        raise HTTPException(status_code=404, detail="Track not found")
    return {"success": True, "track_id": track_id}


@app.post("/tracks/merge")
async def merge_tracks(merge_request: TrackMerge):
    """Merge multiple tracks into a single track."""
    try:
        new_track_id = registry.merge_tracks(
            track_ids=merge_request.track_ids,
            new_category_name=merge_request.category,
        )
        if not new_track_id:
            raise HTTPException(
                status_code=400,
                detail="Merge failed. Ensure all tracks exist, belong to the same source, and at least 2 tracks are provided."
            )
        return {"success": True, "track_id": new_track_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/tracks/{track_id}/split")
async def split_track(track_id: str, split_request: TrackSplit):
    """Split a track at the specified index."""
    try:
        result = registry.split_track(
            track_id=track_id,
            split_index=split_request.split_index,
        )
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Split failed. Ensure track exists and split_index is valid (1 <= index < object_count)."
            )
        first_track_id, second_track_id = result
        return {
            "success": True,
            "first_track_id": first_track_id,
            "second_track_id": second_track_id,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/tracks/{track_id}/objects")
async def add_objects_to_track(track_id: str, request: TrackAddObjects):
    """Add objects to an existing track."""
    success = registry.add_objects_to_track(
        track_id=track_id,
        object_ids=request.object_ids,
        insert_at=request.insert_at,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Track not found")
    return {"success": True, "track_id": track_id}


@app.delete("/tracks/{track_id}/objects")
async def remove_objects_from_track(track_id: str, request: TrackRemoveObjects):
    """Remove objects from a track."""
    success = registry.remove_objects_from_track(
        track_id=track_id,
        object_ids=request.object_ids,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Track not found")
    return {"success": True, "track_id": track_id}


# ==================== Datasets ====================

@app.post("/datasets")
async def create_dataset(dataset: DatasetCreate):
    """Create a dataset configuration."""
    try:
        dataset_id = registry.create_dataset(
            name=dataset.name,
            format=dataset.format,
            filter_config=dataset.filter_config,
            split_config=dataset.split_config,
        )
        return {"success": True, "dataset_id": dataset_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/datasets/{dataset_id}/build")
async def build_dataset(dataset_id: str):
    """Populate dataset with objects based on filter config."""
    try:
        result = registry.build_dataset(dataset_id)
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ==================== Sync & Stats ====================

@app.post("/sync/embeddings")
async def sync_embeddings():
    """Sync pending embeddings to ChromaDB."""
    result = registry.sync_pending_embeddings()
    return {"success": True, "data": result}


@app.get("/stats")
async def get_stats():
    """Get registry statistics."""
    stats = registry.get_stats()
    return {"success": True, "data": stats}


# ==================== Search Optimization ====================

@app.post("/objects/search/batch")
async def search_embeddings_batch(
    embeddings: List[List[float]],
    top_k: int = 10,
    category: Optional[str] = None,
    min_confidence: Optional[float] = None,
):
    """Batch search for multiple embeddings.

    More efficient than multiple single searches.

    Args:
        embeddings: List of embedding vectors
        top_k: Number of results per query
        category: Filter by category name
        min_confidence: Minimum confidence threshold
    """
    try:
        results = registry.search_by_embeddings_batch(
            embeddings=embeddings,
            top_k=top_k,
            category=category,
            min_confidence=min_confidence,
        )
        return {"success": True, "data": results, "count": len(results)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/search/metrics")
async def get_search_metrics():
    """Get embedding search performance metrics."""
    metrics = registry.get_search_metrics()
    return {"success": True, "data": metrics}


@app.post("/search/metrics/reset")
async def reset_search_metrics():
    """Reset search metrics."""
    registry.reset_search_metrics()
    return {"success": True, "message": "Search metrics reset"}


@app.post("/search/cache/clear")
async def clear_search_cache():
    """Clear the embedding search cache."""
    registry.clear_search_cache()
    return {"success": True, "message": "Search cache cleared"}


@app.get("/embeddings/stats")
async def get_embedding_stats():
    """Get embedding storage and search statistics."""
    stats = registry.get_embedding_stats()
    return {"success": True, "data": stats}


@app.post("/embeddings/optimize")
async def optimize_embedding_index():
    """Optimize embedding index for better search performance."""
    result = registry.optimize_index()
    return {"success": True, "data": result}
