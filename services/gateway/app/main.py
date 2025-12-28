"""API Gateway - Routes requests to appropriate services."""
import os
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AgenticLabeling Gateway",
    description="API Gateway for AgenticLabeling microservices",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
SERVICES = {
    "detection": os.getenv("DETECTION_URL", "http://detection-agent:8001"),
    "segmentation": os.getenv("SEGMENTATION_URL", "http://segmentation-agent:8002"),
    "classification": os.getenv("CLASSIFICATION_URL", "http://classification-agent:8003"),
    "labeling": os.getenv("LABELING_URL", "http://labeling-agent:8004"),
    "training": os.getenv("TRAINING_URL", "http://training-agent:8005"),
    "registry": os.getenv("REGISTRY_URL", "http://object-registry:8010"),
    "data_manager": os.getenv("DATA_MANAGER_URL", "http://data-manager:8006"),
    "evaluation": os.getenv("EVALUATION_URL", "http://evaluation-agent:8007"),
}


@app.get("/health")
async def health():
    """Check gateway and service health."""
    service_status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in SERVICES.items():
            try:
                resp = await client.get(f"{url}/health")
                service_status[name] = resp.json() if resp.status_code == 200 else "unhealthy"
            except Exception:
                service_status[name] = "unreachable"

    return {"gateway": "healthy", "services": service_status}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
):
    """Forward detection request to detection-agent."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"classes": classes, "confidence": str(confidence)}

        resp = await client.post(f"{SERVICES['detection']}/detect", files=files, data=data)
        return resp.json()


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    boxes: str = Form(...),
):
    """Forward segmentation request to segmentation-agent."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"boxes": boxes}

        resp = await client.post(f"{SERVICES['segmentation']}/segment", files=files, data=data)
        return resp.json()


@app.post("/detect_and_segment")
async def detect_and_segment(
    image: UploadFile = File(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
):
    """Pipeline: detect objects then segment each one."""
    import json

    image_bytes = await image.read()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Detection
        files = {"image": (image.filename, image_bytes, image.content_type)}
        data = {"classes": classes, "confidence": str(confidence)}

        detect_resp = await client.post(
            f"{SERVICES['detection']}/detect", files=files, data=data
        )
        detect_result = detect_resp.json()

        if not detect_result.get("success"):
            return detect_result

        boxes = detect_result["data"]["boxes"]
        if not boxes:
            return {"success": True, "data": {"detections": [], "masks": []}}

        # Step 2: Segmentation
        files = {"image": (image.filename, image_bytes, image.content_type)}
        data = {"boxes": json.dumps(boxes)}

        segment_resp = await client.post(
            f"{SERVICES['segmentation']}/segment", files=files, data=data
        )
        segment_result = segment_resp.json()

        return {
            "success": True,
            "data": {
                "detections": detect_result["data"],
                "segmentation": segment_result.get("data"),
            },
        }


@app.post("/auto_label")
async def auto_label(
    image: UploadFile = File(...),
    project_id: str = Form(...),
    image_id: str = Form(...),
    classes: str = Form(...),
    confidence: float = Form(0.5),
    save: bool = Form(True),
    register: bool = Form(True),
):
    """Pipeline: detect -> segment -> save labels -> register objects.

    Complete auto-labeling pipeline that detects objects, segments them,
    saves the results to labeling-agent, and registers objects in the registry.
    """
    import base64
    import json

    image_bytes = await image.read()
    classes_list = [c.strip() for c in classes.split(",")]

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Detection
        files = {"image": (image.filename, image_bytes, image.content_type)}
        data = {"classes": classes, "confidence": str(confidence)}

        detect_resp = await client.post(
            f"{SERVICES['detection']}/detect",
            files=files,
            data=data,
        )
        detect_result = detect_resp.json()

        if not detect_result.get("success"):
            return {"success": False, "error": "Detection failed", "detail": detect_result}

        boxes = detect_result["data"]["boxes"]
        labels = detect_result["data"]["labels"]
        scores = detect_result["data"].get("scores", [])
        image_size = detect_result["data"]["image_size"]

        if not boxes:
            return {
                "success": True,
                "data": {"detections": 0, "message": "No objects detected"},
                "saved": False,
                "registered": False,
            }

        # Step 2: Segmentation
        files = {"image": (image.filename, image_bytes, image.content_type)}
        segment_resp = await client.post(
            f"{SERVICES['segmentation']}/segment",
            files=files,
            data={"boxes": json.dumps(boxes)},
        )
        segment_result = segment_resp.json()

        masks = []
        masks_base64 = []
        if segment_result.get("success"):
            for m in segment_result["data"]["masks"]:
                mask_b64 = m["mask"]  # Already base64 encoded
                masks.append(mask_b64)
                masks_base64.append(mask_b64)

        # Step 3: Save to labeling-agent
        saved = False
        if save:
            label_data = {
                "project_id": project_id,
                "image_id": image_id,
                "boxes": boxes,
                "classes": labels,
                "masks": masks,
            }
            save_resp = await client.post(
                f"{SERVICES['labeling']}/labels/save",
                json=label_data,
            )
            saved = save_resp.json().get("success", False)

        # Step 4: Register objects in object-registry
        registered = False
        object_ids = []
        source_id = None

        if register:
            try:
                # Parse image size (can be dict or list)
                img_width = None
                img_height = None
                if image_size:
                    if isinstance(image_size, dict):
                        img_width = image_size.get("width")
                        img_height = image_size.get("height")
                    elif isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                        img_width = image_size[0]
                        img_height = image_size[1]

                # Register source (image)
                source_data = {
                    "source_type": "image",
                    "file_path": image_id,
                    "width": img_width,
                    "height": img_height,
                    "metadata": {"project_id": project_id, "filename": image.filename},
                }
                source_resp = await client.post(
                    f"{SERVICES['registry']}/sources",
                    json=source_data,
                )
                source_result = source_resp.json()

                if source_result.get("success"):
                    source_id = source_result["source_id"]

                    # Prepare batch objects
                    objects_data = []
                    for i, (box, label) in enumerate(zip(boxes, labels)):
                        obj = {
                            "category": label,
                            "bbox": box,  # [x, y, w, h]
                            "confidence": scores[i] if i < len(scores) else None,
                            "detection_model": "florence2",
                        }
                        # Add mask if available
                        if i < len(masks_base64) and masks_base64[i]:
                            obj["mask_base64"] = masks_base64[i]
                        objects_data.append(obj)

                    # Batch register objects
                    batch_data = {
                        "source_id": source_id,
                        "project_id": project_id,
                        "objects": objects_data,
                    }
                    batch_resp = await client.post(
                        f"{SERVICES['registry']}/objects/batch",
                        json=batch_data,
                    )
                    batch_result = batch_resp.json()

                    if batch_result.get("success"):
                        registered = True
                        object_ids = batch_result.get("object_ids", [])
            except Exception as e:
                # Log but don't fail the pipeline
                print(f"Registry error: {e}")

        return {
            "success": True,
            "data": {
                "image_id": image_id,
                "detections": len(boxes),
                "boxes": boxes,
                "labels": labels,
                "masks_count": len(masks),
                "image_size": image_size,
            },
            "saved": saved,
            "registered": registered,
            "registry": {
                "source_id": source_id,
                "object_ids": object_ids,
            } if registered else None,
        }


# ==================== Object Registry Proxy Endpoints ====================

@app.get("/registry/stats")
async def registry_stats():
    """Get object registry statistics."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['registry']}/stats")
        return resp.json()


@app.get("/registry/objects")
async def registry_objects(
    source_id: Optional[str] = None,
    category: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 100,
):
    """Search objects in registry."""
    params = {"limit": limit}
    if source_id:
        params["source_id"] = source_id
    if category:
        params["category"] = category
    if project_id:
        params["project_id"] = project_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['registry']}/objects", params=params)
        return resp.json()


@app.get("/registry/categories")
async def registry_categories():
    """List all categories in registry."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['registry']}/categories")
        return resp.json()


# ==================== Dataset Export Endpoints ====================

@app.post("/export")
async def export_dataset(
    dataset_name: str = Form(...),
    format: str = Form("yolo"),
    project_id: Optional[str] = Form(None),
    categories: Optional[str] = Form(None),
    min_confidence: Optional[float] = Form(None),
    is_validated: Optional[bool] = Form(None),
    train_ratio: float = Form(0.8),
    val_ratio: float = Form(0.1),
    test_ratio: float = Form(0.1),
):
    """Export dataset from Object Registry to YOLO or COCO format.

    Pipeline: Registry objects -> data-manager export -> zip file
    """
    import json

    # Build filter config
    filter_config = {}
    if project_id:
        filter_config["project_id"] = project_id
    if categories:
        filter_config["categories"] = [c.strip() for c in categories.split(",")]
    if min_confidence is not None:
        filter_config["min_confidence"] = min_confidence
    if is_validated is not None:
        filter_config["is_validated"] = is_validated

    # Build split config
    split_config = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{SERVICES['data_manager']}/registry/export",
            json={
                "dataset_name": dataset_name,
                "format": format,
                "filter_config": filter_config if filter_config else None,
                "split_config": split_config,
            },
        )
        return resp.json()


@app.get("/export/{dataset_name}/download")
async def download_dataset(dataset_name: str, format: str = "yolo"):
    """Download exported dataset as zip file."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            f"{SERVICES['data_manager']}/registry/export/{dataset_name}/download",
            params={"format": format},
        )

        if resp.status_code == 404:
            return {"success": False, "error": "Exported dataset not found"}

        # Return zip file
        from fastapi.responses import Response
        return Response(
            content=resp.content,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}_{format}.zip"},
        )


@app.get("/exports")
async def list_exports():
    """List all exported datasets."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['data_manager']}/registry/exports")
        return resp.json()


# ==================== Embedding Search Endpoints ====================

@app.post("/search/similar")
async def search_similar_objects(
    embedding: str = Form(...),
    top_k: int = Form(10),
    category: Optional[str] = Form(None),
    min_confidence: Optional[float] = Form(None),
):
    """Search similar objects by embedding vector.

    Args:
        embedding: JSON array of embedding values (768 dimensions for DINOv2)
        top_k: Number of results to return
        category: Filter by category name
        min_confidence: Minimum confidence threshold
    """
    import json

    try:
        embedding_list = json.loads(embedding)
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid embedding format. Expected JSON array."}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SERVICES['registry']}/objects/search/embedding",
            json={
                "embedding": embedding_list,
                "top_k": top_k,
                "category": category,
                "min_confidence": min_confidence,
            },
        )
        return resp.json()


@app.post("/search/similar_to_object")
async def search_similar_to_object(
    object_id: str = Form(...),
    top_k: int = Form(10),
    category: Optional[str] = Form(None),
):
    """Find objects similar to a given object.

    This endpoint retrieves the embedding for the specified object
    and searches for similar objects.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get object details first
        obj_resp = await client.get(f"{SERVICES['registry']}/objects/{object_id}")
        obj_data = obj_resp.json()

        if not obj_data.get("success"):
            return {"success": False, "error": "Object not found"}

        # Get embedding from ChromaDB via classification-agent
        # For now, return a placeholder - in production, store embeddings in objects table
        return {
            "success": False,
            "error": "Object embedding retrieval not yet implemented. Use /search/similar with embedding directly.",
            "object": obj_data.get("data"),
        }


# ==================== Quality Validation Endpoints ====================

@app.patch("/registry/objects/{object_id}")
async def update_object(object_id: str):
    """Update object fields (for quality validation).

    Accepts JSON body with fields:
    - is_validated: bool
    - validated_by: str
    - quality_score: float
    - is_occluded: bool
    - is_truncated: bool
    - is_difficult: bool
    """
    from fastapi import Request
    # This endpoint proxies to registry's PATCH endpoint
    # Note: This is a simplified version - in production, properly parse the request body

    async with httpx.AsyncClient(timeout=30.0) as client:
        # For now, just return info about the endpoint
        return {
            "success": False,
            "error": "Use direct PATCH to /registry/objects/{object_id} with JSON body",
            "example": {
                "is_validated": True,
                "validated_by": "reviewer_001",
                "quality_score": 0.95,
            }
        }


@app.post("/registry/objects/{object_id}/validate")
async def validate_object(
    object_id: str,
    validated_by: str = Form(...),
    quality_score: float = Form(1.0),
):
    """Mark an object as validated.

    Args:
        object_id: Object ID to validate
        validated_by: Reviewer identifier
        quality_score: Quality score (0.0 to 1.0)
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.patch(
            f"{SERVICES['registry']}/objects/{object_id}",
            json={
                "is_validated": True,
                "validated_by": validated_by,
                "quality_score": quality_score,
            },
        )
        return resp.json()


@app.post("/registry/objects/{object_id}/reject")
async def reject_object(
    object_id: str,
    reason: str = Form(None),
):
    """Reject and delete an object.

    Args:
        object_id: Object ID to reject
        reason: Rejection reason (logged but object is deleted)
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Delete the object
        resp = await client.delete(f"{SERVICES['registry']}/objects/{object_id}")
        result = resp.json()

        if result.get("success"):
            return {
                "success": True,
                "message": f"Object {object_id} rejected and deleted",
                "reason": reason,
            }
        return result


@app.get("/registry/objects/pending")
async def get_pending_objects(
    project_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
):
    """Get objects pending validation.

    Returns objects where is_validated = False.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {
            "is_validated": False,
            "limit": limit,
        }
        if project_id:
            params["project_id"] = project_id
        if category:
            params["category"] = category

        resp = await client.get(f"{SERVICES['registry']}/objects", params=params)
        return resp.json()


# ==================== Training Endpoints ====================

@app.post("/train/start")
async def start_training(
    project_id: str = Form(...),
    dataset_path: str = Form(...),
    model_size: str = Form("yolov8n"),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    image_size: int = Form(640),
    experiment_name: Optional[str] = Form(None),
):
    """Start a YOLO training job.

    Args:
        project_id: Project identifier
        dataset_path: Path to YOLO format dataset (with data.yaml)
        model_size: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        epochs: Number of training epochs
        batch_size: Training batch size
        image_size: Input image size
        experiment_name: MLflow experiment name
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SERVICES['training']}/train/start",
            data={
                "project_id": project_id,
                "dataset_path": dataset_path,
                "model_size": model_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": image_size,
                "experiment_name": experiment_name,
            },
        )
        return resp.json()


@app.post("/train/from-registry")
async def train_from_registry(
    project_id: str = Form(...),
    dataset_name: str = Form(...),
    model_size: str = Form("yolov8n"),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    image_size: int = Form(640),
    filter_categories: Optional[str] = Form(None),
    min_confidence: Optional[float] = Form(None),
    only_validated: bool = Form(False),
):
    """Start training from Object Registry dataset.

    This endpoint:
    1. Exports dataset from Registry
    2. Starts YOLO training on exported data

    Args:
        project_id: Project identifier
        dataset_name: Name for the exported dataset
        model_size: YOLO model size
        epochs: Training epochs
        batch_size: Batch size
        image_size: Image size
        filter_categories: Comma-separated category names
        min_confidence: Minimum confidence filter
        only_validated: Only use validated objects
    """
    import json

    categories = None
    if filter_categories:
        categories = [c.strip() for c in filter_categories.split(",")]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SERVICES['training']}/train/from-registry",
            json={
                "project_id": project_id,
                "dataset_name": dataset_name,
                "model_size": model_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": image_size,
                "filter_categories": categories,
                "min_confidence": min_confidence,
                "only_validated": only_validated,
            },
        )
        return resp.json()


@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Get status of a training job."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['training']}/train/status/{job_id}")
        return resp.json()


@app.get("/train/jobs")
async def list_training_jobs(
    status: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
):
    """List training jobs."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {"limit": limit}
        if status:
            params["status"] = status
        if project_id:
            params["project_id"] = project_id

        resp = await client.get(f"{SERVICES['training']}/train/jobs", params=params)
        return resp.json()


@app.post("/train/stop/{job_id}")
async def stop_training(job_id: str):
    """Stop a running training job."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{SERVICES['training']}/train/stop/{job_id}")
        return resp.json()


@app.get("/models")
async def list_models(project_id: Optional[str] = None):
    """List trained models."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {}
        if project_id:
            params["project_id"] = project_id

        resp = await client.get(f"{SERVICES['training']}/models", params=params)
        return resp.json()


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['training']}/models/{model_id}")
        return resp.json()


@app.post("/models/evaluate")
async def evaluate_model(
    model_path: str = Form(...),
    data_path: str = Form(...),
):
    """Evaluate a model on a dataset."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{SERVICES['training']}/models/evaluate",
            data={"model_path": model_path, "data_path": data_path},
        )
        return resp.json()


@app.post("/predict")
async def predict(
    model_path: str = Form(...),
    image_path: str = Form(...),
    confidence: float = Form(0.25),
    iou_threshold: float = Form(0.45),
):
    """Run inference using a trained model."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{SERVICES['training']}/predict",
            json={
                "model_path": model_path,
                "image_path": image_path,
                "confidence": confidence,
                "iou_threshold": iou_threshold,
            },
        )
        return resp.json()


@app.get("/experiments")
async def list_experiments():
    """List MLflow experiments."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['training']}/experiments")
        return resp.json()


@app.get("/experiments/{experiment_name}/runs")
async def get_experiment_runs(experiment_name: str):
    """Get runs for an MLflow experiment."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['training']}/experiments/{experiment_name}/runs")
        return resp.json()


# ==================== Evaluation Endpoints ====================

@app.post("/evaluate/detection")
async def evaluate_detection(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
    iou_threshold: float = Form(0.5),
):
    """Evaluate detection predictions against ground truth.

    Args:
        predictions: JSON string with predicted boxes [{boxes, labels, scores}, ...]
        ground_truth: JSON string with ground truth boxes [{boxes, labels}, ...]
        iou_threshold: IoU threshold for matching (default: 0.5)
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/evaluate/detection",
            data={
                "predictions": predictions,
                "ground_truth": ground_truth,
                "iou_threshold": str(iou_threshold),
            },
        )
        return resp.json()


@app.post("/evaluate/detection/coco")
async def evaluate_detection_coco(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
):
    """Evaluate detection with COCO-style mAP50-95.

    Calculates mAP at multiple IoU thresholds from 0.5 to 0.95.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/evaluate/detection/coco",
            data={
                "predictions": predictions,
                "ground_truth": ground_truth,
            },
        )
        return resp.json()


@app.post("/evaluate/classification")
async def evaluate_classification(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
):
    """Evaluate classification predictions.

    Args:
        predictions: JSON string with predictions [{class: str}, ...]
        ground_truth: JSON string with ground truth [{class: str}, ...]
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/evaluate/classification",
            data={
                "predictions": predictions,
                "ground_truth": ground_truth,
            },
        )
        return resp.json()


@app.post("/evaluate/segmentation")
async def evaluate_segmentation(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
):
    """Evaluate segmentation predictions.

    Args:
        predictions: JSON string with predictions [{mask: [[bool]]}, ...]
        ground_truth: JSON string with ground truth [{mask: [[bool]]}, ...]
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/evaluate/segmentation",
            data={
                "predictions": predictions,
                "ground_truth": ground_truth,
            },
        )
        return resp.json()


@app.post("/evaluate/batch")
async def evaluate_batch(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
    task: str = Form("detection"),
    iou_threshold: float = Form(0.5),
):
    """Evaluate a batch of predictions.

    Args:
        predictions: JSON array of predictions per image
        ground_truth: JSON array of ground truth per image
        task: "detection", "classification", or "segmentation"
        iou_threshold: IoU threshold for detection (ignored for other tasks)
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/evaluate/batch",
            data={
                "predictions": predictions,
                "ground_truth": ground_truth,
                "task": task,
                "iou_threshold": str(iou_threshold),
            },
        )
        return resp.json()


@app.post("/evaluate/model")
async def evaluate_model_on_dataset(
    model_path: str = Form(...),
    dataset_path: str = Form(...),
    task: str = Form("detection"),
):
    """Evaluate a trained model on a dataset.

    Args:
        model_path: Path to trained model (e.g., /models/best.pt)
        dataset_path: Path to dataset YAML file
        task: Task type (detection, classification, segmentation)
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/evaluate/model",
            data={
                "model_path": model_path,
                "dataset_path": dataset_path,
                "task": task,
            },
        )
        return resp.json()


@app.post("/evaluate/compare")
async def compare_models(
    model_paths: str = Form(...),
    dataset_path: str = Form(...),
    metrics: str = Form("mAP50,mAP50-95,precision,recall"),
):
    """Compare multiple models on the same dataset.

    Args:
        model_paths: JSON array of model paths
        dataset_path: Path to dataset YAML file
        metrics: Comma-separated list of metrics to compare
    """
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/compare/models",
            data={
                "model_paths": model_paths,
                "dataset_path": dataset_path,
                "metrics": metrics,
            },
        )
        return resp.json()


@app.get("/evaluate/metrics")
async def list_evaluation_metrics():
    """List all available evaluation metrics by task type."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['evaluation']}/metrics/list")
        return resp.json()


@app.get("/evaluations/{evaluation_id}")
async def get_evaluation(evaluation_id: str):
    """Get stored evaluation by ID."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SERVICES['evaluation']}/evaluations/{evaluation_id}")
        return resp.json()


@app.get("/evaluations/{evaluation_id}/report")
async def get_evaluation_report(
    evaluation_id: str,
    format: str = "json",
    include_visualizations: bool = False,
):
    """Get evaluation report.

    Args:
        evaluation_id: Evaluation ID
        format: Report format - "json" or "markdown"
        include_visualizations: Include visualization data for charts
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SERVICES['evaluation']}/evaluations/{evaluation_id}/report",
            params={
                "format": format,
                "include_visualizations": include_visualizations,
            },
        )
        return resp.json()


@app.post("/evaluations/{evaluation_id}/report/generate")
async def generate_evaluation_report(
    evaluation_id: str,
    format: str = Form("json"),
    include_visualizations: bool = Form(False),
):
    """Generate evaluation report (POST variant for forms).

    Args:
        evaluation_id: Evaluation ID
        format: Report format - "json" or "markdown"
        include_visualizations: Include visualization data
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SERVICES['evaluation']}/report/generate",
            data={
                "evaluation_id": evaluation_id,
                "format": format,
                "include_visualizations": include_visualizations,
            },
        )
        return resp.json()
