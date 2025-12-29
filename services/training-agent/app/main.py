"""Training Agent Service - YOLO model training and management."""
import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse

from .trainer import YOLOTrainer
from .schemas import (
    TrainingConfig,
    TrainingStatus,
    TrainingResponse,
    RegistryTrainingRequest,
    InferenceRequest,
    InferenceResponse,
    InferenceResult,
)
from .active_learning import ActiveLearner, SamplingStrategy


app = FastAPI(
    title="Training Agent",
    description="YOLO model training and experiment tracking service",
    version="0.2.0",
)

# Configuration
DATA_MANAGER_URL = os.getenv("DATA_MANAGER_URL", "http://data-manager:8006")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://object-registry:8010")

# Store active training jobs
training_jobs: Dict[str, Dict] = {}
trainer = YOLOTrainer()
active_learner = ActiveLearner()


@app.get("/health")
async def health():
    """Health check endpoint."""
    active_jobs = len([j for j in training_jobs.values() if j["status"] == "running"])
    completed_jobs = len([j for j in training_jobs.values() if j["status"] == "completed"])
    return {
        "status": "healthy",
        "active_jobs": active_jobs,
        "completed_jobs": completed_jobs,
        "total_jobs": len(training_jobs),
    }


@app.post("/train/start", response_model=TrainingResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    project_id: str = Form(...),
    dataset_path: str = Form(...),
    model_size: str = Form("yolov8n"),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    image_size: int = Form(640),
    experiment_name: str = Form(None),
    pretrained_weights: str = Form(None),
    augment: bool = Form(True),
    patience: int = Form(50),
):
    """Start a new training job.

    Args:
        project_id: Project identifier
        dataset_path: Path to dataset (YOLO format with data.yaml)
        model_size: YOLOv8 model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Training batch size
        image_size: Input image size
        experiment_name: MLflow experiment name
        pretrained_weights: Path to pretrained weights
        augment: Whether to use augmentation
        patience: Early stopping patience
    """
    try:
        job_id = str(uuid.uuid4())[:8]

        config = TrainingConfig(
            project_id=project_id,
            dataset_path=dataset_path,
            model_size=model_size,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            experiment_name=experiment_name or f"{project_id}_{job_id}",
            pretrained_weights=pretrained_weights,
            augment=augment,
            patience=patience,
        )

        training_jobs[job_id] = {
            "job_id": job_id,
            "config": config.model_dump(),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "metrics": {},
        }

        # Start training in background
        background_tasks.add_task(run_training, job_id, config)

        return TrainingResponse(
            success=True,
            job_id=job_id,
            message="Training job started",
            status="pending",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/train/from-registry", response_model=TrainingResponse)
async def train_from_registry(
    background_tasks: BackgroundTasks,
    request: RegistryTrainingRequest = Body(...),
):
    """Start training from Object Registry dataset.

    This endpoint:
    1. Exports dataset from Registry via Data Manager
    2. Starts training on the exported dataset
    """
    try:
        job_id = str(uuid.uuid4())[:8]

        training_jobs[job_id] = {
            "job_id": job_id,
            "config": request.model_dump(),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "metrics": {},
            "phase": "exporting",
        }

        # Start export and training in background
        background_tasks.add_task(run_registry_training, job_id, request)

        return TrainingResponse(
            success=True,
            job_id=job_id,
            message="Training job started (exporting dataset from registry)",
            status="pending",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


async def run_registry_training(job_id: str, request: RegistryTrainingRequest):
    """Export dataset from registry and run training."""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["phase"] = "exporting"

        # Build filter config
        filter_config = {}
        if request.filter_categories:
            filter_config["categories"] = request.filter_categories
        if request.min_confidence:
            filter_config["min_confidence"] = request.min_confidence
        if request.only_validated:
            filter_config["is_validated"] = True

        # Export dataset via Data Manager
        async with httpx.AsyncClient(timeout=300.0) as client:
            export_response = await client.post(
                f"{DATA_MANAGER_URL}/registry/export",
                json={
                    "name": request.dataset_name,
                    "format": request.format,
                    "filter_config": filter_config,
                    "split_config": request.split_config,
                },
            )

            if export_response.status_code != 200:
                raise Exception(f"Export failed: {export_response.text}")

            export_result = export_response.json()
            dataset_path = export_result.get("output_path")

            if not dataset_path:
                raise Exception("No dataset path returned from export")

        training_jobs[job_id]["phase"] = "training"
        training_jobs[job_id]["dataset_path"] = dataset_path

        # Create training config
        config = TrainingConfig(
            project_id=request.project_id,
            dataset_path=f"{dataset_path}/data.yaml",
            model_size=request.model_size,
            epochs=request.epochs,
            batch_size=request.batch_size,
            image_size=request.image_size,
            experiment_name=f"{request.project_id}_{request.dataset_name}",
        )

        # Run training
        def progress_callback(epoch, total_epochs, metrics):
            training_jobs[job_id]["progress"] = (epoch / total_epochs) * 100
            training_jobs[job_id]["metrics"] = metrics
            training_jobs[job_id]["current_epoch"] = epoch

        result = await asyncio.to_thread(
            trainer.train,
            config=config,
            progress_callback=progress_callback,
        )

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["result"] = result
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


async def run_training(job_id: str, config: TrainingConfig):
    """Run training in background."""
    try:
        training_jobs[job_id]["status"] = "running"

        def progress_callback(epoch, total_epochs, metrics):
            training_jobs[job_id]["progress"] = (epoch / total_epochs) * 100
            training_jobs[job_id]["metrics"] = metrics
            training_jobs[job_id]["current_epoch"] = epoch

        result = await asyncio.to_thread(
            trainer.train,
            config=config,
            progress_callback=progress_callback,
        )

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["result"] = result
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Get status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return training_jobs[job_id]


@app.get("/train/jobs")
async def list_training_jobs(
    status: str = None,
    project_id: str = None,
    limit: int = 20,
):
    """List all training jobs."""
    jobs = list(training_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    if project_id:
        jobs = [j for j in jobs if j["config"].get("project_id") == project_id]

    jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)[:limit]
    return {"jobs": jobs, "total": len(jobs)}


@app.post("/train/stop/{job_id}")
async def stop_training(job_id: str):
    """Stop a running training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if training_jobs[job_id]["status"] != "running":
        return {"success": False, "message": "Job is not running"}

    trainer.stop()
    training_jobs[job_id]["status"] = "stopped"
    return {"success": True, "message": "Training stopped"}


@app.delete("/train/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job from history."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if training_jobs[job_id]["status"] == "running":
        return {"success": False, "message": "Cannot delete running job"}

    del training_jobs[job_id]
    return {"success": True, "message": "Job deleted"}


# ==================== Model Management ====================

@app.get("/models")
async def list_models(project_id: str = None):
    """List trained models."""
    try:
        models = trainer.list_models(project_id)
        return {"success": True, "models": models, "count": len(models)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details by ID."""
    try:
        model = trainer.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"success": True, "model": model}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/models/export")
async def export_model(
    model_path: str = Form(...),
    format: str = Form("onnx"),
):
    """Export model to different format.

    Supported formats: onnx, torchscript, openvino, engine (TensorRT)
    """
    try:
        result = trainer.export_model(model_path, format)
        return {"success": True, "exported_path": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/models/evaluate")
async def evaluate_model(
    model_path: str = Form(...),
    data_path: str = Form(...),
):
    """Evaluate model on dataset."""
    try:
        result = await asyncio.to_thread(
            trainer.evaluate_model, model_path, data_path
        )
        return {"success": True, "metrics": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


# ==================== Inference ====================

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Run inference on an image."""
    try:
        start_time = time.time()

        detections = await asyncio.to_thread(trainer.predict, request)

        inference_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            success=True,
            detections=detections,
            inference_time_ms=round(inference_time, 2),
        )
    except Exception as e:
        return InferenceResponse(
            success=False,
            error=str(e),
        )


@app.post("/predict/batch")
async def predict_batch(
    model_path: str = Form(...),
    image_paths: str = Form(...),  # JSON array of paths
    confidence: float = Form(0.25),
    iou_threshold: float = Form(0.45),
):
    """Run batch inference on multiple images."""
    try:
        paths = json.loads(image_paths)
        results = []

        for path in paths:
            request = InferenceRequest(
                model_path=model_path,
                image_path=path,
                confidence=confidence,
                iou_threshold=iou_threshold,
            )
            detections = await asyncio.to_thread(trainer.predict, request)
            results.append({
                "image_path": path,
                "detections": [d.model_dump() for d in detections],
            })

        return {"success": True, "results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


# ==================== MLflow Experiments ====================

@app.get("/experiments")
async def list_experiments():
    """List MLflow experiments."""
    try:
        experiments = trainer.list_experiments()
        return {"success": True, "experiments": experiments}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/experiments/{experiment_name}/runs")
async def get_experiment_runs(experiment_name: str):
    """Get runs for an experiment."""
    try:
        runs = trainer.get_experiment_runs(experiment_name)
        return {"success": True, "runs": runs}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


# ==================== Comparison ====================

@app.post("/models/compare")
async def compare_models(
    model_paths: str = Form(...),  # JSON array of paths
    data_path: str = Form(...),
):
    """Compare multiple models on the same dataset."""
    try:
        paths = json.loads(model_paths)
        results = []

        for path in paths:
            metrics = await asyncio.to_thread(
                trainer.evaluate_model, path, data_path
            )
            results.append({
                "model_path": path,
                "metrics": metrics,
            })

        # Sort by mAP50-95
        results = sorted(
            results,
            key=lambda x: x["metrics"].get("mAP50-95", 0),
            reverse=True,
        )

        return {"success": True, "comparison": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


# ==================== Active Learning ====================

@app.get("/active-learning/status")
async def get_active_learning_status():
    """Get active learning status and labeling statistics."""
    try:
        stats = await active_learner.get_labeling_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/active-learning/query")
async def query_uncertain_samples(
    strategy: str = Form("uncertainty"),
    n_samples: int = Form(10),
    min_confidence: float = Form(0.0),
    max_confidence: float = Form(0.7),
    category: str = Form(None),
):
    """Query uncertain samples for labeling.

    Strategies:
    - uncertainty: Select samples with lowest confidence
    - entropy: Select samples with highest prediction entropy
    - margin: Select samples with smallest margin between top predictions
    - random: Random sampling (baseline)
    - diversity: Embedding-based diversity sampling

    Args:
        strategy: Sampling strategy
        n_samples: Number of samples to return
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence (filter out high-confidence samples)
        category: Filter by specific category
    """
    try:
        # Map string to enum
        strategy_map = {
            "uncertainty": SamplingStrategy.UNCERTAINTY,
            "entropy": SamplingStrategy.ENTROPY,
            "margin": SamplingStrategy.MARGIN,
            "random": SamplingStrategy.RANDOM,
            "diversity": SamplingStrategy.DIVERSITY,
        }

        sampling_strategy = strategy_map.get(
            strategy.lower(),
            SamplingStrategy.UNCERTAINTY,
        )

        samples = await active_learner.query_uncertain_samples(
            strategy=sampling_strategy,
            n_samples=n_samples,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            exclude_validated=True,
            category=category,
        )

        return {
            "success": True,
            "strategy": strategy,
            "count": len(samples),
            "samples": [
                {
                    "object_id": s.object_id,
                    "source_id": s.source_id,
                    "image_path": s.image_path,
                    "confidence": s.confidence,
                    "entropy": s.entropy,
                    "predicted_category": s.predicted_category,
                    "bbox": s.bbox,
                    "uncertainty_score": round(s.uncertainty_score, 4),
                }
                for s in samples
            ],
            "next_action": "Review and validate these samples in label-studio-lite",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/active-learning/cycle")
async def run_active_learning_cycle(
    background_tasks: BackgroundTasks,
    project_id: str = Form(...),
    model_path: str = Form(None),
    strategy: str = Form("uncertainty"),
    n_query: int = Form(10),
    max_confidence: float = Form(0.7),
):
    """Run one cycle of active learning.

    This endpoint:
    1. Queries uncertain samples using the specified strategy
    2. Returns them for human labeling
    3. After validation, call /active-learning/retrain to retrain

    Args:
        project_id: Project identifier
        model_path: Path to current model (optional)
        strategy: Sampling strategy
        n_query: Number of samples to query
        max_confidence: Maximum confidence threshold for uncertain samples
    """
    try:
        strategy_map = {
            "uncertainty": SamplingStrategy.UNCERTAINTY,
            "entropy": SamplingStrategy.ENTROPY,
            "margin": SamplingStrategy.MARGIN,
            "random": SamplingStrategy.RANDOM,
            "diversity": SamplingStrategy.DIVERSITY,
        }

        sampling_strategy = strategy_map.get(
            strategy.lower(),
            SamplingStrategy.UNCERTAINTY,
        )

        result = await active_learner.run_active_learning_cycle(
            model_path=model_path or "yolov8n.pt",
            project_id=project_id,
            strategy=sampling_strategy,
            n_query=n_query,
            max_confidence=max_confidence,
        )

        return {"success": True, "data": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/active-learning/retrain")
async def retrain_after_labeling(
    background_tasks: BackgroundTasks,
    project_id: str = Form(...),
    model_size: str = Form("yolov8n"),
    epochs: int = Form(50),
    only_validated: bool = Form(True),
):
    """Retrain model after active learning labeling cycle.

    This triggers a new training run using only validated samples.

    Args:
        project_id: Project identifier
        model_size: YOLOv8 model size
        epochs: Number of training epochs
        only_validated: Only use validated samples for training
    """
    try:
        from .schemas import RegistryTrainingRequest

        request = RegistryTrainingRequest(
            project_id=project_id,
            dataset_name=f"{project_id}_al_cycle_{int(time.time())}",
            model_size=model_size,
            epochs=epochs,
            only_validated=only_validated,
        )

        job_id = str(uuid.uuid4())[:8]

        training_jobs[job_id] = {
            "job_id": job_id,
            "config": request.model_dump(),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "metrics": {},
            "phase": "exporting",
            "active_learning": True,
        }

        background_tasks.add_task(run_registry_training, job_id, request)

        return {
            "success": True,
            "job_id": job_id,
            "message": "Active learning retrain started",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/active-learning/estimate")
async def estimate_performance_gain(
    additional_labels: int = 50,
):
    """Estimate model performance gain from additional labels.

    Args:
        additional_labels: Number of additional labels to estimate for
    """
    try:
        stats = await active_learner.get_labeling_stats()
        current = stats.get("validated_objects", 0)
        target = current + additional_labels

        estimate = await active_learner.estimate_model_performance_gain(
            current_validated=current,
            target_validated=target,
        )

        return {"success": True, "data": estimate}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
