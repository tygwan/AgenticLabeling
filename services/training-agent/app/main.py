"""Training Agent Service - YOLO model training and management."""
import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .trainer import YOLOTrainer
from .schemas import TrainingConfig, TrainingStatus, TrainingResponse


app = FastAPI(
    title="Training Agent",
    description="YOLO model training and experiment tracking service",
    version="0.1.0",
)

# Store active training jobs
training_jobs: Dict[str, Dict] = {}
trainer = YOLOTrainer()


@app.get("/health")
async def health():
    """Health check endpoint."""
    active_jobs = len([j for j in training_jobs.values() if j["status"] == "running"])
    return {"status": "healthy", "active_jobs": active_jobs}


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
):
    """Start a new training job.

    Args:
        project_id: Project identifier
        dataset_path: Path to dataset (YOLO format)
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Training batch size
        image_size: Input image size
        experiment_name: MLflow experiment name
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
async def list_training_jobs(status: str = None, limit: int = 20):
    """List all training jobs."""
    jobs = list(training_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

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


@app.get("/models")
async def list_models(project_id: str = None):
    """List trained models."""
    try:
        models = trainer.list_models(project_id)
        return {"success": True, "models": models}
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
    """Export model to different format."""
    try:
        result = trainer.export_model(model_path, format)
        return {"success": True, "exported_path": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


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
