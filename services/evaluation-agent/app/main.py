"""Evaluation Agent Service - Model evaluation and metrics."""
import json
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .evaluator import ModelEvaluator
from .schemas import EvaluationResponse, MetricsResponse


app = FastAPI(
    title="Evaluation Agent",
    description="Model evaluation, metrics calculation, and comparison service",
    version="0.1.0",
)

evaluator = ModelEvaluator()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "evaluation-agent"}


@app.post("/evaluate/detection")
async def evaluate_detection(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
    iou_threshold: float = Form(0.5),
):
    """Evaluate detection predictions against ground truth.

    Args:
        predictions: JSON string with predicted boxes
        ground_truth: JSON string with ground truth boxes
        iou_threshold: IoU threshold for matching
    """
    try:
        preds = json.loads(predictions)
        gt = json.loads(ground_truth)

        metrics = evaluator.evaluate_detection(preds, gt, iou_threshold)

        return EvaluationResponse(
            success=True,
            metrics=metrics,
            message="Evaluation completed",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/evaluate/classification")
async def evaluate_classification(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
):
    """Evaluate classification predictions."""
    try:
        preds = json.loads(predictions)
        gt = json.loads(ground_truth)

        metrics = evaluator.evaluate_classification(preds, gt)

        return EvaluationResponse(
            success=True,
            metrics=metrics,
            message="Classification evaluation completed",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/evaluate/segmentation")
async def evaluate_segmentation(
    predictions: str = Form(...),
    ground_truth: str = Form(...),
):
    """Evaluate segmentation predictions."""
    try:
        preds = json.loads(predictions)
        gt = json.loads(ground_truth)

        metrics = evaluator.evaluate_segmentation(preds, gt)

        return EvaluationResponse(
            success=True,
            metrics=metrics,
            message="Segmentation evaluation completed",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/evaluate/model")
async def evaluate_model(
    model_path: str = Form(...),
    dataset_path: str = Form(...),
    task: str = Form("detection"),
):
    """Evaluate a trained model on a dataset."""
    try:
        metrics = evaluator.evaluate_model(model_path, dataset_path, task)

        return EvaluationResponse(
            success=True,
            metrics=metrics,
            message=f"Model evaluation completed for {task}",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/compare/models")
async def compare_models(
    model_paths: str = Form(...),
    dataset_path: str = Form(...),
    metrics: str = Form("mAP50,mAP50-95,precision,recall"),
):
    """Compare multiple models on the same dataset."""
    try:
        paths = json.loads(model_paths)
        metric_list = [m.strip() for m in metrics.split(",")]

        comparison = evaluator.compare_models(paths, dataset_path, metric_list)

        return {
            "success": True,
            "comparison": comparison,
            "best_model": comparison.get("best"),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/report/generate")
async def generate_report(
    evaluation_id: str = Form(...),
    format: str = Form("json"),
    include_visualizations: bool = Form(False),
):
    """Generate evaluation report."""
    try:
        report = evaluator.generate_report(
            evaluation_id, format, include_visualizations
        )

        return {
            "success": True,
            "report": report,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/metrics/list")
async def list_available_metrics():
    """List all available evaluation metrics."""
    return {
        "detection": [
            "mAP50",
            "mAP50-95",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
        ],
        "classification": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
            "per_class_accuracy",
        ],
        "segmentation": [
            "mIoU",
            "pixel_accuracy",
            "dice_coefficient",
            "boundary_f1",
        ],
    }
