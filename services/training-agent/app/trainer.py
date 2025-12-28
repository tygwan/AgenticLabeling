"""YOLO training implementation."""
import base64
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .schemas import TrainingConfig, InferenceRequest, InferenceResult


class TrainingCallback:
    """Callback for training progress updates."""

    def __init__(self, progress_callback: Optional[Callable] = None, total_epochs: int = 100):
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.metrics = {}

    def on_train_epoch_end(self, trainer):
        """Called at end of each training epoch."""
        self.current_epoch = trainer.epoch + 1
        self.metrics = {
            "box_loss": float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
            "cls_loss": float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
            "dfl_loss": float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0,
        }
        if self.progress_callback:
            self.progress_callback(self.current_epoch, self.total_epochs, self.metrics)

    def on_val_end(self, trainer):
        """Called at end of validation."""
        if hasattr(trainer, 'metrics'):
            self.metrics.update({
                "mAP50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(trainer.metrics.get("metrics/mAP50-95(B)", 0)),
            })


class YOLOTrainer:
    """YOLO model trainer with MLflow integration."""

    def __init__(self, models_dir: str = "/data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._stop_requested = False
        self._loaded_models: Dict[str, Any] = {}

    def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Train YOLO model.

        Args:
            config: Training configuration
            progress_callback: Callback for progress updates

        Returns:
            Training results including model path and metrics
        """
        from ultralytics import YOLO
        import mlflow

        self._stop_requested = False

        # Setup MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        experiment_name = config.experiment_name or f"{config.project_id}_training"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{config.project_id}_{config.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "model_size": config.model_size,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "dataset": config.dataset_path,
                "augment": config.augment,
                "patience": config.patience,
            })

            # Initialize model
            if config.pretrained_weights and Path(config.pretrained_weights).exists():
                model = YOLO(config.pretrained_weights)
            else:
                model = YOLO(f"{config.model_size}.pt")

            # Setup training callback
            callback = TrainingCallback(progress_callback, config.epochs)
            model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)
            model.add_callback("on_val_end", callback.on_val_end)

            # Train
            results = model.train(
                data=config.dataset_path,
                epochs=config.epochs,
                batch=config.batch_size,
                imgsz=config.image_size,
                project=str(self.models_dir / config.project_id),
                name=experiment_name,
                exist_ok=True,
                device=config.device,
                augment=config.augment,
                patience=config.patience,
                verbose=True,
            )

            # Get best model path
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            last_model = Path(results.save_dir) / "weights" / "last.pt"

            # Extract metrics
            metrics = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            }
            mlflow.log_metrics(metrics)

            # Log model artifacts
            if best_model.exists():
                mlflow.log_artifact(str(best_model))
            if last_model.exists():
                mlflow.log_artifact(str(last_model))

            # Log training curves
            results_csv = Path(results.save_dir) / "results.csv"
            if results_csv.exists():
                mlflow.log_artifact(str(results_csv))

            # Generate model ID
            model_id = f"model_{uuid.uuid4().hex[:12]}"

            # Save model metadata
            metadata = {
                "model_id": model_id,
                "project_id": config.project_id,
                "experiment_name": experiment_name,
                "model_size": config.model_size,
                "metrics": metrics,
                "created_at": datetime.utcnow().isoformat(),
                "config": config.model_dump(),
                "mlflow_run_id": mlflow.active_run().info.run_id,
            }
            metadata_path = Path(results.save_dir) / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return {
                "model_id": model_id,
                "model_path": str(best_model),
                "last_model_path": str(last_model),
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id,
                "save_dir": str(results.save_dir),
            }

    def stop(self):
        """Request training stop."""
        self._stop_requested = True

    def predict(self, request: InferenceRequest) -> List[InferenceResult]:
        """Run inference on image."""
        from ultralytics import YOLO
        import cv2

        # Load model (with caching)
        if request.model_path not in self._loaded_models:
            self._loaded_models[request.model_path] = YOLO(request.model_path)
        model = self._loaded_models[request.model_path]

        # Load image
        if request.image_path:
            image = cv2.imread(request.image_path)
        elif request.image_base64:
            image_bytes = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Either image_path or image_base64 must be provided")

        # Run inference
        results = model.predict(
            image,
            conf=request.confidence,
            iou=request.iou_threshold,
            verbose=False,
        )[0]

        # Convert to results
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(InferenceResult(
                class_id=int(box.cls[0]),
                class_name=model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=[x1, y1, x2 - x1, y2 - y1],
            ))

        return detections

    def list_models(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List trained models with metadata."""
        models = []
        search_dir = self.models_dir

        if project_id:
            search_dir = search_dir / project_id

        if not search_dir.exists():
            return models

        for model_file in search_dir.rglob("best.pt"):
            model_info = {
                "path": str(model_file),
                "project": model_file.parent.parent.parent.name,
                "experiment": model_file.parent.parent.name,
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            }

            # Load metadata if exists
            metadata_path = model_file.parent.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    model_info.update({
                        "model_id": metadata.get("model_id"),
                        "metrics": metadata.get("metrics", {}),
                        "model_size": metadata.get("model_size"),
                    })

            models.append(model_info)

        return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        for model_file in self.models_dir.rglob("best.pt"):
            metadata_path = model_file.parent.parent / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    if metadata.get("model_id") == model_id:
                        return {
                            "path": str(model_file),
                            **metadata,
                            "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                        }
        return None

    def export_model(self, model_path: str, format: str = "onnx") -> str:
        """Export model to different format."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        export_path = model.export(format=format)
        return str(export_path)

    def evaluate_model(self, model_path: str, data_path: str) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        results = model.val(data=data_path, verbose=False)

        return {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            "speed": {
                "preprocess": results.speed.get("preprocess", 0),
                "inference": results.speed.get("inference", 0),
                "postprocess": results.speed.get("postprocess", 0),
            },
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List MLflow experiments."""
        import mlflow

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()

            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
                for exp in experiments
            ]
        except Exception:
            return []

    def get_experiment_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Get runs for an experiment."""
        import mlflow

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                return []

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=50,
            )

            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }
                for run in runs
            ]
        except Exception:
            return []
