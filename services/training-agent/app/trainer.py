"""YOLO training implementation."""
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .schemas import TrainingConfig


class YOLOTrainer:
    """YOLO model trainer with MLflow integration."""

    def __init__(self, models_dir: str = "/data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._stop_requested = False

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
        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run(run_name=f"{config.project_id}_{config.model_size}"):
            # Log parameters
            mlflow.log_params({
                "model_size": config.model_size,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "dataset": config.dataset_path,
            })

            # Initialize model
            model = YOLO(f"{config.model_size}.pt")

            # Train
            results = model.train(
                data=config.dataset_path,
                epochs=config.epochs,
                batch=config.batch_size,
                imgsz=config.image_size,
                project=str(self.models_dir / config.project_id),
                name=config.experiment_name,
                exist_ok=True,
            )

            # Get best model path
            best_model = Path(results.save_dir) / "weights" / "best.pt"

            # Log metrics
            metrics = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            }
            mlflow.log_metrics(metrics)

            # Log model artifact
            if best_model.exists():
                mlflow.log_artifact(str(best_model))

            return {
                "model_path": str(best_model),
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

    def stop(self):
        """Request training stop."""
        self._stop_requested = True

    def list_models(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List trained models."""
        models = []
        search_dir = self.models_dir

        if project_id:
            search_dir = search_dir / project_id

        for model_file in search_dir.rglob("best.pt"):
            models.append({
                "path": str(model_file),
                "project": model_file.parent.parent.parent.name,
                "experiment": model_file.parent.parent.name,
                "size_mb": model_file.stat().st_size / (1024 * 1024),
            })

        return models

    def export_model(self, model_path: str, format: str = "onnx") -> str:
        """Export model to different format."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        export_path = model.export(format=format)
        return str(export_path)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List MLflow experiments."""
        import mlflow

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
