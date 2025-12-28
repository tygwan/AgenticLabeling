"""Pydantic schemas for training service."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingConfig(BaseModel):
    project_id: str
    dataset_path: str
    model_size: str = "yolov8n"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    experiment_name: Optional[str] = None
    pretrained_weights: Optional[str] = None
    augment: bool = True
    patience: int = 50
    device: str = "0"


class RegistryTrainingRequest(BaseModel):
    """Request to train from Registry dataset."""
    project_id: str
    dataset_name: str
    format: str = "yolo"
    model_size: str = "yolov8n"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    filter_categories: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    only_validated: bool = False
    split_config: Optional[Dict[str, float]] = Field(
        default={"train": 0.8, "val": 0.1, "test": 0.1}
    )


class TrainingResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class InferenceRequest(BaseModel):
    """Request for model inference."""
    model_path: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    confidence: float = 0.25
    iou_threshold: float = 0.45


class InferenceResult(BaseModel):
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x, y, w, h]


class InferenceResponse(BaseModel):
    success: bool
    detections: Optional[List[InferenceResult]] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    path: str
    project_id: str
    experiment_name: str
    metrics: Dict[str, float]
    created_at: str
    size_mb: float
