"""Pydantic schemas for training service."""
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


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


class TrainingResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
