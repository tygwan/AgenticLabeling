"""Pydantic schemas for labeling service."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExportFormat(str, Enum):
    COCO = "coco"
    YOLO = "yolo"
    VOC = "voc"


class LabelData(BaseModel):
    image_id: str
    boxes: Optional[List[List[float]]] = None
    masks: Optional[List[str]] = None  # Base64 encoded
    classes: List[str]
    confidence: Optional[List[float]] = None


class LabelResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ExportResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    download_url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
