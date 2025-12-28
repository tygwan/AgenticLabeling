"""Pydantic schemas for detection service."""
from typing import List, Optional

from pydantic import BaseModel


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionData(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    confidence: List[float]
    image_size: ImageSize


class DetectionRequest(BaseModel):
    classes: List[str]
    confidence: float = 0.5


class DetectionResponse(BaseModel):
    success: bool
    data: Optional[DetectionData] = None
    message: Optional[str] = None
    error: Optional[str] = None
