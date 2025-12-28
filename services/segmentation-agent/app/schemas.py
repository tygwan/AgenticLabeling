"""Pydantic schemas for segmentation service."""
from typing import List, Optional

from pydantic import BaseModel


class ImageSize(BaseModel):
    width: int
    height: int


class MaskData(BaseModel):
    mask: str  # Base64 encoded PNG
    score: float


class SegmentationData(BaseModel):
    masks: List[MaskData]
    image_size: ImageSize


class SegmentationResponse(BaseModel):
    success: bool
    data: Optional[SegmentationData] = None
    message: Optional[str] = None
    error: Optional[str] = None
