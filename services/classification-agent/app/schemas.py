"""Pydantic schemas for classification service."""
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    similarity: float


class ClassificationData(BaseModel):
    class_name: str
    predicted_class: str
    similarity: float
    margin: float
    confidence_level: str
    top_predictions: List[Dict[str, Any]]
    all_scores: Dict[str, float]
    status: str


class ClassificationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class SupportSetResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    classes: Optional[List[str]] = None
    images_per_class: Optional[Dict[str, int]] = None
    error: Optional[str] = None
