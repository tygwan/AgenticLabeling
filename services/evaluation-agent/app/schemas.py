"""Pydantic schemas for evaluation service."""
from typing import Any, Dict, Optional

from pydantic import BaseModel


class EvaluationResponse(BaseModel):
    success: bool
    metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    detection: list
    classification: list
    segmentation: list
