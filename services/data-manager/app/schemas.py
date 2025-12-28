"""Pydantic schemas for data manager service."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DatasetInfo(BaseModel):
    name: str
    description: Optional[str] = None
    classes: List[str]
    image_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0


class DatasetResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
