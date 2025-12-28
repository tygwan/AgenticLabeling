"""Pytest configuration and fixtures."""
import os
import tempfile
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="agenticlabeling_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_bbox():
    """Sample bounding box [x, y, w, h]."""
    return [100.0, 150.0, 200.0, 180.0]


@pytest.fixture
def sample_objects_data():
    """Sample objects data for batch registration."""
    return [
        {
            "category": "person",
            "bbox": [100, 100, 50, 80],
            "confidence": 0.95,
            "detection_model": "florence2",
        },
        {
            "category": "car",
            "bbox": [200, 150, 120, 80],
            "confidence": 0.88,
            "detection_model": "florence2",
        },
        {
            "category": "person",
            "bbox": [350, 100, 45, 75],
            "confidence": 0.92,
            "detection_model": "florence2",
        },
    ]
