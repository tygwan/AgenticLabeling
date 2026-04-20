"""Smoke tests for the monolith MVP app."""

from __future__ import annotations

import importlib
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image


def _make_test_image() -> bytes:
    image = Image.new("RGB", (64, 64), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_mvp_home_and_upload(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_MODELS", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "data" / "sqlite" / "mvp.db"))
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path / "data" / "assets"))
    monkeypatch.setenv("MASKS_DIR", str(tmp_path / "data" / "masks"))
    monkeypatch.setenv("EXPORTS_DIR", str(tmp_path / "data" / "exports"))

    import mvp_app.config
    import mvp_app.main

    importlib.reload(mvp_app.config)
    importlib.reload(mvp_app.main)

    with TestClient(mvp_app.main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        health_payload = health.json()
        assert health_payload["status"] == "healthy"
        assert health_payload["app"] == "AgenticLabeling"
        assert "stats" in health_payload
        assert "segmentation_backend" in health_payload

        home = client.get("/")
        assert home.status_code == 200
        assert "AgenticLabeling" in home.text

        upload = client.post(
            "/api/pipeline/auto-label",
            files={"image": ("sample.png", _make_test_image(), "image/png")},
            data={"project_id": "test-project", "classes": "person,car"},
        )
        assert upload.status_code == 200
        payload = upload.json()
        assert payload["success"] is True
        assert payload["detections"] == 1
        assert "source_id" in payload
        assert "object_ids" in payload
        assert "segmentation_backend" in payload

        review = client.get("/review")
        assert review.status_code == 200
        assert "Review Workspace" in review.text
        assert "BBox Overlay" in review.text
        assert "Segmentation Overlay" in review.text
