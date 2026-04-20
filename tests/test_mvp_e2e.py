"""End-to-end smoke tests for the monolith MVP app."""

from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def test_mvp_image_review_and_export(monkeypatch, tmp_path):
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

    sample_image = Path("data/images/test_street.jpg")
    assert sample_image.exists()

    with TestClient(mvp_app.main.app) as client:
        upload = client.post(
            "/api/pipeline/auto-label",
            files={"image": (sample_image.name, sample_image.read_bytes(), "image/jpeg")},
            data={"project_id": "e2e-project", "classes": "car,person"},
        )
        assert upload.status_code == 200
        upload_payload = upload.json()
        assert upload_payload["success"] is True
        assert upload_payload["detections"] == 1
        source_id = upload_payload["source_id"]
        object_id = upload_payload["object_ids"][0]

        sources = client.get("/api/review/sources")
        assert sources.status_code == 200
        assert len(sources.json()["data"]) == 1

        objects = client.get(f"/api/review/objects?source_id={source_id}")
        assert objects.status_code == 200
        assert len(objects.json()["data"]) == 1
        assert objects.json()["data"][0]["object_id"] == object_id

        mask = client.get(f"/api/masks/{object_id}")
        assert mask.status_code == 200
        assert mask.headers["content-type"] == "image/png"

        overlay = client.get(f"/api/assets/{source_id}/overlay")
        assert overlay.status_code == 200
        assert overlay.headers["content-type"] == "image/png"
        assert overlay.content.startswith(b"\x89PNG")

        bbox_overlay = client.get(f"/api/assets/{source_id}/bbox-overlay")
        assert bbox_overlay.status_code == 200
        assert bbox_overlay.headers["content-type"] == "image/png"
        assert bbox_overlay.content.startswith(b"\x89PNG")

        approve = client.patch(f"/api/review/objects/{object_id}")
        assert approve.status_code == 200

        export = client.post(
            "/api/export",
            data={"dataset_name": "e2e-dataset", "export_format": "yolo", "only_validated": "true"},
        )
        assert export.status_code == 200
        export_payload = export.json()
        assert export_payload["success"] is True
        assert export_payload["image_count"] == 1
        assert export_payload["object_count"] == 1

        review = client.get(f"/review?source_id={source_id}")
        assert review.status_code == 200
        assert "BBox Overlay" in review.text
        assert "Segmentation Overlay" in review.text

        export_zip = tmp_path / "data" / "exports" / "e2e-dataset.zip"
        assert export_zip.exists()
