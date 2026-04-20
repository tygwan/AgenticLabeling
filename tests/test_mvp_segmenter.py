"""Segmenter regression tests for the MVP app."""

from __future__ import annotations

import base64
import importlib
import io

from PIL import Image


def _make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_segmenter_falls_back_when_sam3_is_unavailable(monkeypatch):
    monkeypatch.setenv("FAKE_MODELS", "0")

    import mvp_app.config
    import mvp_app.segmenter

    importlib.reload(mvp_app.config)
    importlib.reload(mvp_app.segmenter)

    service = mvp_app.segmenter.SegmentationService()
    service._processor = None

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sam3"):
            raise ModuleNotFoundError("No module named 'sam3'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result = service.segment(_make_test_image_bytes(), [[4, 5, 20, 18]])

    assert len(result["masks"]) == 1
    assert service._processor is False
    decoded = base64.b64decode(result["masks"][0]["mask"])
    mask = Image.open(io.BytesIO(decoded))
    assert mask.size == (32, 24)
