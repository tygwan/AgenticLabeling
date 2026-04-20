"""Detector regression tests for the MVP app."""

from __future__ import annotations

import importlib

import torch


class _DummyBatch(dict):
    def to(self, *_args, **_kwargs):
        return self


class _DummyProcessor:
    def __call__(self, *, text, images, return_tensors):
        assert text.startswith("<CAPTION_TO_PHRASE_GROUNDING>")
        assert return_tensors == "pt"
        return _DummyBatch(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            pixel_values=torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        )

    def batch_decode(self, generated_ids, skip_special_tokens=False):
        assert generated_ids.shape == (1, 3)
        assert skip_special_tokens is False
        return ["dummy"]

    def post_process_generation(self, generated_text, task, image_size):
        assert generated_text == "dummy"
        assert task == "<CAPTION_TO_PHRASE_GROUNDING>"
        return {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[10, 20, 30, 40]],
                "labels": ["car"],
            }
        }


class _DummyModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def test_detector_disables_cache_for_florence_generation(monkeypatch):
    monkeypatch.setenv("FAKE_MODELS", "0")

    import mvp_app.config
    import mvp_app.detector

    importlib.reload(mvp_app.config)
    importlib.reload(mvp_app.detector)

    service = mvp_app.detector.DetectionService()
    dummy_model = _DummyModel()
    service._model = dummy_model
    service._processor = _DummyProcessor()

    result = service.detect(
        image_bytes=(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\xf8\xff\xff?"
            b"\x00\x05\xfe\x02\xfeA\x89\x18\x00\x00\x00\x00IEND\xaeB`\x82"
        ),
        classes=["car", "person"],
    )

    assert dummy_model.kwargs is not None
    assert dummy_model.kwargs["use_cache"] is False
    assert result["labels"] == ["car"]
    assert result["boxes"] == [[10, 20, 30, 40]]


def test_detector_matches_pluralized_florence_labels():
    import mvp_app.detector

    service = mvp_app.detector.DetectionService()
    assert service._match_requested_class("cars", ["car", "person"]) == "car"
    assert service._match_requested_class("a person", ["car", "person"]) == "person"
    assert service._match_requested_class("traffic light", ["car", "person"]) is None
