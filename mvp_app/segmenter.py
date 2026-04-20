"""SAM3 segmenter adapted for the monolith MVP app."""

from __future__ import annotations

import base64
import io
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .config import get_settings


class SegmentationService:
    """Lazy SAM3 segmenter with an optional fake mode for tests."""

    _instance: Optional["SegmentationService"] = None
    _processor = None

    def __new__(cls) -> "SegmentationService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sam3_import_error: Optional[str] = None

    @property
    def backend_name(self) -> str:
        if self._processor not in (None, False):
            return "sam3"
        if self._sam3_import_error:
            return "box-fallback"
        return "uninitialized"

    @staticmethod
    def _encode_mask(mask_np: np.ndarray, score: float) -> dict:
        buffer = io.BytesIO()
        Image.fromarray(mask_np).save(buffer, format="PNG")
        return {"mask": base64.b64encode(buffer.getvalue()).decode(), "score": score}

    @staticmethod
    def _encode_box_masks(image_np: np.ndarray, boxes: list[list[float]]) -> list[dict]:
        """Fallback mask generator that fills the provided bounding boxes."""
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
            buffer = io.BytesIO()
            Image.fromarray(mask).save(buffer, format="PNG")
            masks.append({"mask": base64.b64encode(buffer.getvalue()).decode(), "score": 1.0})
        return masks

    @staticmethod
    def _ensure_vendor_python_path() -> None:
        vendor_sam3 = os.path.join(os.getcwd(), "vendor", "sam3")
        if os.path.isdir(vendor_sam3) and vendor_sam3 not in sys.path:
            sys.path.insert(0, vendor_sam3)

    def _ensure_loaded(self) -> None:
        settings = get_settings()
        if settings.fake_models or self._processor is not None:
            return
        self._ensure_vendor_python_path()
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ModuleNotFoundError as exc:
            self._sam3_import_error = str(exc)
            self._processor = False
            return

        try:
            checkpoint = settings.sam3_checkpoint or None
            model = build_sam3_image_model(
                device=str(self.device),
                eval_mode=True,
                checkpoint_path=checkpoint,
                load_from_HF=(checkpoint is None),
                enable_inst_interactivity=False,
                enable_segmentation=True,
            )
            self._processor = Sam3Processor(
                model, device=str(self.device), confidence_threshold=0.3,
            )
            self._sam3_import_error = None
        except Exception as exc:
            self._sam3_import_error = str(exc)
            self._processor = False

    @staticmethod
    def _xyxy_to_cxcywh_norm(box: list[float], width: int, height: int) -> list[float]:
        """Convert [x1,y1,x2,y2] pixel coords to [cx,cy,w,h] normalized [0,1]."""
        x1, y1, x2, y2 = box
        return [
            (x1 + x2) / 2.0 / width,
            (y1 + y2) / 2.0 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

    def segment(self, image_bytes: bytes, boxes: list[list[float]]) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        settings = get_settings()
        if not boxes:
            return {"masks": [], "image_size": {"width": image.width, "height": image.height}}

        image_np = np.array(image)
        if settings.fake_models:
            return {"masks": self._encode_box_masks(image_np, boxes), "image_size": {"width": image.width, "height": image.height}}

        self._ensure_loaded()
        if self._processor is False:
            return {"masks": self._encode_box_masks(image_np, boxes), "image_size": {"width": image.width, "height": image.height}}

        masks_data = []
        state = self._processor.set_image(image)
        for box in boxes:
            self._processor.reset_all_prompts(state)
            box_norm = self._xyxy_to_cxcywh_norm(box, image.width, image.height)
            state = self._processor.add_geometric_prompt(
                box=box_norm, label=True, state=state,
            )
            if "masks" in state and len(state["masks"]) > 0:
                scores = state["scores"]
                best_idx = int(scores.argmax())
                mask_tensor = state["masks"][best_idx][0]
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                masks_data.append(self._encode_mask(mask_np, float(scores[best_idx])))
            else:
                masks_data.append(self._encode_box_masks(image_np, [box])[0])

        return {"masks": masks_data, "image_size": {"width": image.width, "height": image.height}}

    def unload(self) -> None:
        if self._processor not in (None, False):
            del self._processor
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
