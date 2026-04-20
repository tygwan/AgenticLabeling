"""SAM3 segmenter implementation with lazy loading (legacy service)."""
import base64
import io
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class SAM3Segmenter:
    """SAM3 based segmenter with singleton pattern."""

    _instance: Optional["SAM3Segmenter"] = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _xyxy_to_cxcywh_norm(box: List[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = box
        return [
            (x1 + x2) / 2.0 / width,
            (y1 + y2) / 2.0 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

    def _ensure_loaded(self):
        if self._processor is None:
            vendor_sam3 = os.path.join(os.getcwd(), "vendor", "sam3")
            if os.path.isdir(vendor_sam3) and vendor_sam3 not in sys.path:
                sys.path.insert(0, vendor_sam3)

            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model(
                device=str(self.device),
                eval_mode=True,
                enable_inst_interactivity=False,
                enable_segmentation=True,
            )
            self._processor = Sam3Processor(
                model, device=str(self.device), confidence_threshold=0.3,
            )

    def segment(
        self,
        image_bytes: bytes,
        boxes: List[List[float]],
    ) -> dict:
        self._ensure_loaded()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
                buffer = io.BytesIO()
                Image.fromarray(mask_np).save(buffer, format="PNG")
                masks_data.append({
                    "mask": base64.b64encode(buffer.getvalue()).decode(),
                    "score": float(scores[best_idx]),
                })
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                x1, y1, x2, y2 = [int(v) for v in box]
                mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
                buffer = io.BytesIO()
                Image.fromarray(mask).save(buffer, format="PNG")
                masks_data.append({
                    "mask": base64.b64encode(buffer.getvalue()).decode(),
                    "score": 0.0,
                })

        return {
            "masks": masks_data,
            "image_size": {"width": image.width, "height": image.height},
        }

    def unload(self):
        if self._processor is not None:
            del self._processor
            self._processor = None
            torch.cuda.empty_cache()
