"""SAM2 segmenter implementation with lazy loading."""
import base64
import io
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class SAM2Segmenter:
    """SAM2 based segmenter with singleton pattern."""

    _instance: Optional["SAM2Segmenter"] = None
    _predictor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self._predictor is None:
            import os
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            checkpoint = os.path.expanduser(
                "~/.cache/autodistill/segment_anything_2/sam2_hiera_base_plus.pt"
            )
            model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
            sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
            self._predictor = SAM2ImagePredictor(sam2_model)

    def segment(
        self,
        image_bytes: bytes,
        boxes: List[List[float]],
    ) -> dict:
        """Segment objects using bounding boxes.

        Args:
            image_bytes: Raw image bytes
            boxes: List of bounding boxes [[x1,y1,x2,y2], ...]

        Returns:
            Dict with masks as base64 encoded PNGs
        """
        self._ensure_loaded()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        masks_data = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.set_image(image_np)

            for box in boxes:
                masks, scores, _ = self._predictor.predict(
                    box=np.array(box),
                    multimask_output=False,
                )
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.uint8) * 255

                # Encode mask as base64 PNG
                mask_img = Image.fromarray(mask)
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                mask_b64 = base64.b64encode(buffer.getvalue()).decode()
                masks_data.append({
                    "mask": mask_b64,
                    "score": float(scores[best_idx]),
                })

        return {
            "masks": masks_data,
            "image_size": {"width": image.width, "height": image.height},
        }

    def segment_with_points(
        self,
        image_bytes: bytes,
        points: List[List[float]],
        labels: List[int],
    ) -> dict:
        """Segment using point prompts.

        Args:
            image_bytes: Raw image bytes
            points: List of points [[x, y], ...]
            labels: List of labels (1=foreground, 0=background)

        Returns:
            Dict with masks as base64 encoded PNGs
        """
        self._ensure_loaded()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.set_image(image_np)

            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                multimask_output=True,
            )

            masks_data = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                mask_uint8 = mask.astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_uint8)
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                mask_b64 = base64.b64encode(buffer.getvalue()).decode()
                masks_data.append({
                    "mask": mask_b64,
                    "score": float(score),
                })

        return {
            "masks": masks_data,
            "image_size": {"width": image.width, "height": image.height},
        }

    def unload(self):
        """Unload model to free GPU memory."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            torch.cuda.empty_cache()
