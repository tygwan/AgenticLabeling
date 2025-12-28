"""Florence-2 detector implementation with lazy loading."""
import io
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class Florence2Detector:
    """Florence-2 based object detector with singleton pattern."""

    _instance: Optional["Florence2Detector"] = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = "microsoft/Florence-2-large"
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Explicitly use float32 to avoid dtype mismatch
                attn_implementation="eager",  # Avoid SDPA compatibility issues
            ).to(self.device).eval()
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )

    def detect(
        self,
        image_bytes: bytes,
        classes: List[str],
        confidence: float = 0.5,
    ) -> dict:
        """Detect objects in image.

        Args:
            image_bytes: Raw image bytes
            classes: List of class names to detect
            confidence: Minimum confidence threshold

        Returns:
            Dict with boxes, labels, and confidence scores
        """
        self._ensure_loaded()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = "A photo of " + ", and ".join(classes) + "."

        inputs = self._processor(
            text=prompt + text_input,
            images=image,
            return_tensors="pt",
        )
        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,  # Use greedy decoding to avoid beam search compatibility issues
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        result = self._processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )

        parsed = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        boxes = parsed.get("bboxes", [])
        labels = parsed.get("labels", [])

        # Filter by requested classes
        filtered_boxes = []
        filtered_labels = []
        for box, label in zip(boxes, labels):
            if label in classes:
                filtered_boxes.append(box)
                filtered_labels.append(label)

        return {
            "boxes": filtered_boxes,
            "labels": filtered_labels,
            "confidence": [1.0] * len(filtered_boxes),  # Florence-2 doesn't provide confidence
            "image_size": {"width": image.width, "height": image.height},
        }

    def unload(self):
        """Unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()
