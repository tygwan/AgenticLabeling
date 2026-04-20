"""Florence-2 detector adapted for the monolith MVP app."""

from __future__ import annotations

import io
import re
from typing import Optional

import torch
from PIL import Image

from .config import get_settings


class DetectionService:
    """Lazy Florence-2 detector with an optional fake mode for tests."""

    _instance: Optional["DetectionService"] = None
    _model = None
    _processor = None

    def __new__(cls) -> "DetectionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _normalize_label(value: str) -> str:
        """Normalize labels for loose class matching."""
        normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        tokens = []
        for token in normalized.split():
            if token.endswith("es") and len(token) > 3:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            tokens.append(token)
        return " ".join(tokens)

    def _match_requested_class(self, raw_label: str, classes: list[str]) -> Optional[str]:
        """Map a raw Florence label to the closest requested class if possible."""
        if not classes:
            return raw_label

        normalized_raw = self._normalize_label(raw_label)
        normalized_map = {name: self._normalize_label(name) for name in classes}

        for original, normalized in normalized_map.items():
            if normalized_raw == normalized:
                return original

        for original, normalized in normalized_map.items():
            if normalized and (normalized in normalized_raw or normalized_raw in normalized):
                return original

        raw_tokens = set(normalized_raw.split())
        for original, normalized in normalized_map.items():
            normalized_tokens = set(normalized.split())
            if raw_tokens and normalized_tokens and raw_tokens & normalized_tokens:
                return original

        return None

    _DTYPE_MAP = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }

    @classmethod
    def _resolve_dtype(cls, name: str) -> torch.dtype:
        key = name.lower()
        if key not in cls._DTYPE_MAP:
            raise ValueError(
                f"FLORENCE_DTYPE='{name}' not recognized. "
                f"Expected one of: {sorted(set(cls._DTYPE_MAP))}"
            )
        return cls._DTYPE_MAP[key]

    def _ensure_loaded(self) -> None:
        settings = get_settings()
        if settings.fake_models or self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            settings.florence_model_id,
            trust_remote_code=True,
            dtype=self._resolve_dtype(settings.florence_dtype),
            attn_implementation="eager",
        ).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(
            settings.florence_model_id,
            trust_remote_code=True,
        )

    def _generate_grounding(self, inputs: dict) -> torch.Tensor:
        """Run Florence-2 generation with cache disabled for compatibility."""
        return self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )

    def detect(self, image_bytes: bytes, classes: list[str]) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        settings = get_settings()
        if settings.fake_models:
            return {
                "boxes": [[image.width * 0.2, image.height * 0.2, image.width * 0.8, image.height * 0.8]],
                "labels": [classes[0] if classes else "object"],
                "scores": [0.99],
                "image_size": {"width": image.width, "height": image.height},
            }

        self._ensure_loaded()
        prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = "A photo of " + ", and ".join(classes) + "."
        inputs = self._processor(
            text=prompt + text_input,
            images=image,
            return_tensors="pt",
        )
        try:
            model_dtype = next(self._model.parameters()).dtype
        except (StopIteration, AttributeError):
            model_dtype = None
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if model_dtype is not None and isinstance(v, torch.Tensor) and v.is_floating_point():
                    moved[k] = v.to(device=self.device, dtype=model_dtype)
                else:
                    moved[k] = v.to(self.device)
            else:
                moved[k] = v
        inputs = moved

        with torch.inference_mode():
            generated_ids = self._generate_grounding(inputs)

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = self._processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        parsed = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        raw_boxes = parsed.get("bboxes", [])
        raw_labels = parsed.get("labels", [])

        boxes = []
        filtered_labels = []
        for box, raw_label in zip(raw_boxes, raw_labels):
            matched_label = self._match_requested_class(raw_label, classes)
            if matched_label is not None:
                boxes.append(box)
                filtered_labels.append(matched_label)

        # Florence-2 often returns phrase variants like "cars" or "a person".
        # For the MVP, keep raw detections instead of dropping everything when
        # prompt-constrained labels fail exact matching.
        if raw_boxes and not boxes:
            boxes = list(raw_boxes)
            filtered_labels = list(raw_labels)

        return {
            "boxes": boxes,
            "labels": filtered_labels,
            "scores": [1.0] * len(boxes),
            "image_size": {"width": image.width, "height": image.height},
        }

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
