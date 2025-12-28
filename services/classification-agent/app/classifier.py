"""Feature extraction and cosine similarity classification."""
import io
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from PIL import Image


class FeatureExtractor:
    """Feature extractor supporting CLIP, DINOv2, and ResNet."""

    def __init__(self, model_name: str = "dino"):
        self.model_name = model_name.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.preprocess = None

    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self.model is not None:
            return

        if self.model_name == "clip":
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        elif self.model_name == "dino":
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.model.to(self.device)
            self.model.eval()

        elif self.model_name == "resnet":
            import torchvision.models as models
            from torchvision import transforms

            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
            self.model.to(self.device)
            self.model.eval()

            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract feature vector from PIL Image."""
        self._ensure_loaded()

        with torch.no_grad():
            if self.model_name == "clip":
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model.encode_image(image_input)

            elif self.model_name == "dino":
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.pooler_output

            elif self.model_name == "resnet":
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model(image_input)

            return features.cpu().numpy().flatten()

    def extract_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Extract features from raw image bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.extract_features(image)

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            del self.preprocess
            self.model = None
            self.processor = None
            self.preprocess = None
            torch.cuda.empty_cache()


class CosineSimilarityClassifier:
    """Few-shot classifier using cosine similarity."""

    def __init__(self, model_name: str = "dino", k_shot: int = 5):
        self.extractor = FeatureExtractor(model_name)
        self.model_name = model_name
        self.k_shot = k_shot
        self.support_features: Dict[str, np.ndarray] = {}

    def has_support_set(self) -> bool:
        """Check if support set is loaded."""
        return len(self.support_features) > 0

    def load_support_set_from_bytes(self, support_data: Dict[str, List[bytes]]):
        """Load support set from image bytes.

        Args:
            support_data: Dict mapping class names to lists of image bytes
        """
        self.support_features.clear()

        for class_name, image_bytes_list in support_data.items():
            features = []
            for img_bytes in image_bytes_list:
                try:
                    feat = self.extractor.extract_from_bytes(img_bytes)
                    features.append(feat)
                except Exception:
                    continue

            if features:
                self.support_features[class_name] = np.array(features)

    def classify(
        self,
        image_bytes: bytes,
        threshold: float = 0.75,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Classify an image.

        Args:
            image_bytes: Raw image bytes
            threshold: Minimum similarity threshold
            top_k: Number of top predictions to return

        Returns:
            Classification result dict
        """
        query_feature = self.extractor.extract_from_bytes(image_bytes)

        # Compute similarities
        similarities = {}
        for class_name, features in self.support_features.items():
            class_sims = []
            for feat in features:
                sim = self._cosine_similarity(query_feature, feat)
                class_sims.append(sim)
            similarities[class_name] = float(np.mean(class_sims))

        # Sort by similarity
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        if not sorted_sims:
            return {
                "class": "Unknown",
                "similarity": 0.0,
                "all_scores": {},
                "status": "no_support_set",
            }

        best_class, best_score = sorted_sims[0]

        # Determine final class based on threshold
        final_class = best_class if best_score >= threshold else "Unknown"

        # Compute margin for confidence estimation
        margin = 0.0
        if len(sorted_sims) > 1:
            margin = best_score - sorted_sims[1][1]

        confidence_level = "high" if margin > 0.2 else ("medium" if margin > 0.1 else "low")

        return {
            "class": final_class,
            "predicted_class": best_class,
            "similarity": best_score,
            "margin": margin,
            "confidence_level": confidence_level,
            "top_predictions": [
                {"class": cls, "similarity": score}
                for cls, score in sorted_sims[:top_k]
            ],
            "all_scores": similarities,
            "status": "accepted" if final_class != "Unknown" else "rejected",
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def unload(self):
        """Free resources."""
        self.extractor.unload()
        self.support_features.clear()
