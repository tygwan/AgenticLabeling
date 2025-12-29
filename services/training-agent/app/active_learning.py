"""Active Learning module for uncertainty sampling and iterative training."""
import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np


class SamplingStrategy(str, Enum):
    """Active learning sampling strategies."""
    UNCERTAINTY = "uncertainty"  # Lowest confidence samples
    ENTROPY = "entropy"  # Highest prediction entropy
    MARGIN = "margin"  # Smallest margin between top predictions
    RANDOM = "random"  # Random sampling (baseline)
    DIVERSITY = "diversity"  # Embedding-based diversity sampling


@dataclass
class UncertainSample:
    """Represents an uncertain sample for labeling."""
    object_id: str
    source_id: str
    image_path: str
    confidence: float
    entropy: float
    predicted_category: str
    bbox: List[float]
    uncertainty_score: float  # Combined score for ranking


class ActiveLearner:
    """Active Learning engine for iterative model improvement."""

    def __init__(
        self,
        registry_url: str = None,
        data_manager_url: str = None,
    ):
        self.registry_url = registry_url or os.getenv(
            "REGISTRY_URL", "http://object-registry:8010"
        )
        self.data_manager_url = data_manager_url or os.getenv(
            "DATA_MANAGER_URL", "http://data-manager:8006"
        )
        self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def calculate_entropy(self, confidences: List[float]) -> float:
        """Calculate prediction entropy from confidence scores.

        Higher entropy = more uncertain prediction.
        """
        if not confidences:
            return 0.0

        probs = np.array(confidences)
        probs = probs / probs.sum()  # Normalize

        # Avoid log(0) issues
        probs = np.clip(probs, 1e-10, 1.0)

        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def calculate_margin(self, confidences: List[float]) -> float:
        """Calculate margin between top-2 predictions.

        Smaller margin = more uncertain prediction.
        """
        if len(confidences) < 2:
            return 1.0  # Max confidence margin

        sorted_conf = sorted(confidences, reverse=True)
        margin = sorted_conf[0] - sorted_conf[1]
        return float(margin)

    def calculate_uncertainty_score(
        self,
        confidence: float,
        entropy: float = 0.0,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
    ) -> float:
        """Calculate unified uncertainty score for ranking.

        Higher score = more uncertain = should be labeled first.
        """
        if strategy == SamplingStrategy.UNCERTAINTY:
            # Lower confidence = higher uncertainty
            return 1.0 - confidence

        elif strategy == SamplingStrategy.ENTROPY:
            # Normalize entropy (max entropy for binary = ln(2) ≈ 0.693)
            max_entropy = 2.0  # Approximate max for multi-class
            return min(entropy / max_entropy, 1.0)

        elif strategy == SamplingStrategy.MARGIN:
            # Already calculated as margin
            return entropy  # Reuse field for margin

        elif strategy == SamplingStrategy.RANDOM:
            return np.random.random()

        else:
            return 1.0 - confidence

    async def query_uncertain_samples(
        self,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
        n_samples: int = 10,
        min_confidence: float = 0.0,
        max_confidence: float = 0.7,  # Only consider low-confidence samples
        exclude_validated: bool = True,
        category: Optional[str] = None,
    ) -> List[UncertainSample]:
        """Query uncertain samples from registry for labeling.

        Args:
            strategy: Sampling strategy to use
            n_samples: Number of samples to return
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold (filter out high-confidence)
            exclude_validated: Skip already validated objects
            category: Filter by specific category

        Returns:
            List of uncertain samples sorted by uncertainty score
        """
        # Query objects from registry
        params = {
            "limit": n_samples * 5,  # Get more for filtering
            "min_confidence": min_confidence,
        }

        if exclude_validated:
            params["is_validated"] = False

        if category:
            params["category"] = category

        response = await self.client.get(
            f"{self.registry_url}/objects",
            params=params,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to query registry: {response.text}")

        data = response.json()
        objects = data.get("data", [])

        # Filter by max confidence and calculate uncertainty
        uncertain_samples = []
        for obj in objects:
            confidence = obj.get("confidence", 0.0) or 0.0

            # Skip high-confidence samples (likely correct)
            if confidence > max_confidence:
                continue

            # Get source info for image path
            source_id = obj.get("source_id")
            source_response = await self.client.get(
                f"{self.registry_url}/sources/{source_id}"
            )

            image_path = ""
            if source_response.status_code == 200:
                source_data = source_response.json().get("data", {})
                image_path = source_data.get("file_path", "")

            # Calculate uncertainty score
            entropy = self.calculate_entropy([confidence, 1 - confidence])
            uncertainty = self.calculate_uncertainty_score(
                confidence=confidence,
                entropy=entropy,
                strategy=strategy,
            )

            uncertain_samples.append(UncertainSample(
                object_id=obj.get("object_id"),
                source_id=source_id,
                image_path=image_path,
                confidence=confidence,
                entropy=entropy,
                predicted_category=obj.get("category_name", "unknown"),
                bbox=[
                    obj.get("bbox_x", 0),
                    obj.get("bbox_y", 0),
                    obj.get("bbox_w", 0),
                    obj.get("bbox_h", 0),
                ],
                uncertainty_score=uncertainty,
            ))

        # Sort by uncertainty (highest first)
        uncertain_samples.sort(key=lambda x: x.uncertainty_score, reverse=True)

        return uncertain_samples[:n_samples]

    async def query_by_embedding_diversity(
        self,
        n_samples: int = 10,
        exclude_validated: bool = True,
    ) -> List[UncertainSample]:
        """Select diverse samples based on embedding distances.

        Uses embedding search to find samples that are most different
        from already validated samples.
        """
        # Get validated embeddings as reference
        params = {"is_validated": True, "limit": 100}
        response = await self.client.get(
            f"{self.registry_url}/objects",
            params=params,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to query registry: {response.text}")

        # For diversity sampling, we'd need embeddings which requires
        # integration with classification-agent. For now, fall back to uncertainty.
        return await self.query_uncertain_samples(
            strategy=SamplingStrategy.UNCERTAINTY,
            n_samples=n_samples,
            exclude_validated=exclude_validated,
        )

    async def run_active_learning_cycle(
        self,
        model_path: str,
        project_id: str,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
        n_query: int = 10,
        min_confidence: float = 0.0,
        max_confidence: float = 0.7,
    ) -> Dict:
        """Run one cycle of active learning.

        Steps:
        1. Query uncertain samples
        2. Return them for human labeling
        3. (After labeling) Retrain model

        Returns:
            Dict with cycle status and samples to label
        """
        # Query uncertain samples
        samples = await self.query_uncertain_samples(
            strategy=strategy,
            n_samples=n_query,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            exclude_validated=True,
        )

        return {
            "status": "samples_ready",
            "project_id": project_id,
            "model_path": model_path,
            "strategy": strategy.value,
            "n_samples": len(samples),
            "samples": [
                {
                    "object_id": s.object_id,
                    "source_id": s.source_id,
                    "image_path": s.image_path,
                    "confidence": s.confidence,
                    "entropy": s.entropy,
                    "predicted_category": s.predicted_category,
                    "bbox": s.bbox,
                    "uncertainty_score": s.uncertainty_score,
                }
                for s in samples
            ],
            "next_action": "Label these samples and call /active-learning/retrain",
        }

    async def get_labeling_stats(self) -> Dict:
        """Get statistics about labeling progress for active learning."""
        # Get total and validated counts
        response = await self.client.get(f"{self.registry_url}/stats")

        if response.status_code != 200:
            raise Exception(f"Failed to get stats: {response.text}")

        stats = response.json().get("data", {})

        total_objects = stats.get("objects", 0)
        validated_objects = stats.get("validated_objects", 0)

        # Calculate labeling progress
        labeling_rate = (validated_objects / total_objects * 100) if total_objects > 0 else 0

        return {
            "total_objects": total_objects,
            "validated_objects": validated_objects,
            "unlabeled_objects": total_objects - validated_objects,
            "labeling_rate_percent": round(labeling_rate, 2),
            "categories": stats.get("objects_per_category", {}),
        }

    async def estimate_model_performance_gain(
        self,
        current_validated: int,
        target_validated: int,
    ) -> Dict:
        """Estimate expected model performance gain from more labels.

        Uses empirical power law: performance ∝ n^α where α ≈ 0.3-0.5
        """
        if current_validated <= 0:
            return {
                "current_estimate": 0,
                "target_estimate": 50,  # Rough estimate
                "expected_gain_percent": 50,
            }

        alpha = 0.4  # Typical learning curve exponent

        # Normalize to get relative performance
        current_perf = current_validated ** alpha
        target_perf = target_validated ** alpha

        # Estimate mAP improvement (rough approximation)
        # Assume baseline 50% mAP with minimal data
        baseline_map = 50
        max_map = 95

        current_map = baseline_map + (max_map - baseline_map) * (
            1 - np.exp(-current_perf / 100)
        )
        target_map = baseline_map + (max_map - baseline_map) * (
            1 - np.exp(-target_perf / 100)
        )

        return {
            "current_validated": current_validated,
            "target_validated": target_validated,
            "additional_labels_needed": target_validated - current_validated,
            "current_estimated_mAP": round(current_map, 1),
            "target_estimated_mAP": round(target_map, 1),
            "expected_gain_percent": round(target_map - current_map, 1),
            "note": "These are rough estimates based on power law scaling",
        }
