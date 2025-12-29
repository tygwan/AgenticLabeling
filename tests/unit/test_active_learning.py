"""Unit tests for Active Learning module."""
import os
import sys
import importlib.util
import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Dynamic import to avoid conflicts with other 'app' packages
_al_path = os.path.join(
    os.path.dirname(__file__), "../../services/training-agent/app/active_learning.py"
)
_spec = importlib.util.spec_from_file_location("active_learning", _al_path)
_al_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_al_module)
ActiveLearner = _al_module.ActiveLearner
SamplingStrategy = _al_module.SamplingStrategy
UncertainSample = _al_module.UncertainSample


class TestActiveLearner:
    """Test ActiveLearner class."""

    @pytest.fixture
    def learner(self):
        """Create an ActiveLearner instance."""
        return ActiveLearner(
            registry_url="http://mock-registry:8010",
            data_manager_url="http://mock-data-manager:8006",
        )

    def test_calculate_entropy_uniform(self, learner):
        """Test entropy calculation for uniform distribution."""
        # Uniform distribution should have high entropy
        confidences = [0.25, 0.25, 0.25, 0.25]
        entropy = learner.calculate_entropy(confidences)
        # Max entropy for 4 classes is ln(4) ≈ 1.386
        assert entropy > 1.0

    def test_calculate_entropy_certain(self, learner):
        """Test entropy calculation for certain prediction."""
        # One dominant class should have low entropy
        confidences = [0.99, 0.01]
        entropy = learner.calculate_entropy(confidences)
        assert entropy < 0.1

    def test_calculate_entropy_empty(self, learner):
        """Test entropy calculation for empty input."""
        entropy = learner.calculate_entropy([])
        assert entropy == 0.0

    def test_calculate_margin_high_confidence(self, learner):
        """Test margin calculation with high confidence gap."""
        confidences = [0.9, 0.1]
        margin = learner.calculate_margin(confidences)
        assert margin == 0.8

    def test_calculate_margin_low_confidence(self, learner):
        """Test margin calculation with low confidence gap."""
        confidences = [0.51, 0.49]
        margin = learner.calculate_margin(confidences)
        assert abs(margin - 0.02) < 0.001

    def test_calculate_margin_single_class(self, learner):
        """Test margin calculation with single class."""
        margin = learner.calculate_margin([0.95])
        assert margin == 1.0  # Max margin for single class

    def test_uncertainty_score_low_confidence(self, learner):
        """Test uncertainty score for low confidence sample."""
        score = learner.calculate_uncertainty_score(
            confidence=0.3,
            strategy=SamplingStrategy.UNCERTAINTY,
        )
        assert score == 0.7  # 1 - 0.3

    def test_uncertainty_score_high_confidence(self, learner):
        """Test uncertainty score for high confidence sample."""
        score = learner.calculate_uncertainty_score(
            confidence=0.95,
            strategy=SamplingStrategy.UNCERTAINTY,
        )
        # Use approximate comparison for floating point
        assert abs(score - 0.05) < 0.0001  # 1 - 0.95

    def test_uncertainty_score_entropy_strategy(self, learner):
        """Test uncertainty score with entropy strategy."""
        score = learner.calculate_uncertainty_score(
            confidence=0.5,
            entropy=1.0,
            strategy=SamplingStrategy.ENTROPY,
        )
        # entropy=1.0, max_entropy=2.0, so score = 0.5
        assert score == 0.5

    def test_uncertainty_score_random_strategy(self, learner):
        """Test uncertainty score with random strategy."""
        scores = [
            learner.calculate_uncertainty_score(
                confidence=0.5,
                strategy=SamplingStrategy.RANDOM,
            )
            for _ in range(10)
        ]
        # Random should produce varied scores (at least 2 different values in 10 tries)
        assert len(set(scores)) >= 2

    def test_query_uncertain_samples(self, learner):
        """Test querying uncertain samples."""
        # Mock the HTTP client
        mock_objects_response = MagicMock()
        mock_objects_response.status_code = 200
        mock_objects_response.json.return_value = {
            "data": [
                {
                    "object_id": "obj_001",
                    "source_id": "src_001",
                    "confidence": 0.3,
                    "category_name": "person",
                    "bbox_x": 10,
                    "bbox_y": 20,
                    "bbox_w": 100,
                    "bbox_h": 200,
                },
                {
                    "object_id": "obj_002",
                    "source_id": "src_001",
                    "confidence": 0.9,  # High confidence, should be filtered
                    "category_name": "car",
                    "bbox_x": 50,
                    "bbox_y": 60,
                    "bbox_w": 150,
                    "bbox_h": 100,
                },
                {
                    "object_id": "obj_003",
                    "source_id": "src_002",
                    "confidence": 0.4,
                    "category_name": "person",
                    "bbox_x": 30,
                    "bbox_y": 40,
                    "bbox_w": 80,
                    "bbox_h": 160,
                },
            ]
        }

        mock_source_response = MagicMock()
        mock_source_response.status_code = 200
        mock_source_response.json.return_value = {
            "data": {"file_path": "/data/images/test.jpg"}
        }

        async def mock_get(*args, **kwargs):
            if "/objects" in args[0]:
                return mock_objects_response
            return mock_source_response

        with patch.object(learner, '_client') as mock_client:
            mock_client.get = AsyncMock(side_effect=mock_get)
            learner._client = mock_client

            # Run async test
            samples = asyncio.get_event_loop().run_until_complete(
                learner.query_uncertain_samples(
                    strategy=SamplingStrategy.UNCERTAINTY,
                    n_samples=5,
                    max_confidence=0.7,
                )
            )

        # Should return 2 samples (0.3 and 0.4 confidence)
        assert len(samples) == 2
        # First should be lowest confidence
        assert samples[0].confidence == 0.3
        assert samples[0].uncertainty_score > samples[1].uncertainty_score

    def test_get_labeling_stats(self, learner):
        """Test getting labeling statistics."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "objects": 100,
                "validated_objects": 25,
                "objects_per_category": {
                    "person": 60,
                    "car": 40,
                },
            }
        }

        with patch.object(learner, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            learner._client = mock_client

            stats = asyncio.get_event_loop().run_until_complete(
                learner.get_labeling_stats()
            )

        assert stats["total_objects"] == 100
        assert stats["validated_objects"] == 25
        assert stats["unlabeled_objects"] == 75
        assert stats["labeling_rate_percent"] == 25.0

    def test_estimate_model_performance_gain(self, learner):
        """Test estimating model performance gain."""
        estimate = asyncio.get_event_loop().run_until_complete(
            learner.estimate_model_performance_gain(
                current_validated=50,
                target_validated=100,
            )
        )

        assert estimate["current_validated"] == 50
        assert estimate["target_validated"] == 100
        assert estimate["additional_labels_needed"] == 50
        assert estimate["expected_gain_percent"] > 0
        assert isinstance(estimate["current_estimated_mAP"], (int, float))

    def test_estimate_performance_zero_labels(self, learner):
        """Test estimating performance with zero current labels."""
        estimate = asyncio.get_event_loop().run_until_complete(
            learner.estimate_model_performance_gain(
                current_validated=0,
                target_validated=50,
            )
        )

        assert estimate["current_estimate"] == 0
        assert estimate["expected_gain_percent"] > 0

    def test_run_active_learning_cycle(self, learner):
        """Test running an active learning cycle."""
        mock_objects_response = MagicMock()
        mock_objects_response.status_code = 200
        mock_objects_response.json.return_value = {
            "data": [
                {
                    "object_id": "obj_001",
                    "source_id": "src_001",
                    "confidence": 0.3,
                    "category_name": "person",
                    "bbox_x": 10,
                    "bbox_y": 20,
                    "bbox_w": 100,
                    "bbox_h": 200,
                },
            ]
        }

        mock_source_response = MagicMock()
        mock_source_response.status_code = 200
        mock_source_response.json.return_value = {
            "data": {"file_path": "/data/images/test.jpg"}
        }

        async def mock_get(*args, **kwargs):
            if "/objects" in args[0]:
                return mock_objects_response
            return mock_source_response

        with patch.object(learner, '_client') as mock_client:
            mock_client.get = AsyncMock(side_effect=mock_get)
            learner._client = mock_client

            result = asyncio.get_event_loop().run_until_complete(
                learner.run_active_learning_cycle(
                    model_path="yolov8n.pt",
                    project_id="test_project",
                    strategy=SamplingStrategy.UNCERTAINTY,
                    n_query=10,
                )
            )

        assert result["status"] == "samples_ready"
        assert result["project_id"] == "test_project"
        assert result["n_samples"] == 1
        assert len(result["samples"]) == 1


class TestSamplingStrategy:
    """Test SamplingStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all expected strategies are defined."""
        expected = ["UNCERTAINTY", "ENTROPY", "MARGIN", "RANDOM", "DIVERSITY"]
        for strategy in expected:
            assert hasattr(SamplingStrategy, strategy)

    def test_strategy_values(self):
        """Test strategy string values."""
        assert SamplingStrategy.UNCERTAINTY.value == "uncertainty"
        assert SamplingStrategy.ENTROPY.value == "entropy"
        assert SamplingStrategy.MARGIN.value == "margin"
        assert SamplingStrategy.RANDOM.value == "random"
        assert SamplingStrategy.DIVERSITY.value == "diversity"


class TestUncertainSample:
    """Test UncertainSample dataclass."""

    def test_create_sample(self):
        """Test creating an uncertain sample."""
        sample = UncertainSample(
            object_id="obj_001",
            source_id="src_001",
            image_path="/data/test.jpg",
            confidence=0.35,
            entropy=0.9,
            predicted_category="person",
            bbox=[10, 20, 100, 200],
            uncertainty_score=0.65,
        )

        assert sample.object_id == "obj_001"
        assert sample.confidence == 0.35
        assert sample.uncertainty_score == 0.65
        assert sample.bbox == [10, 20, 100, 200]
