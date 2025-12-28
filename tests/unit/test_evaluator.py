"""Unit tests for Evaluation Agent evaluator module."""
import sys
import os
import importlib.util
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Dynamic import to avoid conflicts with other 'app' packages
_evaluator_path = os.path.join(
    os.path.dirname(__file__), "../../services/evaluation-agent/app/evaluator.py"
)
_spec = importlib.util.spec_from_file_location("evaluator", _evaluator_path)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
ModelEvaluator = _evaluator_module.ModelEvaluator


class TestIoUCalculation:
    """Test IoU calculation methods."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_compute_iou_identical_boxes(self, evaluator):
        """Test IoU of identical boxes is 1.0."""
        box = np.array([0, 0, 100, 100])
        iou = evaluator._compute_iou(box, box)
        assert pytest.approx(iou, 0.001) == 1.0

    def test_compute_iou_no_overlap(self, evaluator):
        """Test IoU of non-overlapping boxes is 0."""
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])
        iou = evaluator._compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self, evaluator):
        """Test IoU of partially overlapping boxes."""
        box1 = np.array([0, 0, 100, 100])  # Area: 10000
        box2 = np.array([50, 50, 150, 150])  # Area: 10000
        # Intersection: [50, 50, 100, 100] = 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 = 0.1428...
        iou = evaluator._compute_iou(box1, box2)
        assert pytest.approx(iou, 0.01) == 2500 / 17500

    def test_compute_iou_contained_box(self, evaluator):
        """Test IoU when one box contains the other."""
        box1 = np.array([0, 0, 100, 100])  # Area: 10000
        box2 = np.array([25, 25, 75, 75])  # Area: 2500
        # Intersection = 2500 (inner box)
        # Union = 10000 (outer box)
        iou = evaluator._compute_iou(box1, box2)
        assert pytest.approx(iou, 0.01) == 0.25

    def test_compute_iou_matrix(self, evaluator):
        """Test IoU matrix computation."""
        boxes1 = np.array([[0, 0, 50, 50], [100, 100, 200, 200]])
        boxes2 = np.array([[0, 0, 50, 50], [25, 25, 75, 75]])

        matrix = evaluator._compute_iou_matrix(boxes1, boxes2)

        assert matrix.shape == (2, 2)
        assert pytest.approx(matrix[0, 0], 0.001) == 1.0  # Identical
        assert matrix[0, 1] > 0  # Overlapping
        assert matrix[1, 0] == 0  # No overlap
        assert matrix[1, 1] == 0  # No overlap


class TestDetectionEvaluation:
    """Test detection evaluation methods."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_evaluate_detection_perfect_match(self, evaluator):
        """Test perfect detection evaluation."""
        predictions = [
            {
                "boxes": [[0, 0, 100, 100]],
                "labels": ["person"],
                "scores": [0.95],
            }
        ]
        ground_truth = [
            {
                "boxes": [[0, 0, 100, 100]],
                "labels": ["person"],
            }
        ]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert pytest.approx(metrics["precision"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["recall"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["f1_score"], abs=1e-6) == 1.0
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_evaluate_detection_false_positive(self, evaluator):
        """Test detection with false positive."""
        predictions = [
            {
                "boxes": [[0, 0, 50, 50], [200, 200, 300, 300]],
                "labels": ["person", "person"],
                "scores": [0.9, 0.8],
            }
        ]
        ground_truth = [
            {
                "boxes": [[0, 0, 50, 50]],
                "labels": ["person"],
            }
        ]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 0
        assert pytest.approx(metrics["precision"], abs=1e-6) == 0.5  # 1/(1+1)
        assert pytest.approx(metrics["recall"], abs=1e-6) == 1.0  # 1/(1+0)

    def test_evaluate_detection_false_negative(self, evaluator):
        """Test detection with missed object."""
        predictions = [
            {
                "boxes": [[0, 0, 50, 50]],
                "labels": ["person"],
                "scores": [0.9],
            }
        ]
        ground_truth = [
            {
                "boxes": [[0, 0, 50, 50], [200, 200, 300, 300]],
                "labels": ["person", "person"],
            }
        ]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 1
        assert pytest.approx(metrics["precision"], abs=1e-6) == 1.0  # 1/(1+0)
        assert pytest.approx(metrics["recall"], abs=1e-6) == 0.5  # 1/(1+1)

    def test_evaluate_detection_empty_predictions(self, evaluator):
        """Test detection with no predictions."""
        predictions = [{"boxes": [], "labels": [], "scores": []}]
        ground_truth = [{"boxes": [[0, 0, 50, 50]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 1

    def test_evaluate_detection_empty_ground_truth(self, evaluator):
        """Test detection with no ground truth."""
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [], "labels": []}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 0

    def test_evaluate_detection_iou_threshold(self, evaluator):
        """Test IoU threshold affects matching."""
        predictions = [
            {"boxes": [[0, 0, 100, 100]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [
            {"boxes": [[40, 40, 140, 140]], "labels": ["person"]}
        ]

        # Lower IoU threshold - should match
        metrics_low = evaluator.evaluate_detection(predictions, ground_truth, iou_threshold=0.2)
        assert metrics_low["true_positives"] == 1

        # Higher IoU threshold - should not match
        metrics_high = evaluator.evaluate_detection(predictions, ground_truth, iou_threshold=0.9)
        assert metrics_high["true_positives"] == 0
        assert metrics_high["false_positives"] == 1
        assert metrics_high["false_negatives"] == 1

    def test_evaluate_detection_multiclass(self, evaluator):
        """Test detection with multiple classes."""
        predictions = [
            {
                "boxes": [[0, 0, 50, 50], [100, 100, 150, 150]],
                "labels": ["person", "car"],
                "scores": [0.9, 0.8],
            }
        ]
        ground_truth = [
            {
                "boxes": [[0, 0, 50, 50], [100, 100, 150, 150]],
                "labels": ["person", "car"],
            }
        ]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert "person" in metrics["per_class"]
        assert "car" in metrics["per_class"]
        assert metrics["per_class"]["person"]["true_positives"] == 1
        assert metrics["per_class"]["car"]["true_positives"] == 1

    def test_evaluate_detection_returns_confusion_matrix(self, evaluator):
        """Test that detection returns confusion matrix."""
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 50, 50]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        assert "confusion_matrix" in metrics
        assert "classes" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_evaluate_detection_stores_evaluation(self, evaluator):
        """Test that evaluation is stored and retrievable."""
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 50, 50]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        eval_id = metrics["evaluation_id"]
        stored = evaluator.get_evaluation(eval_id)

        assert stored is not None
        assert stored["type"] == "detection"
        assert "metrics" in stored


class TestMAPCalculation:
    """Test mAP calculation methods."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_calculate_ap_perfect(self, evaluator):
        """Test AP calculation with perfect predictions."""
        # All true positives
        scores = [(0.9, True), (0.8, True), (0.7, True)]
        ap = evaluator._calculate_ap(scores)
        assert pytest.approx(ap, 0.01) == 1.0

    def test_calculate_ap_all_false(self, evaluator):
        """Test AP calculation with all false positives."""
        scores = [(0.9, False), (0.8, False), (0.7, False)]
        ap = evaluator._calculate_ap(scores)
        assert ap == 0.0

    def test_calculate_ap_empty(self, evaluator):
        """Test AP calculation with empty scores."""
        scores = []
        ap = evaluator._calculate_ap(scores)
        assert ap == 0.0

    def test_calculate_ap_mixed(self, evaluator):
        """Test AP calculation with mixed results."""
        # High score true, then false, then true
        scores = [(0.9, True), (0.7, False), (0.5, True)]
        ap = evaluator._calculate_ap(scores)
        assert 0 < ap < 1

    def test_map_at_multiple_thresholds(self, evaluator):
        """Test mAP at multiple IoU thresholds (COCO-style)."""
        predictions = [
            {"boxes": [[0, 0, 100, 100]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 100, 100]], "labels": ["person"]}]

        metrics = evaluator.calculate_map_at_thresholds(predictions, ground_truth)

        assert "mAP50" in metrics
        assert "mAP50-95" in metrics
        assert "per_threshold" in metrics
        assert metrics["mAP50"] <= 1.0 + 1e-6  # Allow small floating point error
        assert metrics["mAP50-95"] <= 1.0 + 1e-6

    def test_map_at_thresholds_custom(self, evaluator):
        """Test mAP at custom IoU thresholds."""
        predictions = [
            {"boxes": [[0, 0, 100, 100]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 100, 100]], "labels": ["person"]}]

        thresholds = [0.5, 0.75]
        metrics = evaluator.calculate_map_at_thresholds(
            predictions, ground_truth, iou_thresholds=thresholds
        )

        assert "mAP50" in metrics["per_threshold"]
        assert "mAP75" in metrics["per_threshold"]


class TestClassificationEvaluation:
    """Test classification evaluation methods."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_evaluate_classification_perfect(self, evaluator):
        """Test perfect classification."""
        predictions = [{"class": "cat"}, {"class": "dog"}, {"class": "cat"}]
        ground_truth = [{"class": "cat"}, {"class": "dog"}, {"class": "cat"}]

        metrics = evaluator.evaluate_classification(predictions, ground_truth)

        assert pytest.approx(metrics["accuracy"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["macro_precision"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["macro_recall"], abs=1e-6) == 1.0

    def test_evaluate_classification_some_errors(self, evaluator):
        """Test classification with errors."""
        predictions = [{"class": "cat"}, {"class": "cat"}, {"class": "dog"}]
        ground_truth = [{"class": "cat"}, {"class": "dog"}, {"class": "dog"}]

        metrics = evaluator.evaluate_classification(predictions, ground_truth)

        assert metrics["accuracy"] == pytest.approx(2 / 3, 0.01)
        assert metrics["correct"] == 2
        assert metrics["total"] == 3

    def test_evaluate_classification_confusion_matrix(self, evaluator):
        """Test confusion matrix generation."""
        predictions = [{"class": "A"}, {"class": "B"}, {"class": "A"}, {"class": "B"}]
        ground_truth = [{"class": "A"}, {"class": "A"}, {"class": "B"}, {"class": "B"}]

        metrics = evaluator.evaluate_classification(predictions, ground_truth)

        assert "confusion_matrix" in metrics
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape[0] == cm.shape[1]  # Square matrix

    def test_evaluate_classification_per_class(self, evaluator):
        """Test per-class metrics."""
        predictions = [{"class": "cat"}, {"class": "cat"}, {"class": "dog"}]
        ground_truth = [{"class": "cat"}, {"class": "dog"}, {"class": "dog"}]

        metrics = evaluator.evaluate_classification(predictions, ground_truth)

        assert "per_class" in metrics
        assert "cat" in metrics["per_class"]
        assert "dog" in metrics["per_class"]
        assert "precision" in metrics["per_class"]["cat"]
        assert "recall" in metrics["per_class"]["cat"]
        assert "f1" in metrics["per_class"]["cat"]


class TestSegmentationEvaluation:
    """Test segmentation evaluation methods."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_evaluate_segmentation_perfect(self, evaluator):
        """Test perfect segmentation."""
        mask = [[True, True], [True, True]]
        predictions = [{"mask": mask}]
        ground_truth = [{"mask": mask}]

        metrics = evaluator.evaluate_segmentation(predictions, ground_truth)

        assert pytest.approx(metrics["mIoU"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["dice_coefficient"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["pixel_accuracy"], abs=1e-6) == 1.0

    def test_evaluate_segmentation_no_overlap(self, evaluator):
        """Test segmentation with no overlap."""
        pred_mask = [[True, False], [False, False]]
        gt_mask = [[False, False], [False, True]]
        predictions = [{"mask": pred_mask}]
        ground_truth = [{"mask": gt_mask}]

        metrics = evaluator.evaluate_segmentation(predictions, ground_truth)

        assert metrics["mIoU"] == pytest.approx(0.0, abs=0.01)
        assert metrics["dice_coefficient"] == pytest.approx(0.0, abs=0.01)

    def test_evaluate_segmentation_partial(self, evaluator):
        """Test partial segmentation overlap."""
        pred_mask = [[True, True], [False, False]]
        gt_mask = [[True, False], [True, False]]
        predictions = [{"mask": pred_mask}]
        ground_truth = [{"mask": gt_mask}]

        metrics = evaluator.evaluate_segmentation(predictions, ground_truth)

        assert 0 < metrics["mIoU"] < 1
        assert 0 < metrics["dice_coefficient"] < 1

    def test_evaluate_segmentation_empty(self, evaluator):
        """Test segmentation with empty inputs."""
        predictions = [{"mask": []}]
        ground_truth = [{"mask": []}]

        metrics = evaluator.evaluate_segmentation(predictions, ground_truth)

        assert metrics["samples_evaluated"] == 0


class TestReportGeneration:
    """Test report generation methods."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_generate_report_json(self, evaluator):
        """Test JSON report generation."""
        # Create an evaluation first
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 50, 50]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)
        eval_id = metrics["evaluation_id"]

        report = evaluator.generate_report(eval_id, format="json")

        assert "evaluation_id" in report
        assert "type" in report
        assert "metrics" in report

    def test_generate_report_markdown(self, evaluator):
        """Test Markdown report generation."""
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 50, 50]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)
        eval_id = metrics["evaluation_id"]

        report = evaluator.generate_report(eval_id, format="markdown")

        assert "content" in report
        assert "# Evaluation Report" in report["content"]
        assert "Evaluation ID" in report["content"]

    def test_generate_report_with_visualizations(self, evaluator):
        """Test report with visualization data."""
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 50, 50]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)
        eval_id = metrics["evaluation_id"]

        report = evaluator.generate_report(eval_id, include_visualizations=True)

        assert "visualizations" in report
        assert "confusion_matrix" in report["visualizations"]

    def test_generate_report_not_found(self, evaluator):
        """Test report for non-existent evaluation."""
        report = evaluator.generate_report("nonexistent_id")

        assert "error" in report

    def test_get_evaluation_not_found(self, evaluator):
        """Test getting non-existent evaluation."""
        result = evaluator.get_evaluation("nonexistent_id")
        assert result is None


# Check if ultralytics is available
try:
    import ultralytics
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
class TestModelEvaluation:
    """Test model evaluation with YOLO (mocked).

    These tests require ultralytics to be installed.
    They will be skipped if ultralytics is not available.
    """

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_evaluate_model(self, evaluator):
        """Test model evaluation with mocked YOLO."""
        with patch("ultralytics.YOLO") as mock_yolo:
            # Setup mock
            mock_model = MagicMock()
            mock_results = MagicMock()
            mock_results.results_dict = {
                "metrics/mAP50(B)": 0.85,
                "metrics/mAP50-95(B)": 0.65,
                "metrics/precision(B)": 0.88,
                "metrics/recall(B)": 0.82,
            }
            mock_results.maps = None
            mock_results.confusion_matrix = None
            mock_results.speed = {"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0}
            mock_model.val.return_value = mock_results
            mock_yolo.return_value = mock_model

            metrics = evaluator.evaluate_model("/models/best.pt", "/data/dataset.yaml")

            assert metrics["mAP50"] == 0.85
            assert metrics["mAP50-95"] == 0.65
            assert metrics["precision"] == 0.88
            assert metrics["recall"] == 0.82
            assert "evaluation_id" in metrics

    def test_compare_models(self, evaluator):
        """Test model comparison."""
        with patch("ultralytics.YOLO") as mock_yolo:
            # Setup mock - create new mock model for each call
            mock_model1 = MagicMock()
            mock_model2 = MagicMock()

            # Different results for each model
            results1 = MagicMock()
            results1.results_dict = {
                "metrics/mAP50(B)": 0.85,
                "metrics/mAP50-95(B)": 0.65,
                "metrics/precision(B)": 0.88,
                "metrics/recall(B)": 0.82,
            }
            results1.maps = None
            results1.confusion_matrix = None
            results1.speed = {}

            results2 = MagicMock()
            results2.results_dict = {
                "metrics/mAP50(B)": 0.90,
                "metrics/mAP50-95(B)": 0.70,
                "metrics/precision(B)": 0.92,
                "metrics/recall(B)": 0.87,
            }
            results2.maps = None
            results2.confusion_matrix = None
            results2.speed = {}

            mock_model1.val.return_value = results1
            mock_model2.val.return_value = results2
            mock_yolo.side_effect = [mock_model1, mock_model2]

            comparison = evaluator.compare_models(
                ["/models/model1.pt", "/models/model2.pt"],
                "/data/dataset.yaml",
                ["mAP50", "mAP50-95"],
            )

            assert "models" in comparison
            assert "best" in comparison
            assert "ranking" in comparison
            assert len(comparison["models"]) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp results dir."""
        return ModelEvaluator(results_dir=str(tmp_path / "evaluations"))

    def test_collect_classes_empty(self, evaluator):
        """Test class collection with empty data."""
        predictions = [{"labels": []}]
        ground_truth = [{"labels": []}]

        classes = evaluator._collect_classes(predictions, ground_truth)
        assert classes == []

    def test_collect_classes_mixed(self, evaluator):
        """Test class collection from both sources."""
        predictions = [{"labels": ["cat", "dog"]}]
        ground_truth = [{"labels": ["dog", "bird"]}]

        classes = evaluator._collect_classes(predictions, ground_truth)
        assert set(classes) == {"cat", "dog", "bird"}
        assert classes == sorted(classes)  # Should be sorted

    def test_detection_wrong_class_match(self, evaluator):
        """Test that boxes with different classes don't match."""
        predictions = [
            {"boxes": [[0, 0, 100, 100]], "labels": ["car"], "scores": [0.9]}
        ]
        ground_truth = [{"boxes": [[0, 0, 100, 100]], "labels": ["person"]}]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        # Should not match even though boxes are identical
        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1

    def test_multiple_images(self, evaluator):
        """Test evaluation across multiple images."""
        predictions = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.9]},
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"], "scores": [0.8]},
            {"boxes": [], "labels": [], "scores": []},
        ]
        ground_truth = [
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"]},
            {"boxes": [[100, 100, 150, 150]], "labels": ["person"]},
            {"boxes": [[0, 0, 50, 50]], "labels": ["person"]},
        ]

        metrics = evaluator.evaluate_detection(predictions, ground_truth)

        # Image 1: 1 TP
        # Image 2: 1 FP, 1 FN (boxes don't match)
        # Image 3: 1 FN (no prediction)
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 2
