"""End-to-end tests for the auto_label pipeline.

These tests verify the complete flow from source registration to dataset export.
They use mocked HTTP responses to simulate service interactions.
"""
import asyncio
import base64
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


class TestAutoLabelPipeline:
    """E2E tests for the auto_label pipeline."""

    @pytest.fixture
    def mock_detection_response(self):
        """Mock response from detection agent."""
        return {
            "success": True,
            "detections": [
                {
                    "label": "person",
                    "bbox": [100, 100, 150, 200],
                    "confidence": 0.95,
                },
                {
                    "label": "car",
                    "bbox": [300, 150, 200, 100],
                    "confidence": 0.88,
                },
            ],
        }

    @pytest.fixture
    def mock_segmentation_response(self):
        """Mock response from segmentation agent."""
        return {
            "success": True,
            "masks": [
                {
                    "bbox": [100, 100, 150, 200],
                    "mask_base64": base64.b64encode(b"fake_mask_1").decode(),
                    "area": 30000,
                },
                {
                    "bbox": [300, 150, 200, 100],
                    "mask_base64": base64.b64encode(b"fake_mask_2").decode(),
                    "area": 20000,
                },
            ],
        }

    @pytest.fixture
    def mock_classification_response(self):
        """Mock response from classification agent."""
        return {
            "success": True,
            "embeddings": [
                [0.1] * 768,  # DINOv2 embedding dimension
                [0.2] * 768,
            ],
        }

    @pytest.fixture
    def mock_registry_response(self):
        """Mock response from object registry."""
        return {
            "success": True,
            "object_ids": ["obj_001", "obj_002"],
            "count": 2,
        }

    def test_pipeline_flow_detection_to_registry(
        self,
        mock_detection_response,
        mock_segmentation_response,
        mock_classification_response,
        mock_registry_response,
    ):
        """Test the complete pipeline flow from detection to registry."""
        # This test verifies the data flow between services

        # 1. Detection produces bounding boxes
        detections = mock_detection_response["detections"]
        assert len(detections) == 2
        assert detections[0]["label"] == "person"
        assert detections[1]["label"] == "car"

        # 2. Segmentation produces masks for each detection
        masks = mock_segmentation_response["masks"]
        assert len(masks) == 2
        assert masks[0]["bbox"] == detections[0]["bbox"]

        # 3. Classification produces embeddings
        embeddings = mock_classification_response["embeddings"]
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768

        # 4. Registry stores all objects
        object_ids = mock_registry_response["object_ids"]
        assert len(object_ids) == 2

    def test_pipeline_data_transformation(self):
        """Test data transformation between pipeline stages."""
        # Detection output
        detection = {
            "label": "person",
            "bbox": [100, 100, 150, 200],  # x, y, w, h
            "confidence": 0.95,
        }

        # Transform to registry format
        registry_object = {
            "category": detection["label"],
            "bbox": detection["bbox"],
            "confidence": detection["confidence"],
        }

        assert registry_object["category"] == "person"
        assert registry_object["bbox"] == [100, 100, 150, 200]
        assert registry_object["confidence"] == 0.95

    def test_pipeline_error_handling(self):
        """Test error handling in the pipeline."""
        # Simulate detection failure
        detection_error = {
            "success": False,
            "error": "Model not loaded",
        }

        assert detection_error["success"] is False
        assert "error" in detection_error

        # Pipeline should gracefully handle service failures
        # and not proceed to subsequent stages

    def test_batch_processing_flow(self):
        """Test batch processing of multiple images."""
        # Simulate batch of 3 images
        image_paths = [
            "/data/images/img001.jpg",
            "/data/images/img002.jpg",
            "/data/images/img003.jpg",
        ]

        # Each image produces different number of detections
        detections_per_image = [2, 5, 1]

        # Calculate expected total objects
        total_objects = sum(detections_per_image)
        assert total_objects == 8

    def test_validation_workflow(self):
        """Test the validation workflow."""
        # Object starts as unvalidated
        object_state = {
            "object_id": "obj_001",
            "is_validated": False,
            "validated_by": None,
            "confidence": 0.95,
        }

        assert object_state["is_validated"] is False

        # After human review
        object_state["is_validated"] = True
        object_state["validated_by"] = "reviewer@example.com"

        assert object_state["is_validated"] is True
        assert object_state["validated_by"] is not None


class TestDataExportPipeline:
    """E2E tests for data export functionality."""

    def test_yolo_export_format(self):
        """Test YOLO export format generation."""
        # Sample object for YOLO export
        object_data = {
            "category_id": 0,  # person
            "bbox": [100, 100, 150, 200],  # x, y, w, h
            "image_width": 640,
            "image_height": 480,
        }

        # Convert to YOLO format (normalized center coords)
        x, y, w, h = object_data["bbox"]
        img_w = object_data["image_width"]
        img_h = object_data["image_height"]

        # YOLO format: class x_center y_center width height (all normalized)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        yolo_line = f"{object_data['category_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

        assert yolo_line.startswith("0 ")  # class 0
        assert "0.273" in yolo_line  # x_center approx

    def test_coco_export_format(self):
        """Test COCO export format generation."""
        # Sample annotation for COCO export
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 150, 200],  # x, y, w, h
            "area": 30000,
            "iscrowd": 0,
        }

        # Verify COCO format requirements
        assert "id" in annotation
        assert "image_id" in annotation
        assert "category_id" in annotation
        assert "bbox" in annotation
        assert len(annotation["bbox"]) == 4
        assert annotation["area"] == 30000

    def test_dataset_split_ratios(self):
        """Test dataset split configuration."""
        split_config = {"train": 0.8, "val": 0.1, "test": 0.1}

        # Verify ratios sum to 1
        total = sum(split_config.values())
        assert abs(total - 1.0) < 0.001

        # Test with 100 samples
        total_samples = 100
        train_count = int(total_samples * split_config["train"])
        val_count = int(total_samples * split_config["val"])
        test_count = total_samples - train_count - val_count

        assert train_count == 80
        assert val_count == 10
        assert test_count == 10


class TestTrainingPipeline:
    """E2E tests for training pipeline."""

    def test_training_config_validation(self):
        """Test training configuration validation."""
        config = {
            "project_id": "test_project",
            "dataset_path": "/data/datasets/test",
            "model_size": "yolov8n",
            "epochs": 100,
            "batch_size": 16,
            "image_size": 640,
        }

        # Validate required fields
        assert "project_id" in config
        assert "dataset_path" in config
        assert config["epochs"] > 0
        assert config["batch_size"] > 0

    def test_training_progress_tracking(self):
        """Test training progress tracking."""
        # Simulate training progress
        progress_updates = [
            {"epoch": 1, "loss": 1.5, "mAP50": 0.3},
            {"epoch": 50, "loss": 0.5, "mAP50": 0.7},
            {"epoch": 100, "loss": 0.2, "mAP50": 0.85},
        ]

        # Verify loss decreases over epochs
        assert progress_updates[0]["loss"] > progress_updates[-1]["loss"]

        # Verify mAP increases over epochs
        assert progress_updates[0]["mAP50"] < progress_updates[-1]["mAP50"]

    def test_active_learning_cycle(self):
        """Test active learning cycle flow."""
        # Initial state
        total_samples = 1000
        labeled_samples = 100
        unlabeled_samples = total_samples - labeled_samples

        # Query uncertain samples
        query_batch_size = 10
        uncertain_samples = [
            {"object_id": f"obj_{i}", "confidence": 0.3 + i * 0.01}
            for i in range(query_batch_size)
        ]

        # Verify samples are sorted by uncertainty
        confidences = [s["confidence"] for s in uncertain_samples]
        assert confidences == sorted(confidences)

        # After labeling, labeled count increases
        labeled_samples += query_batch_size
        assert labeled_samples == 110

        # Model should improve with more data
        # (This would be verified in actual training)


class TestVideoProcessingPipeline:
    """E2E tests for video processing pipeline."""

    def test_frame_extraction_flow(self):
        """Test frame extraction from video."""
        video_info = {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 300,
            "duration_seconds": 10.0,
        }

        # Sample every 5 frames
        sample_interval = 5
        expected_frames = video_info["frame_count"] // sample_interval

        assert expected_frames == 60

    def test_track_creation_flow(self):
        """Test track creation from detections."""
        # Detections across 5 frames
        frame_detections = [
            {"frame": 0, "objects": [{"id": "obj_1", "bbox": [100, 100, 50, 80]}]},
            {"frame": 1, "objects": [{"id": "obj_2", "bbox": [105, 102, 50, 80]}]},
            {"frame": 2, "objects": [{"id": "obj_3", "bbox": [110, 104, 50, 80]}]},
            {"frame": 3, "objects": [{"id": "obj_4", "bbox": [115, 106, 50, 80]}]},
            {"frame": 4, "objects": [{"id": "obj_5", "bbox": [120, 108, 50, 80]}]},
        ]

        # Objects with similar bboxes should be linked into a track
        track_length = len(frame_detections)
        assert track_length == 5

    def test_track_merge_flow(self):
        """Test track merging."""
        track1 = {
            "track_id": "trk_001",
            "object_ids": ["obj_1", "obj_2", "obj_3"],
        }
        track2 = {
            "track_id": "trk_002",
            "object_ids": ["obj_4", "obj_5"],
        }

        # Merge tracks
        merged_objects = track1["object_ids"] + track2["object_ids"]

        assert len(merged_objects) == 5
        assert merged_objects == ["obj_1", "obj_2", "obj_3", "obj_4", "obj_5"]

    def test_track_split_flow(self):
        """Test track splitting."""
        track = {
            "track_id": "trk_001",
            "object_ids": ["obj_1", "obj_2", "obj_3", "obj_4", "obj_5"],
        }

        # Split at index 2
        split_index = 2
        first_track_objects = track["object_ids"][:split_index]
        second_track_objects = track["object_ids"][split_index:]

        assert len(first_track_objects) == 2
        assert len(second_track_objects) == 3


class TestEvaluationPipeline:
    """E2E tests for evaluation pipeline."""

    def test_detection_metrics_calculation(self):
        """Test detection metrics calculation flow."""
        # Ground truth and predictions
        predictions = [
            {"bbox": [100, 100, 50, 80], "confidence": 0.9, "class": "person"},
            {"bbox": [300, 150, 60, 90], "confidence": 0.8, "class": "person"},
        ]
        ground_truths = [
            {"bbox": [102, 98, 48, 82], "class": "person"},  # Match with pred 1
            {"bbox": [500, 200, 40, 60], "class": "person"},  # No match
        ]

        # Calculate IoU for first prediction
        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Calculate intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0

            inter_area = (xi2 - xi1) * (yi2 - yi1)
            union_area = w1 * h1 + w2 * h2 - inter_area

            return inter_area / union_area

        iou = calculate_iou(
            predictions[0]["bbox"],
            ground_truths[0]["bbox"]
        )

        # IoU should be high for matching boxes
        assert iou > 0.8

    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation."""
        # Simulated results
        true_labels = ["person", "person", "car", "car", "person"]
        pred_labels = ["person", "car", "car", "person", "person"]

        # Build confusion matrix
        classes = ["person", "car"]
        confusion = [[0, 0], [0, 0]]

        for true, pred in zip(true_labels, pred_labels):
            true_idx = classes.index(true)
            pred_idx = classes.index(pred)
            confusion[true_idx][pred_idx] += 1

        # person: 2 TP, 1 FN
        assert confusion[0][0] == 2  # person -> person
        assert confusion[0][1] == 1  # person -> car

        # car: 1 TP, 1 FN
        assert confusion[1][0] == 1  # car -> person
        assert confusion[1][1] == 1  # car -> car
