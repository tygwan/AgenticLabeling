"""Unit tests for Training Agent."""
import os
import sys
import json
import importlib.util
import pytest


def load_training_schemas():
    """Load schemas from training-agent service."""
    schemas_path = os.path.join(
        os.path.dirname(__file__),
        "../../services/training-agent/app/schemas.py"
    )
    spec = importlib.util.spec_from_file_location("training_schemas", schemas_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training_schemas = load_training_schemas()
TrainingConfig = training_schemas.TrainingConfig
TrainingStatus = training_schemas.TrainingStatus
TrainingResponse = training_schemas.TrainingResponse
RegistryTrainingRequest = training_schemas.RegistryTrainingRequest
InferenceRequest = training_schemas.InferenceRequest
InferenceResult = training_schemas.InferenceResult
InferenceResponse = training_schemas.InferenceResponse


class TestTrainingSchemas:
    """Test training schemas."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig(
            project_id="test_project",
            dataset_path="/data/test_dataset",
        )

        assert config.project_id == "test_project"
        assert config.dataset_path == "/data/test_dataset"
        assert config.model_size == "yolov8n"
        assert config.epochs == 100
        assert config.batch_size == 16
        assert config.image_size == 640
        assert config.augment is True
        assert config.patience == 50
        assert config.device == "0"

    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            project_id="custom_project",
            dataset_path="/data/custom",
            model_size="yolov8m",
            epochs=50,
            batch_size=32,
            image_size=1280,
            experiment_name="my_experiment",
            pretrained_weights="/models/pretrained.pt",
            augment=False,
            patience=20,
            device="cuda:1",
        )

        assert config.model_size == "yolov8m"
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.image_size == 1280
        assert config.experiment_name == "my_experiment"
        assert config.pretrained_weights == "/models/pretrained.pt"
        assert config.augment is False
        assert config.patience == 20
        assert config.device == "cuda:1"

    def test_registry_training_request(self):
        """Test RegistryTrainingRequest schema."""
        request = RegistryTrainingRequest(
            project_id="test_project",
            dataset_name="my_dataset",
            filter_categories=["person", "car"],
            min_confidence=0.8,
            only_validated=True,
        )

        assert request.project_id == "test_project"
        assert request.dataset_name == "my_dataset"
        assert request.format == "yolo"
        assert request.filter_categories == ["person", "car"]
        assert request.min_confidence == 0.8
        assert request.only_validated is True
        assert request.split_config == {"train": 0.8, "val": 0.1, "test": 0.1}

    def test_training_status_enum(self):
        """Test TrainingStatus enum values."""
        assert TrainingStatus.PENDING == "pending"
        assert TrainingStatus.RUNNING == "running"
        assert TrainingStatus.COMPLETED == "completed"
        assert TrainingStatus.FAILED == "failed"
        assert TrainingStatus.STOPPED == "stopped"

    def test_training_response(self):
        """Test TrainingResponse schema."""
        response = TrainingResponse(
            success=True,
            job_id="abc123",
            message="Training started",
            status="pending",
        )

        assert response.success is True
        assert response.job_id == "abc123"
        assert response.message == "Training started"
        assert response.status == "pending"
        assert response.error is None

    def test_training_response_error(self):
        """Test TrainingResponse with error."""
        response = TrainingResponse(
            success=False,
            error="Dataset not found",
        )

        assert response.success is False
        assert response.error == "Dataset not found"
        assert response.job_id is None

    def test_inference_request(self):
        """Test InferenceRequest schema."""
        request = InferenceRequest(
            model_path="/models/best.pt",
            image_path="/data/test.jpg",
            confidence=0.5,
            iou_threshold=0.6,
        )

        assert request.model_path == "/models/best.pt"
        assert request.image_path == "/data/test.jpg"
        assert request.confidence == 0.5
        assert request.iou_threshold == 0.6
        assert request.image_base64 is None

    def test_inference_request_base64(self):
        """Test InferenceRequest with base64 image."""
        request = InferenceRequest(
            model_path="/models/best.pt",
            image_base64="base64encodedimage==",
        )

        assert request.model_path == "/models/best.pt"
        assert request.image_base64 == "base64encodedimage=="
        assert request.image_path is None

    def test_inference_result(self):
        """Test InferenceResult schema."""
        result = InferenceResult(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=[100.0, 150.0, 200.0, 300.0],
        )

        assert result.class_id == 0
        assert result.class_name == "person"
        assert result.confidence == 0.95
        assert result.bbox == [100.0, 150.0, 200.0, 300.0]

    def test_inference_response_success(self):
        """Test InferenceResponse with successful detections."""
        detections = [
            InferenceResult(class_id=0, class_name="person", confidence=0.9, bbox=[10, 20, 30, 40]),
            InferenceResult(class_id=2, class_name="car", confidence=0.8, bbox=[100, 200, 150, 100]),
        ]

        response = InferenceResponse(
            success=True,
            detections=detections,
            inference_time_ms=25.5,
        )

        assert response.success is True
        assert len(response.detections) == 2
        assert response.inference_time_ms == 25.5
        assert response.error is None

    def test_inference_response_error(self):
        """Test InferenceResponse with error."""
        response = InferenceResponse(
            success=False,
            error="Model not found",
        )

        assert response.success is False
        assert response.error == "Model not found"
        assert response.detections is None


class TestTrainingConfigValidation:
    """Test training config validation."""

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(Exception):
            TrainingConfig()  # Missing required fields

    def test_model_dump(self):
        """Test model_dump method."""
        config = TrainingConfig(
            project_id="test",
            dataset_path="/data",
        )

        dump = config.model_dump()
        assert isinstance(dump, dict)
        assert dump["project_id"] == "test"
        assert dump["dataset_path"] == "/data"
        assert dump["model_size"] == "yolov8n"

    def test_model_json(self):
        """Test JSON serialization."""
        config = TrainingConfig(
            project_id="test",
            dataset_path="/data",
        )

        json_str = config.model_dump_json()
        data = json.loads(json_str)
        assert data["project_id"] == "test"


class TestRegistryTrainingRequest:
    """Test RegistryTrainingRequest schema."""

    def test_default_split_config(self):
        """Test default split config."""
        request = RegistryTrainingRequest(
            project_id="test",
            dataset_name="dataset1",
        )

        assert request.split_config["train"] == 0.8
        assert request.split_config["val"] == 0.1
        assert request.split_config["test"] == 0.1

    def test_custom_split_config(self):
        """Test custom split config."""
        request = RegistryTrainingRequest(
            project_id="test",
            dataset_name="dataset1",
            split_config={"train": 0.7, "val": 0.2, "test": 0.1},
        )

        assert request.split_config["train"] == 0.7
        assert request.split_config["val"] == 0.2

    def test_all_fields(self):
        """Test all fields together."""
        request = RegistryTrainingRequest(
            project_id="my_project",
            dataset_name="my_dataset",
            format="coco",
            model_size="yolov8l",
            epochs=200,
            batch_size=8,
            image_size=1024,
            filter_categories=["dog", "cat"],
            min_confidence=0.7,
            only_validated=True,
            split_config={"train": 0.9, "val": 0.05, "test": 0.05},
        )

        assert request.project_id == "my_project"
        assert request.dataset_name == "my_dataset"
        assert request.format == "coco"
        assert request.model_size == "yolov8l"
        assert request.epochs == 200
        assert request.batch_size == 8
        assert request.image_size == 1024
        assert request.filter_categories == ["dog", "cat"]
        assert request.min_confidence == 0.7
        assert request.only_validated is True
        assert request.split_config["train"] == 0.9
