"""Integration tests for API Gateway."""
import os
import sys
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from io import BytesIO

# Add gateway path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../services/gateway"))

from fastapi.testclient import TestClient
from httpx import Response


class TestGatewayHealth:
    """Test Gateway health endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint exists."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "gateway" in data
        assert data["gateway"] == "healthy"

    def test_health_returns_service_status(self, client):
        """Test that health returns service status dict."""
        response = client.get("/health")
        data = response.json()
        assert "services" in data
        assert isinstance(data["services"], dict)


class TestGatewayEndpoints:
    """Test Gateway API endpoints structure."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "info" in schema

    def test_docs_available(self, client):
        """Test Swagger docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_detect_endpoint_requires_image(self, client):
        """Test detect endpoint requires image file."""
        response = client.post("/detect", data={"classes": "person"})
        assert response.status_code == 422  # Validation error

    def test_segment_endpoint_requires_image(self, client):
        """Test segment endpoint requires image file."""
        response = client.post("/segment", data={"boxes": "[]"})
        assert response.status_code == 422

    def test_export_endpoint_requires_dataset_name(self, client):
        """Test export endpoint requires dataset_name."""
        response = client.post("/export", data={})
        assert response.status_code == 422


class TestGatewayRouting:
    """Test Gateway routing to services."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_image(self):
        """Create a mock image file."""
        return ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")

    @patch("httpx.AsyncClient.post")
    def test_detect_routes_to_detection_service(self, mock_post, client, mock_image):
        """Test detect endpoint routes to detection-agent."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "detections": [
                {"label": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]}
            ]
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/detect",
            files={"image": mock_image},
            data={"classes": "person", "confidence": "0.5"}
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_segment_routes_to_segmentation_service(self, mock_post, client, mock_image):
        """Test segment endpoint routes to segmentation-agent."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "masks": []
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/segment",
            files={"image": mock_image},
            data={"boxes": "[[100, 100, 200, 300]]"}
        )

        assert response.status_code == 200


class TestRegistryEndpoints:
    """Test Registry-related Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.get")
    def test_registry_stats_endpoint(self, mock_get, client):
        """Test registry stats endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sources": 5,
            "objects": 10,
            "categories": 7
        }
        mock_get.return_value = mock_response

        response = client.get("/registry/stats")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_registry_objects_endpoint(self, mock_get, client):
        """Test registry objects listing endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": []
        }
        mock_get.return_value = mock_response

        response = client.get("/registry/objects")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_pending_objects_endpoint(self, mock_get, client):
        """Test pending objects endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": []
        }
        mock_get.return_value = mock_response

        response = client.get("/registry/objects/pending")
        assert response.status_code == 200


class TestTrainingEndpoints:
    """Test Training-related Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_train_start_requires_params(self, client):
        """Test train/start requires project_id and dataset_path."""
        response = client.post("/train/start", data={})
        assert response.status_code == 422

    @patch("httpx.AsyncClient.post")
    def test_train_start_routes_correctly(self, mock_post, client):
        """Test train/start routes to training-agent."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "job_id": "abc123",
            "status": "pending"
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/train/start",
            data={
                "project_id": "test_project",
                "dataset_path": "/data/test",
                "model_size": "yolov8n",
                "epochs": "10"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_train_jobs_endpoint(self, mock_get, client):
        """Test train/jobs endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jobs": [],
            "total": 0
        }
        mock_get.return_value = mock_response

        response = client.get("/train/jobs")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_models_endpoint(self, mock_get, client):
        """Test models listing endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "models": [],
            "count": 0
        }
        mock_get.return_value = mock_response

        response = client.get("/models")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_experiments_endpoint(self, mock_get, client):
        """Test experiments listing endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "experiments": []
        }
        mock_get.return_value = mock_response

        response = client.get("/experiments")
        assert response.status_code == 200


class TestExportEndpoints:
    """Test Export-related Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.post")
    def test_export_endpoint(self, mock_post, client):
        """Test export endpoint routes correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "output_path": "/data/exports/test_yolo"
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/export",
            data={
                "dataset_name": "test_dataset",
                "format": "yolo",
                "train_ratio": "0.8",
                "val_ratio": "0.1",
                "test_ratio": "0.1"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_exports_list_endpoint(self, mock_get, client):
        """Test exports listing endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "exports": []
        }
        mock_get.return_value = mock_response

        response = client.get("/exports")
        assert response.status_code == 200


class TestSearchEndpoints:
    """Test Search-related Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_similar_search_requires_embedding(self, client):
        """Test similar search requires embedding."""
        response = client.post("/search/similar", data={})
        assert response.status_code == 422

    @patch("httpx.AsyncClient.post")
    def test_similar_search_routes_correctly(self, mock_post, client):
        """Test similar search routes to registry."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "results": []
        }
        mock_post.return_value = mock_response

        # Create a fake 768-dim embedding
        fake_embedding = [0.1] * 768

        response = client.post(
            "/search/similar",
            data={
                "embedding": json.dumps(fake_embedding),
                "top_k": "10"
            }
        )

        assert response.status_code == 200


class TestValidationEndpoints:
    """Test Validation-related Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.patch")
    def test_validate_object_endpoint(self, mock_patch, client):
        """Test validate object endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_patch.return_value = mock_response

        response = client.post(
            "/registry/objects/obj_123/validate",
            data={
                "validated_by": "test_user",
                "quality_score": "0.95"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.delete")
    def test_reject_object_endpoint(self, mock_delete, client):
        """Test reject object endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_delete.return_value = mock_response

        response = client.post(
            "/registry/objects/obj_123/reject",
            data={"reason": "poor quality"}
        )

        assert response.status_code == 200


class TestPredictEndpoints:
    """Test Prediction-related Gateway endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_predict_requires_params(self, client):
        """Test predict requires model_path and image_path."""
        response = client.post("/predict", data={})
        assert response.status_code == 422

    @patch("httpx.AsyncClient.post")
    def test_predict_routes_correctly(self, mock_post, client):
        """Test predict routes to training-agent."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "detections": [],
            "inference_time_ms": 25.5
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/predict",
            data={
                "model_path": "/models/best.pt",
                "image_path": "/data/test.jpg",
                "confidence": "0.25"
            }
        )

        assert response.status_code == 200


class TestDetectAndSegmentPipeline:
    """Test detect_and_segment combined pipeline."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_image(self):
        """Create a mock image file."""
        return ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")

    @patch("httpx.AsyncClient.post")
    def test_detect_and_segment_success(self, mock_post, client, mock_image):
        """Test successful detect and segment pipeline."""
        # First call returns detection, second returns segmentation
        detection_response = MagicMock()
        detection_response.json.return_value = {
            "success": True,
            "data": {
                "boxes": [[100, 100, 50, 80]],
                "labels": ["person"],
                "scores": [0.95],
                "image_size": [640, 480]
            }
        }

        segment_response = MagicMock()
        segment_response.json.return_value = {
            "success": True,
            "data": {"masks": [{"mask": "base64data"}]}
        }

        mock_post.side_effect = [detection_response, segment_response]

        response = client.post(
            "/detect_and_segment",
            files={"image": mock_image},
            data={"classes": "person", "confidence": "0.5"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch("httpx.AsyncClient.post")
    def test_detect_and_segment_no_detections(self, mock_post, client, mock_image):
        """Test pipeline with no detections."""
        detection_response = MagicMock()
        detection_response.json.return_value = {
            "success": True,
            "data": {"boxes": [], "labels": [], "image_size": [640, 480]}
        }
        mock_post.return_value = detection_response

        response = client.post(
            "/detect_and_segment",
            files={"image": mock_image},
            data={"classes": "person", "confidence": "0.5"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["detections"] == []

    @patch("httpx.AsyncClient.post")
    def test_detect_and_segment_detection_failure(self, mock_post, client, mock_image):
        """Test pipeline when detection fails."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Model not loaded"
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/detect_and_segment",
            files={"image": mock_image},
            data={"classes": "person", "confidence": "0.5"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestAutoLabelPipeline:
    """Test auto_label full pipeline."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_image(self):
        """Create a mock image file."""
        return ("test.jpg", BytesIO(b"fake image data"), "image/jpeg")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_auto_label_full_pipeline(self, mock_get, mock_post, client, mock_image):
        """Test complete auto_label pipeline."""
        # Mock responses for each service call
        responses = [
            # Detection response
            MagicMock(json=lambda: {
                "success": True,
                "data": {
                    "boxes": [[100, 100, 50, 80]],
                    "labels": ["person"],
                    "scores": [0.95],
                    "image_size": {"width": 640, "height": 480}
                }
            }),
            # Segmentation response
            MagicMock(json=lambda: {
                "success": True,
                "data": {"masks": [{"mask": "base64mask"}]}
            }),
            # Save labels response
            MagicMock(json=lambda: {"success": True}),
            # Register source response
            MagicMock(json=lambda: {"success": True, "source_id": "src_001"}),
            # Batch register objects response
            MagicMock(json=lambda: {"success": True, "object_ids": ["obj_001"]})
        ]
        mock_post.side_effect = responses

        response = client.post(
            "/auto_label",
            files={"image": mock_image},
            data={
                "project_id": "test_project",
                "image_id": "img_001",
                "classes": "person,car",
                "confidence": "0.5",
                "save": "true",
                "register": "true"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["detections"] == 1

    @patch("httpx.AsyncClient.post")
    def test_auto_label_no_detections(self, mock_post, client, mock_image):
        """Test auto_label with no detections."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "boxes": [],
                "labels": [],
                "scores": [],
                "image_size": [640, 480]
            }
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/auto_label",
            files={"image": mock_image},
            data={
                "project_id": "test_project",
                "image_id": "img_001",
                "classes": "person",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["detections"] == 0

    @patch("httpx.AsyncClient.post")
    def test_auto_label_detection_failure(self, mock_post, client, mock_image):
        """Test auto_label when detection fails."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": False, "error": "Detection error"}
        mock_post.return_value = mock_response

        response = client.post(
            "/auto_label",
            files={"image": mock_image},
            data={
                "project_id": "test_project",
                "image_id": "img_001",
                "classes": "person",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestRegistryCategories:
    """Test registry categories endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.get")
    def test_registry_categories_endpoint(self, mock_get, client):
        """Test categories listing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "categories": ["person", "car", "dog"]
        }
        mock_get.return_value = mock_response

        response = client.get("/registry/categories")
        assert response.status_code == 200


class TestTrainFromRegistry:
    """Test train from registry endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.post")
    def test_train_from_registry_endpoint(self, mock_post, client):
        """Test train from registry endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "job_id": "train_001",
            "status": "started"
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/train/from-registry",
            data={
                "project_id": "test_project",
                "dataset_name": "test_dataset",
                "model_size": "yolov8n",
                "epochs": "100"
            }
        )

        assert response.status_code == 200


class TestTrainingControl:
    """Test training job control endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.get")
    def test_training_status_endpoint(self, mock_get, client):
        """Test training status endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "job_id": "train_001",
            "status": "running",
            "progress": 0.5
        }
        mock_get.return_value = mock_response

        response = client.get("/train/status/train_001")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_stop_training_endpoint(self, mock_post, client):
        """Test stop training endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "message": "Training stopped"
        }
        mock_post.return_value = mock_response

        response = client.post("/train/stop/train_001")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_get_model_endpoint(self, mock_get, client):
        """Test get model details endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "model": {"id": "model_001", "path": "/models/best.pt"}
        }
        mock_get.return_value = mock_response

        response = client.get("/models/model_001")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_evaluate_model_endpoint(self, mock_post, client):
        """Test model evaluation endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "mAP50": 0.85
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/models/evaluate",
            data={
                "model_path": "/models/best.pt",
                "data_path": "/data/test.yaml"
            }
        )

        assert response.status_code == 200


class TestExperimentEndpoints:
    """Test experiment-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.get")
    def test_experiment_runs_endpoint(self, mock_get, client):
        """Test experiment runs endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "runs": []
        }
        mock_get.return_value = mock_response

        response = client.get("/experiments/test_experiment/runs")
        assert response.status_code == 200


class TestEvaluationEndpoints:
    """Test evaluation-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.post")
    def test_evaluate_detection_endpoint(self, mock_post, client):
        """Test detection evaluation endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "mAP50": 0.85
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/detection",
            data={
                "predictions": "[]",
                "ground_truth": "[]",
                "iou_threshold": "0.5"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_evaluate_detection_coco_endpoint(self, mock_post, client):
        """Test COCO-style detection evaluation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "mAP50-95": 0.65
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/detection/coco",
            data={
                "predictions": "[]",
                "ground_truth": "[]"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_evaluate_classification_endpoint(self, mock_post, client):
        """Test classification evaluation endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "accuracy": 0.95
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/classification",
            data={
                "predictions": "[]",
                "ground_truth": "[]"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_evaluate_segmentation_endpoint(self, mock_post, client):
        """Test segmentation evaluation endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "mean_iou": 0.75
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/segmentation",
            data={
                "predictions": "[]",
                "ground_truth": "[]"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_evaluate_batch_endpoint(self, mock_post, client):
        """Test batch evaluation endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "results": []
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/batch",
            data={
                "predictions": "[]",
                "ground_truth": "[]",
                "task": "detection"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_evaluate_model_on_dataset_endpoint(self, mock_post, client):
        """Test model evaluation on dataset endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "metrics": {"mAP50": 0.85}
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/model",
            data={
                "model_path": "/models/best.pt",
                "dataset_path": "/data/test.yaml",
                "task": "detection"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_compare_models_endpoint(self, mock_post, client):
        """Test model comparison endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "comparison": []
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluate/compare",
            data={
                "model_paths": '["model1.pt", "model2.pt"]',
                "dataset_path": "/data/test.yaml"
            }
        )

        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_list_evaluation_metrics_endpoint(self, mock_get, client):
        """Test list evaluation metrics endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "metrics": {"detection": ["mAP50", "mAP50-95"]}
        }
        mock_get.return_value = mock_response

        response = client.get("/evaluate/metrics")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_get_evaluation_endpoint(self, mock_get, client):
        """Test get evaluation by ID endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "evaluation": {"id": "eval_001"}
        }
        mock_get.return_value = mock_response

        response = client.get("/evaluations/eval_001")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.get")
    def test_get_evaluation_report_endpoint(self, mock_get, client):
        """Test get evaluation report endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "report": {}
        }
        mock_get.return_value = mock_response

        response = client.get("/evaluations/eval_001/report")
        assert response.status_code == 200

    @patch("httpx.AsyncClient.post")
    def test_generate_evaluation_report_endpoint(self, mock_post, client):
        """Test generate evaluation report endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "report": {}
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/evaluations/eval_001/report/generate",
            data={"format": "json"}
        )

        assert response.status_code == 200


class TestDownloadEndpoints:
    """Test download-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.get")
    def test_download_dataset_not_found(self, mock_get, client):
        """Test download dataset not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        response = client.get("/export/nonexistent/download")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    @patch("httpx.AsyncClient.get")
    def test_download_dataset_success(self, mock_get, client):
        """Test successful dataset download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"PK\x03\x04..."  # Fake zip content
        mock_get.return_value = mock_response

        response = client.get("/export/test_dataset/download?format=yolo")
        assert response.status_code == 200


class TestSimilarObjectSearch:
    """Test similar object search endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @patch("httpx.AsyncClient.get")
    def test_search_similar_to_object_endpoint(self, mock_get, client):
        """Test similar object search endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {"object_id": "obj_001", "embedding": None}
        }
        mock_get.return_value = mock_response

        response = client.post(
            "/search/similar_to_object",
            data={"object_id": "obj_001", "top_k": "10"}
        )

        assert response.status_code == 200

    def test_search_similar_invalid_embedding(self, client):
        """Test similar search with invalid embedding format."""
        response = client.post(
            "/search/similar",
            data={"embedding": "not-valid-json", "top_k": "10"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestUpdateObjectEndpoint:
    """Test update object endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_update_object_info_endpoint(self, client):
        """Test update object returns info."""
        response = client.patch("/registry/objects/obj_001")
        assert response.status_code == 200
        data = response.json()
        assert "example" in data
