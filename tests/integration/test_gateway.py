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
