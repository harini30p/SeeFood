"""Integration and unit tests for SeeFood FastAPI endpoints and uncertainty calibration logic.

Covers:
- GET / (Welcome and documentation links)
- GET /health (Health check, model status, and threshold metadata)
- POST /predict (Valid JPEG, PNG, and WebP images with 3-class support: Hotdog, Not Hotdog, Uncertain)
- POST /predict edge cases (Corrupt files, empty uploads, unsupported formats, missing fields)
- Response schema validation including null confidence for Uncertain results
- Unit tests for threshold decision logic and configuration loading edge cases
"""

import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.inference import classify_probability, load_threshold_config, DEFAULT_THRESHOLDS


class TestRootEndpoint:
    """Tests for the root GET / endpoint."""

    def test_root_returns_ok_and_expected_fields(self, client: TestClient):
        """GET / should return HTTP 200 with welcome message and docs links."""
        response = client.get("/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "message" in data, "Response should include 'message'"
        assert "docs_url" in data, "Response should include 'docs_url'"
        assert "redoc_url" in data, "Response should include 'redoc_url'"
        assert data["docs_url"] == "/docs"
        assert data["redoc_url"] == "/redoc"


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_check_returns_ok_model_and_thresholds(self, client: TestClient):
        """GET /health should return HTTP 200, model status, and threshold bounds."""
        response = client.get("/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "healthy", "Expected status to be 'healthy'"
        assert data.get("model_loaded") is True, "Expected 'model_loaded' to be True"
        assert "thresholds" in data, "Expected 'thresholds' metadata in health response"
        thresholds = data["thresholds"]
        assert "low" in thresholds
        assert "high" in thresholds
        assert thresholds.get("status") == "provisional"


class TestPredictEndpointValidImages:
    """Tests for POST /predict with valid supported image formats."""

    def test_predict_valid_jpeg(self, client: TestClient, create_image_bytes):
        """POST /predict with a valid JPEG image returns 200 and complete schema."""
        jpeg_bytes = create_image_bytes(fmt="JPEG", size=(224, 224), color="orange")
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        self._validate_prediction_schema(data)

    def test_predict_valid_png(self, client: TestClient, create_image_bytes):
        """POST /predict with a valid PNG image returns 200 and complete schema."""
        png_bytes = create_image_bytes(fmt="PNG", size=(150, 150), color="green")
        response = client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        self._validate_prediction_schema(data)

    def test_predict_valid_webp(self, client: TestClient, create_image_bytes):
        """POST /predict with a valid WebP image returns 200 and complete schema."""
        webp_bytes = create_image_bytes(fmt="WEBP", size=(300, 200), color="yellow")
        response = client.post(
            "/predict",
            files={"file": ("test.webp", io.BytesIO(webp_bytes), "image/webp")},
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        self._validate_prediction_schema(data)

    @staticmethod
    def _validate_prediction_schema(data: dict):
        """Helper to assert prediction payload meets contract requirements with 3-class support."""
        assert "label" in data, "Response must contain 'label'"
        assert "hotdog_probability" in data, "Response must contain 'hotdog_probability'"
        assert "confidence" in data, "Response must contain 'confidence'"
        assert "gradcam_base64" in data, "Response must contain 'gradcam_base64'"

        # Label must be one of three outcomes
        assert data["label"] in ("Hotdog", "Not Hotdog", "Uncertain"), (
            f"Label must be 'Hotdog', 'Not Hotdog', or 'Uncertain', got '{data['label']}'"
        )

        # hotdog_probability must always be a float in [0, 1]
        assert isinstance(data["hotdog_probability"], (int, float)), "hotdog_probability must be numeric"
        assert 0.0 <= data["hotdog_probability"] <= 1.0, (
            f"hotdog_probability must be in [0, 1], got {data['hotdog_probability']}"
        )

        # Confidence handling: numeric percentage for decisive labels; null for Uncertain
        if data["label"] == "Uncertain":
            assert data["confidence"] is None, (
                f"Confidence must be null for Uncertain predictions, got {data['confidence']}"
            )
        else:
            assert isinstance(data["confidence"], (int, float)), "confidence must be numeric for decisive labels"
            assert 0.0 <= data["confidence"] <= 100.0, (
                f"confidence must be in [0, 100], got {data['confidence']}"
            )

        # Grad-CAM data URL validation
        gradcam = data["gradcam_base64"]
        assert isinstance(gradcam, str), "gradcam_base64 must be a string"
        assert gradcam.startswith("data:image/jpeg;base64,"), (
            "gradcam_base64 must be a JPEG data URL starting with 'data:image/jpeg;base64,'"
        )
        assert len(gradcam) > len("data:image/jpeg;base64,"), "gradcam_base64 must not be empty"


class TestPredictEndpointErrorHandling:
    """Tests for POST /predict error handling and edge cases."""

    def test_predict_corrupt_image_returns_400(self, client: TestClient):
        """Corrupt or non-image bytes should return HTTP 400 Bad Request."""
        corrupt_bytes = b"This is not a valid image content at all."
        response = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")},
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        assert response.json().get("detail") == "Uploaded file is not a valid or readable image."

    def test_predict_empty_file_returns_400(self, client: TestClient):
        """Empty file upload should return HTTP 400 Bad Request."""
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        assert response.json().get("detail") == "The uploaded file is empty."

    def test_predict_unsupported_format_returns_415(self, client: TestClient, create_image_bytes):
        """Unsupported image format (e.g. GIF) should return HTTP 415 Unsupported Media Type."""
        gif_bytes = create_image_bytes(fmt="GIF", size=(100, 100), color="blue")
        response = client.post(
            "/predict",
            files={"file": ("animation.gif", io.BytesIO(gif_bytes), "image/gif")},
        )
        assert response.status_code == 415, f"Expected 415, got {response.status_code}"
        detail = response.json().get("detail", "")
        assert "Unsupported image format" in detail, f"Expected format error in detail, got: {detail}"

    def test_predict_missing_file_field_returns_422(self, client: TestClient):
        """Calling /predict without multipart file field should return HTTP 422 Unprocessable Entity."""
        response = client.post("/predict")
        assert response.status_code == 422, f"Expected 422 for missing required file, got {response.status_code}"


class TestThresholdDecisionLogic:
    """Unit tests for the dual-threshold classification logic."""

    def test_classify_above_upper_threshold(self):
        """Probability >= threshold_high should return Hotdog with positive confidence."""
        label, conf = classify_probability(0.65, threshold_low=0.35, threshold_high=0.65)
        assert label == "Hotdog"
        assert conf == pytest.approx(65.0)

        label, conf = classify_probability(0.92, threshold_low=0.35, threshold_high=0.65)
        assert label == "Hotdog"
        assert conf == pytest.approx(92.0)

    def test_classify_below_lower_threshold(self):
        """Probability <= threshold_low should return Not Hotdog with inverted confidence."""
        label, conf = classify_probability(0.35, threshold_low=0.35, threshold_high=0.65)
        assert label == "Not Hotdog"
        assert conf == pytest.approx(65.0)  # (1 - 0.35) * 100

        label, conf = classify_probability(0.05, threshold_low=0.35, threshold_high=0.65)
        assert label == "Not Hotdog"
        assert conf == pytest.approx(95.0)

    def test_classify_in_uncertain_band(self):
        """Probability inside (threshold_low, threshold_high) must return Uncertain and null confidence."""
        label, conf = classify_probability(0.50, threshold_low=0.35, threshold_high=0.65)
        assert label == "Uncertain"
        assert conf is None

        label, conf = classify_probability(0.351, threshold_low=0.35, threshold_high=0.65)
        assert label == "Uncertain"
        assert conf is None

        label, conf = classify_probability(0.649, threshold_low=0.35, threshold_high=0.65)
        assert label == "Uncertain"
        assert conf is None


class TestThresholdConfigurationLoading:
    """Unit tests for loading, validating, and falling back on threshold configs."""

    def test_load_valid_config(self, tmp_path: Path):
        """Valid JSON should load numeric thresholds successfully."""
        cfg_file = tmp_path / "valid.json"
        cfg_file.write_text(json.dumps({
            "threshold_low": 0.4,
            "threshold_high": 0.6,
            "status": "provisional",
            "calibrated_on_validation_set": False
        }))
        res = load_threshold_config(cfg_file)
        assert res["threshold_low"] == 0.4
        assert res["threshold_high"] == 0.6
        assert res["status"] == "provisional"

    def test_load_missing_file_fallback(self, tmp_path: Path):
        """Missing file should fall back to provisional defaults without error."""
        nonexistent = tmp_path / "does_not_exist.json"
        res = load_threshold_config(nonexistent)
        assert res == DEFAULT_THRESHOLDS

    def test_load_invalid_range_fallback(self, tmp_path: Path):
        """Thresholds with low >= high or out of range should fall back to provisional defaults."""
        invalid_cfg = tmp_path / "inverted.json"
        invalid_cfg.write_text(json.dumps({
            "threshold_low": 0.8,
            "threshold_high": 0.2
        }))
        res = load_threshold_config(invalid_cfg)
        assert res == DEFAULT_THRESHOLDS
