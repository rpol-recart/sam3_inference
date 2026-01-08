"""Tests for image inference API endpoints."""
import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def sample_image_base64():
    """Create sample image as base64."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_segment_text_prompt(client: TestClient, sample_image_base64: str):
    """Test segmentation with text prompt."""
    response = client.post(
        "/api/v1/image/segment",
        json={
            "image": sample_image_base64,
            "prompts": [{"type": "text", "text": "person"}],
            "confidence_threshold": 0.5,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "masks" in data
    assert "boxes" in data
    assert "scores" in data
    assert "num_masks" in data
    assert "inference_time_ms" in data


def test_segment_box_prompt(client: TestClient, sample_image_base64: str):
    """Test segmentation with box prompt."""
    response = client.post(
        "/api/v1/image/segment",
        json={
            "image": sample_image_base64,
            "prompts": [
                {"type": "box", "box": [0.5, 0.5, 0.2, 0.2], "label": True}
            ],
            "confidence_threshold": 0.5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["boxes"], list)


def test_segment_combined_prompts(client: TestClient, sample_image_base64: str):
    """Test segmentation with combined prompts."""
    response = client.post(
        "/api/v1/image/segment",
        json={
            "image": sample_image_base64,
            "prompts": [
                {"type": "text", "text": "person"},
                {"type": "box", "box": [0.5, 0.5, 0.2, 0.2], "label": True},
            ],
            "confidence_threshold": 0.5,
        },
    )

    assert response.status_code == 200


def test_cached_features(client: TestClient, sample_image_base64: str):
    """Test cached features endpoint."""
    response = client.post(
        "/api/v1/image/cached-features",
        json={
            "image": sample_image_base64,
            "text_prompts": ["person", "car", "dog"],
            "confidence_threshold": 0.5,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 3
    assert "cache_hit" in data
    assert "inference_time_ms" in data


def test_invalid_image(client: TestClient):
    """Test with invalid base64 image."""
    response = client.post(
        "/api/v1/image/segment",
        json={
            "image": "invalid_base64",
            "prompts": [{"type": "text", "text": "person"}],
        },
    )

    assert response.status_code == 400


def test_empty_prompts(client: TestClient, sample_image_base64: str):
    """Test with empty prompts list."""
    response = client.post(
        "/api/v1/image/segment",
        json={
            "image": sample_image_base64,
            "prompts": [],
        },
    )

    # Should fail validation
    assert response.status_code == 422
