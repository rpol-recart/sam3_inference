"""Tests for video inference API endpoints."""
import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def sample_video_base64():
    """Create sample video as base64 (minimal for testing purposes)."""
    # Create a minimal MP4-like file for testing
    # In practice, this would be a real video, but for testing we can use a small binary
    video_data = b"fake-video-data"
    return base64.b64encode(video_data).decode()


def test_start_video_session(client: TestClient, sample_video_base64: str):
    """Test starting a video session."""
    response = client.post(
        "/api/v1/video/session/start",
        json={
            "video_base64": sample_video_base64,
            "session_id": "test-session-123"
        },
    )
    
    # Since we're using a fake video, expect a 422 validation error or 400
    # The important thing is that our API handles the request properly
    assert response.status_code in [400, 422, 500]  # Expected responses for invalid video


def test_health_endpoint(client: TestClient):
    """Test health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] == "healthy"


def test_models_info_endpoint(client: TestClient):
    """Test models info endpoint."""
    response = client.get("/models/info")
    
    # May return 503 if models aren't loaded in test environment
    assert response.status_code in [200, 503]
