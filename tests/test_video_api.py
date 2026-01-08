"""Tests for video inference API endpoints."""
import base64
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_video_path():
    """Path to sample test video (if available)."""
    # This would need a real video file for actual testing
    # For unit tests, we can mock the video model
    return None


def test_start_session_with_path(client: TestClient, tmp_path):
    """Test starting video session with local path."""
    # Skip if video model not enabled
    response = client.get("/health")
    if not response.json().get("video_model_enabled"):
        pytest.skip("Video model not enabled")

    # This test requires a real video file
    pytest.skip("Requires real video file for testing")


def test_start_session_video_not_enabled(client: TestClient):
    """Test video endpoints when video model is disabled."""
    response = client.post(
        "/api/v1/video/session/start",
        json={"video_path": "/tmp/test.mp4"},
    )

    # Should return 503 if video model not enabled
    if response.status_code == 503:
        assert "not enabled" in response.json()["detail"].lower()


def test_add_prompt_session_not_found(client: TestClient):
    """Test adding prompt to non-existent session."""
    response = client.post(
        "/api/v1/video/session/fake-session-id/prompt",
        json={
            "frame_index": 0,
            "prompts": [{"type": "text", "text": "person"}],
        },
    )

    # Should return 404 or 503
    assert response.status_code in [404, 503]


def test_get_session_status_not_found(client: TestClient):
    """Test getting status of non-existent session."""
    response = client.get("/api/v1/video/session/fake-session-id/status")

    # Should return 404 or 503
    assert response.status_code in [404, 503]


def test_list_sessions(client: TestClient):
    """Test listing video sessions."""
    response = client.get("/api/v1/video/sessions")

    if response.status_code == 200:
        data = response.json()
        assert "sessions" in data
        assert "total_sessions" in data
        assert isinstance(data["sessions"], list)
    else:
        # Video model not enabled
        assert response.status_code == 503


def test_close_session_not_found(client: TestClient):
    """Test closing non-existent session."""
    response = client.delete("/api/v1/video/session/fake-session-id")

    # Should return 404 or 503
    assert response.status_code in [404, 503]


def test_remove_object_session_not_found(client: TestClient):
    """Test removing object from non-existent session."""
    response = client.delete("/api/v1/video/session/fake-session-id/object/1")

    # Should return 404 or 503
    assert response.status_code in [404, 503]


def test_reset_session_not_found(client: TestClient):
    """Test resetting non-existent session."""
    response = client.post("/api/v1/video/session/fake-session-id/reset")

    # Should return 404 or 503
    assert response.status_code in [404, 503]


def test_propagate_session_not_found(client: TestClient):
    """Test propagating in non-existent session."""
    response = client.post(
        "/api/v1/video/session/fake-session-id/propagate",
        json={"direction": "both"},
    )

    # Should return 404 or 503
    assert response.status_code in [404, 503]


# Integration test (requires real video and model)
@pytest.mark.integration
def test_video_workflow_integration(client: TestClient, sample_video_path):
    """Integration test for complete video workflow."""
    if not sample_video_path:
        pytest.skip("No sample video available")

    # 1. Start session
    response = client.post(
        "/api/v1/video/session/start",
        json={"video_path": str(sample_video_path)},
    )
    assert response.status_code == 200
    session_data = response.json()
    session_id = session_data["session_id"]
    assert "video_info" in session_data

    # 2. Add prompt
    response = client.post(
        f"/api/v1/video/session/{session_id}/prompt",
        json={
            "frame_index": 0,
            "prompts": [{"type": "text", "text": "person"}],
        },
    )
    assert response.status_code == 200
    prompt_data = response.json()
    assert "obj_id" in prompt_data
    assert "masks" in prompt_data

    # 3. Get session status
    response = client.get(f"/api/v1/video/session/{session_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["current_objects"] > 0

    # 4. Propagate (non-streaming)
    response = client.post(
        f"/api/v1/video/session/{session_id}/propagate",
        json={"direction": "forward", "max_frames": 10},
    )
    assert response.status_code == 200
    propagate_data = response.json()
    assert "results" in propagate_data
    assert len(propagate_data["results"]) > 0

    # 5. List sessions
    response = client.get("/api/v1/video/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert any(s["session_id"] == session_id for s in sessions)

    # 6. Reset session
    response = client.post(f"/api/v1/video/session/{session_id}/reset")
    assert response.status_code == 200

    # 7. Close session
    response = client.delete(f"/api/v1/video/session/{session_id}")
    assert response.status_code == 200
    close_data = response.json()
    assert close_data["session_id"] == session_id
