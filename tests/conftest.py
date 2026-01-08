"""Pytest configuration and fixtures."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """Create test client for FastAPI app."""
    from server import app

    return TestClient(app)


@pytest.fixture(scope="session")
def mock_image_model():
    """Mock image model for testing without loading actual SAM3."""
    # TODO: Implement mock model
    pass
