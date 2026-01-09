"""Pytest configuration and fixtures."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_image_model():
    """Mock image model for testing without loading actual SAM3."""
    # Create a mock that mimics the SAM3ImageModel interface
    class MockImageModel:
        def __init__(self):
            self.feature_cache = {}
        
        def segment_combined(self, image, text_prompts=None, boxes=None, points=None):
            # Return mock segmentation results
            return ["mock_mask_1", "mock_mask_2"], [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.3, 0.3]], [0.9, 0.8]
        
        def segment_with_cached_features(self, cache_key, text_prompts):
            # Return mock results for each prompt
            results = []
            for _ in text_prompts:
                results.append((["mock_mask"], [[0.1, 0.1, 0.5, 0.5]], [0.9]))
            return results
        
        def cache_features(self, image, cache_key):
            self.feature_cache[cache_key] = {"image": image}
    
    return MockImageModel()


@pytest.fixture
def mock_video_model():
    """Mock video model for testing without loading actual SAM3."""
    class MockVideoModel:
        def start_session(self, video_path, session_id=None):
            return session_id or "mock_session_id", {
                "total_frames": 100,
                "resolution": {"width": 1920, "height": 1080},
                "fps": 30.0,
                "duration_seconds": 10.0,
            }
        
        def add_prompt(self, session_id, frame_index, text_prompt=None, points=None, 
                      point_labels=None, boxes=None, box_labels=None, obj_id=None):
            return frame_index, [obj_id or 1], ["mock_mask"], [[0.1, 0.1, 0.5, 0.5]], [0.9]
        
        def propagate_in_video(self, session_id, direction="both", 
                              start_frame_index=None, max_frames=None):
            # Yield mock frame data
            for i in range(5):  # Just return 5 frames for testing
                yield {
                    "frame_index": i,
                    "objects": [{
                        "id": 1,
                        "mask": "mock_mask",
                        "box": [0.1, 0.1, 0.5, 0.5],
                        "score": 0.9
                    }]
                }
        
        def get_session_info(self, session_id):
            return {
                "num_frames": 100,
                "num_objects": 1,
                "gpu_memory_mb": 100.0,
                "start_time": "2023-01-01T00:00:00Z"
            }
        
        def remove_object(self, session_id, obj_id):
            return True
        
        def reset_session(self, session_id):
            return True
        
        def close_session(self, session_id):
            return True
    
    return MockVideoModel()
