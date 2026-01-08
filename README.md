# SAM3 Inference Server

FastAPI-based inference server for SAM3 (Segment Anything 3) with support for image and video segmentation.

## Features

### Image Mode ✅ (Implemented)
- ✅ Text-based segmentation ("person", "car in red")
- ✅ Bounding box prompts (visual exemplars)
- ✅ Combined prompts (text + boxes)
- ✅ Feature caching for multiple prompts (~10x speedup)
- ✅ Batch processing
- ✅ Real-time inference

### Video Mode ✅ (Implemented)
- ✅ Multi-object tracking
- ✅ Session management with lifecycle control
- ✅ WebSocket streaming for real-time results
- ✅ Forward/backward/both propagation
- ✅ Text and box prompts on video frames
- ✅ Multi-GPU support

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
# Minimum required:
# - SAM3_CHECKPOINT=facebook/sam3
# - IMAGE_MODEL_DEVICE=cuda:0
```

### 3. Run Server

```bash
# Development mode (with auto-reload)
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at:
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API Endpoints

### Image Inference

#### POST `/api/v1/image/segment`
Segment image with text or box prompts.

**Request**:
```json
{
  "image": "base64_encoded_image",
  "prompts": [
    {"type": "text", "text": "person"},
    {"type": "box", "box": [0.3, 0.4, 0.2, 0.3], "label": true}
  ],
  "confidence_threshold": 0.5
}
```

**Response**:
```json
{
  "masks": ["rle_encoded_mask_1", "rle_encoded_mask_2"],
  "boxes": [[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.15, 0.25]],
  "scores": [0.95, 0.87],
  "num_masks": 2,
  "image_size": {"width": 1024, "height": 768},
  "inference_time_ms": 125.3
}
```

#### POST `/api/v1/image/cached-features`
Segment with multiple text prompts using feature caching (10x faster).

**Request**:
```json
{
  "image": "base64_encoded_image",
  "text_prompts": ["person", "car", "bicycle", "dog"],
  "confidence_threshold": 0.5
}
```

**Response**:
```json
{
  "results": [
    {
      "prompt": "person",
      "masks": [...],
      "boxes": [...],
      "scores": [...],
      "num_masks": 2
    },
    ...
  ],
  "cache_hit": true,
  "inference_time_ms": 450.2
}
```

### Health & Monitoring

#### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 4,
  "active_sessions": 0
}
```

#### GET `/models/info`
Get loaded models information.

#### GET `/metrics`
Prometheus-style metrics.

### Video Inference

#### POST `/api/v1/video/session/start`
Start a new video inference session.

**Request**:
```json
{
  "video_path": "/path/to/video.mp4",
  "session_id": "optional-custom-id",
  "gpu_ids": [0, 1]
}
```

**Response**:
```json
{
  "session_id": "uuid-generated",
  "video_info": {
    "total_frames": 300,
    "fps": 30.0,
    "resolution": {"width": 1920, "height": 1080},
    "duration_seconds": 10.0
  },
  "status": "ready"
}
```

#### POST `/api/v1/video/session/{session_id}/prompt`
Add prompt to a specific frame.

**Request**:
```json
{
  "frame_index": 0,
  "prompts": [
    {"type": "text", "text": "person in red shirt"}
  ],
  "obj_id": null
}
```

**Response**:
```json
{
  "frame_index": 0,
  "obj_id": 1,
  "masks": ["rle_encoded_mask"],
  "boxes": [[0.5, 0.5, 0.2, 0.3]],
  "scores": [0.95]
}
```

#### POST `/api/v1/video/session/{session_id}/propagate`
Propagate tracking through video (batch mode).

**Request**:
```json
{
  "direction": "both",
  "start_frame_index": 0,
  "max_frames": 100,
  "stream": false
}
```

**Response**:
```json
{
  "session_id": "uuid",
  "results": {
    "0": {
      "frame_index": 0,
      "objects": [
        {"id": 1, "mask": "rle", "box": [0.5, 0.5, 0.2, 0.3], "score": 0.95}
      ]
    },
    ...
  },
  "total_frames": 100,
  "processing_time_ms": 5234.5
}
```

#### WS `/api/v1/video/ws/propagate/{session_id}`
WebSocket endpoint for streaming propagation results.

**Send**:
```json
{
  "direction": "forward",
  "start_frame_index": 0,
  "max_frames": null
}
```

**Receive (per frame)**:
```json
{
  "type": "frame",
  "frame_index": 5,
  "objects": [
    {"id": 1, "mask": "rle", "box": [...], "score": 0.93}
  ]
}
```

**Receive (completion)**:
```json
{
  "type": "complete",
  "total_frames": 300
}
```

#### GET `/api/v1/video/session/{session_id}/status`
Get session status and statistics.

#### DELETE `/api/v1/video/session/{session_id}/object/{obj_id}`
Remove object from tracking.

#### POST `/api/v1/video/session/{session_id}/reset`
Reset session (clear all prompts/objects).

#### DELETE `/api/v1/video/session/{session_id}`
Close and cleanup session.

#### GET `/api/v1/video/sessions`
List all active sessions.

## Usage Examples

### Python Client

```python
import base64
import requests
from PIL import Image
from io import BytesIO

# Load image
image = Image.open("image.jpg")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode()

# Segment with text prompt
response = requests.post(
    "http://localhost:8000/api/v1/image/segment",
    json={
        "image": image_base64,
        "prompts": [
            {"type": "text", "text": "person"}
        ],
        "confidence_threshold": 0.5
    }
)

result = response.json()
print(f"Found {result['num_masks']} masks")
print(f"Scores: {result['scores']}")

# Video tracking example
video_response = requests.post(
    "http://localhost:8000/api/v1/video/session/start",
    json={"video_path": "/path/to/video.mp4"}
)
session = video_response.json()
session_id = session["session_id"]

# Add prompt on first frame
requests.post(
    f"http://localhost:8000/api/v1/video/session/{session_id}/prompt",
    json={
        "frame_index": 0,
        "prompts": [{"type": "text", "text": "person"}]
    }
)

# Propagate through video
propagate_response = requests.post(
    f"http://localhost:8000/api/v1/video/session/{session_id}/propagate",
    json={"direction": "both", "max_frames": 100}
)
results = propagate_response.json()
print(f"Processed {results['total_frames']} frames")

# Close session
requests.delete(f"http://localhost:8000/api/v1/video/session/{session_id}")
```

### cURL

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 image.jpg)

# Send request
curl -X POST "http://localhost:8000/api/v1/image/segment" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$IMAGE_B64\",
    \"prompts\": [{\"type\": \"text\", \"text\": \"person\"}],
    \"confidence_threshold\": 0.5
  }"
```

## Performance

### Image Inference Benchmarks

| GPU | Resolution | Latency (single) | Throughput |
|-----|------------|------------------|------------|
| H100 | 1008x1008 | ~100ms | 10 img/s |
| A100 | 1008x1008 | ~150ms | 6-7 img/s |
| RTX 4090 | 1008x1008 | ~200ms | 5 img/s |

### Feature Caching

| Prompts | Without Cache | With Cache | Speedup |
|---------|--------------|------------|---------|
| 4 prompts | ~400ms | ~50ms | **8x** |
| 8 prompts | ~800ms | ~80ms | **10x** |

## Configuration

See [.env.example](.env.example) for all configuration options.

**Key Settings**:
```bash
# Model
SAM3_CHECKPOINT=facebook/sam3
IMAGE_MODEL_DEVICE=cuda:0
IMAGE_MODEL_CONFIDENCE_THRESHOLD=0.5

# Server
SERVER_PORT=8000
LOG_LEVEL=INFO

# Cache
ENABLE_FEATURE_CACHE=true
FEATURE_CACHE_TTL_SECONDS=600
```

## Docker Deployment

**Quick Start**:
```bash
# Option 1: GitHub SAM3 (recommended)
cd D:\Projects\Sam_agent\model_inference\sam3
docker-compose up -d

# Option 2: Local SAM3
cd D:\Projects\Sam_agent
docker-compose -f model_inference/sam3/docker-compose.local.yml up -d
```

**Подробные инструкции**: См. [DOCKER_BUILD_GUIDE.md](DOCKER_BUILD_GUIDE.md)

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black .
flake8 .
```

### Adding New Endpoints

1. Create schema in `api/schemas/`
2. Create route in `api/routes/`
3. Include router in `server.py`

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image resolution:
```bash
IMAGE_MODEL_RESOLUTION=896  # default: 1008
```

### Slow Inference

Enable torch.compile (requires PyTorch 2.0+):
```bash
IMAGE_MODEL_COMPILE=true
```

### Model Download Issues

Set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
```

## Architecture

```
sam3/
├── server.py              # FastAPI app
├── config.py              # Configuration
├── api/
│   ├── routes/
│   │   ├── image.py      # Image endpoints
│   │   ├── video.py      # Video endpoints
│   │   └── health.py     # Health/metrics
│   └── schemas/          # Pydantic models
├── models/
│   ├── sam3_image.py     # Image model wrapper
│   └── sam3_video.py     # Video model wrapper
├── services/
│   └── session_manager.py # Session lifecycle
└── tests/                # Test suite
```

## License

SAM 3 License - See parent repository

## References

- [SAM3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [Project Documentation](../../docs/SAM3_INTEGRATION.md)

---

**Status**: Image inference ✅ | Video inference ✅ | Production Ready
