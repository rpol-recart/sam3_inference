# SAM3 Inference Server API Documentation

## Overview

The SAM3 Inference Server provides a FastAPI-based interface for image and video segmentation using the Segment Anything Model 3 (SAM3). The API supports various types of prompts for segmentation including text, bounding boxes, and point prompts.

### Base URL
```
http://localhost:8000
```

### API Version
```
v1
```

## Authentication

Authentication is optional by default, but can be configured via API keys in the server configuration.

## Common Data Types

### Prompt Types
- `TEXT`: Text description of the object to segment
- `POINT`: Point-based prompt with positive/negative clicks
- `BOX`: Bounding box prompt

### Common Response Fields
- `success`: Boolean indicating if the request was successful
- `error`: Error message (when applicable)

## Endpoints

## Health & Monitoring

### GET `/health`
Health check endpoint to verify server status.

#### Response
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 1,
  "active_sessions": 0
}
```

### GET `/models/info`
Get information about loaded models.

#### Response
```json
{
  "image_model": {
    "loaded": true,
    "checkpoint": "/app/server/sam_weights/sam3.pt",
    "device": "cuda:0",
    "memory_mb": 1200.5,
    "capabilities": [
      "text_prompt",
      "box_prompt",
      "batch_processing",
      "feature_caching"
    ]
  },
  "server_version": "1.0.0",
  "sam3_version": "1.0.0"
}
```

### GET `/metrics`
Prometheus-style metrics endpoint.

#### Response
```
sam3_gpu_memory_allocated_bytes{gpu="0"} 1258291200
sam3_gpu_memory_reserved_bytes{gpu="0"} 2147483648
sam3_cpu_usage_percent 15.2
sam3_memory_usage_percent 42.8
```

## Image Inference

### POST `/api/v1/image/segment`
Segment an image using various prompt types.

#### Request Body
```json
{
  "image": "base64_encoded_image_string",
  "prompts": [
    {
      "type": "text",
      "text": "person wearing red shirt"
    },
    {
      "type": "box",
      "box": [0.2, 0.3, 0.4, 0.5],
      "label": true
    },
    {
      "type": "point",
      "points": [[0.3, 0.4], [0.6, 0.7]],
      "point_labels": [1, 1]
    }
  ],
  "confidence_threshold": 0.5,
  "return_visualization": false
}
```

#### Request Schema
- `image`: Base64-encoded image string
- `prompts`: Array of prompt objects (at least 1 required)
- `confidence_threshold`: Float between 0.0 and 1.0 (default: 0.5)
- `return_visualization`: Boolean (default: false)

#### Response
```json
{
  "masks": ["rle_encoded_mask_1", "rle_encoded_mask_2"],
  "boxes": [[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]],
  "scores": [0.92, 0.87],
  "num_masks": 2,
  "image_size": {
    "width": 640,
    "height": 480
  },
  "visualization_url": null,
  "inference_time_ms": 125.3
}
```

#### Response Schema
- `masks`: Array of RLE-encoded segmentation masks
- `boxes`: Array of bounding boxes in [cx, cy, w, h] format (normalized)
- `scores`: Array of confidence scores
- `num_masks`: Total number of masks returned
- `image_size`: Original image dimensions
- `visualization_url`: URL to visualization if requested (null if not requested)
- `inference_time_ms`: Processing time in milliseconds

### POST `/api/v1/image/cached-features`
Segment image with multiple text prompts using feature caching for improved performance.

#### Request Body
```json
{
  "image": "base64_encoded_image_string",
  "text_prompts": ["person", "car", "tree"],
  "confidence_threshold": 0.5
}
```

#### Request Schema
- `image`: Base64-encoded image string
- `text_prompts`: Array of text prompts (at least 1 required)
- `confidence_threshold`: Float between 0.0 and 1.0 (default: 0.5)

#### Response
```json
{
  "results": [
    {
      "prompt": "person",
      "masks": ["rle_encoded_mask"],
      "boxes": [[0.2, 0.3, 0.4, 0.5]],
      "scores": [0.92],
      "num_masks": 1
    },
    {
      "prompt": "car",
      "masks": ["rle_encoded_mask"],
      "boxes": [[0.6, 0.7, 0.2, 0.15]],
      "scores": [0.88],
      "num_masks": 1
    }
  ],
  "cache_hit": true,
  "inference_time_ms": 85.2
}
```

#### Response Schema
- `results`: Array of results for each prompt
  - `prompt`: The text prompt
  - `masks`: Array of RLE-encoded masks
  - `boxes`: Array of bounding boxes
  - `scores`: Array of confidence scores
  - `num_masks`: Number of masks for this prompt
- `cache_hit`: Boolean indicating if cached features were used
- `inference_time_ms`: Processing time in milliseconds

## Video Inference

### POST `/api/v1/video/session/start`
Start a new video inference session.

#### Request Body
```json
{
  "video_url": "https://example.com/video.mp4",
  "session_id": "custom-session-id",
  "gpu_ids": [0, 1]
}
```
OR
```json
{
  "video_base64": "base64_encoded_video_string",
  "session_id": "custom-session-id"
}
```
OR
```json
{
  "video_path": "/path/to/local/video.mp4",
  "session_id": "custom-session-id"
}
```

#### Request Schema
- One of: `video_url`, `video_base64`, or `video_path` (required)
- `session_id`: Custom session ID (auto-generated if not provided)
- `gpu_ids`: Array of GPU IDs for multi-GPU processing

#### Response
```json
{
  "session_id": "session-12345",
  "video_info": {
    "total_frames": 300,
    "fps": 30.0,
    "resolution": {
      "width": 1920,
      "height": 1080
    },
    "duration_seconds": 10.0
  },
  "status": "ready"
}
```

#### Response Schema
- `session_id`: Unique session identifier
- `video_info`: Video metadata
  - `total_frames`: Total number of frames in the video
  - `fps`: Frames per second
  - `resolution`: Width and height of video
  - `duration_seconds`: Video duration in seconds
- `status`: Current session status ("ready")

### POST `/api/v1/video/session/{session_id}/prompt`
Add prompts to a specific frame in the video for object tracking initialization.

#### Path Parameters
- `session_id`: The session ID

#### Request Body
```json
{
  "frame_index": 0,
  "prompts": [
    {
      "type": "text",
      "text": "red car"
    },
    {
      "type": "box",
      "box": [0.2, 0.3, 0.4, 0.5],
      "label": true
    }
  ],
  "obj_id": null
}
```

#### Request Schema
- `frame_index`: Frame index to add prompt (required, >= 0)
- `prompts`: Array of prompts (at least 1 required)
- `obj_id`: Object ID to refine (null for new object)

#### Response
```json
{
  "frame_index": 0,
  "obj_id": [1],
  "masks": ["rle_encoded_mask"],
  "boxes": [[0.2, 0.3, 0.4, 0.5]],
  "scores": [0.92],
  "status": "prompt_added"
}
```

#### Response Schema
- `frame_index`: The frame index
- `obj_id`: Array of assigned object IDs
- `masks`: Array of RLE-encoded masks
- `boxes`: Array of bounding boxes
- `scores`: Array of confidence scores
- `status`: Status message

### POST `/api/v1/video/session/{session_id}/propagate`
Propagate object tracking through video frames (non-streaming).

#### Path Parameters
- `session_id`: The session ID

#### Request Body
```json
{
  "direction": "both",
  "start_frame_index": 0,
  "max_frames": 100,
  "stream": false
}
```

#### Request Schema
- `direction`: Direction for propagation ("forward", "backward", "both") - default: "both"
- `start_frame_index`: Starting frame index - default: 0
- `max_frames`: Max frames to process (null for all) - default: null
- `stream`: Use WebSocket streaming if true - default: false

#### Response
```json
{
  "session_id": "session-12345",
  "results": {
    "0": {
      "frame_index": 0,
      "objects": [
        {
          "id": 1,
          "mask": "rle_encoded_mask",
          "box": [0.2, 0.3, 0.4, 0.5],
          "score": 0.92
        }
      ]
    },
    "1": {
      "frame_index": 1,
      "objects": [
        {
          "id": 1,
          "mask": "rle_encoded_mask",
          "box": [0.21, 0.31, 0.4, 0.5],
          "score": 0.89
        }
      ]
    }
  },
  "total_frames": 2,
  "processing_time_ms": 450.2
}
```

#### Response Schema
- `session_id`: Session identifier
- `results`: Dictionary mapping frame indices to frame results
  - `frame_index`: Frame index
  - `objects`: Array of tracked objects in the frame
    - `id`: Object ID
    - `mask`: RLE-encoded mask
    - `box`: Bounding box [cx, cy, w, h]
    - `score`: Confidence score
- `total_frames`: Total number of processed frames
- `processing_time_ms`: Processing time in milliseconds

### WebSocket `/api/v1/video/ws/propagate/{session_id}`
WebSocket endpoint for streaming video propagation results.

#### Connection
Connect to the WebSocket endpoint to receive real-time frame results.

#### Send Message (Request)
```json
{
  "direction": "forward",
  "start_frame_index": 0,
  "max_frames": null
}
```

#### Receive Messages (Responses)
Frame result message:
```json
{
  "type": "frame",
  "frame_index": 5,
  "objects": [
    {
      "id": 1,
      "mask": "rle_encoded_mask",
      "box": [0.25, 0.35, 0.4, 0.5],
      "score": 0.91
    }
  ]
}
```

Completion message:
```json
{
  "type": "complete",
  "total_frames": 10
}
```

Error message:
```json
{
  "type": "error",
  "error": "Processing failed"
}
```

### GET `/api/v1/video/session/{session_id}/status`
Get current status of a video session.

#### Path Parameters
- `session_id`: The session ID

#### Response
```json
{
  "session_id": "session-12345",
  "status": "ready",
  "current_objects": 3,
  "frames_processed": 50,
  "total_frames": 300,
  "gpu_memory_used_mb": 1200.5
}
```

#### Response Schema
- `session_id`: Session identifier
- `status`: Current session status ("ready", "processing", "error", "closed")
- `current_objects`: Number of currently tracked objects
- `frames_processed`: Number of frames processed
- `total_frames`: Total number of frames in video
- `gpu_memory_used_mb`: GPU memory used in MB

### DELETE `/api/v1/video/session/{session_id}/object/{obj_id}`
Remove an object from tracking in the session.

#### Path Parameters
- `session_id`: The session ID
- `obj_id`: The object ID to remove

#### Response
```json
{
  "session_id": "session-12345",
  "obj_id": 1,
  "status": "removed"
}
```

#### Response Schema
- `session_id`: Session identifier
- `obj_id`: Removed object ID
- `status`: Status message

### POST `/api/v1/video/session/{session_id}/reset`
Reset session to initial state (clears all prompts and objects).

#### Path Parameters
- `session_id`: The session ID

#### Response
```json
{
  "session_id": "session-12345",
  "status": "reset",
  "objects_cleared": 3
}
```

#### Response Schema
- `session_id`: Session identifier
- `status`: Status message
- `objects_cleared`: Number of objects cleared

### DELETE `/api/v1/video/session/{session_id}`
Close and cleanup video session.

#### Path Parameters
- `session_id`: The session ID

#### Response
```json
{
  "session_id": "session-12345",
  "status": "closed",
  "memory_freed_mb": 1200.5
}
```

#### Response Schema
- `session_id`: Session identifier
- `status`: Status message
- `memory_freed_mb`: Memory freed in MB

### GET `/api/v1/video/sessions`
List all active video sessions.

#### Response
```json
{
  "sessions": [
    {
      "session_id": "session-12345",
      "type": "video",
      "created_at": "2023-10-15T10:30:00Z",
      "status": "ready",
      "objects_count": 3
    }
  ],
  "total_sessions": 1
}
```

#### Response Schema
- `sessions`: Array of active sessions
  - `session_id`: Session identifier
  - `type`: Session type ("video" or "image_batch")
  - `created_at`: Creation timestamp in ISO format
  - `status`: Current status
  - `objects_count`: Number of tracked objects (for video sessions)
  - `images_processed`: Number of processed images (for batch sessions)
- `total_sessions`: Total number of active sessions

## Error Responses

The API returns standard HTTP error codes:

- `400 Bad Request`: Invalid request format or parameters
- `404 Not Found`: Resource not found (e.g., invalid session ID)
- `422 Unprocessable Entity`: Validation error for request body
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Model not loaded or service unavailable

### Error Response Format
```json
{
  "detail": "Error message describing the issue"
}
```

## Configuration

The server behavior can be configured using environment variables defined in the `.env` file:

- `SERVER_HOST`: Host to bind to (default: "0.0.0.0")
- `SERVER_PORT`: Port to listen on (default: 8000)
- `SAM3_CHECKPOINT`: Path to SAM3 model checkpoint
- `IMAGE_MODEL_DEVICE`: Device for image model (default: "cuda:0")
- `VIDEO_MODEL_ENABLED`: Enable video inference (default: true)
- `MAX_CONCURRENT_SESSIONS`: Maximum concurrent video sessions (default: 10)

## Performance Notes

1. **Feature Caching**: The `/api/v1/image/cached-features` endpoint caches image features and reuses them for multiple text prompts, resulting in ~10x speed improvement compared to separate requests.

2. **Video Streaming**: For real-time applications, use the WebSocket endpoint (`/ws/propagate`) to receive frame results as they're processed rather than waiting for batch completion.

3. **GPU Memory Management**: Sessions consume GPU memory, so remember to close sessions when finished using the DELETE endpoint.

## Examples

### Python Client Example
```python
import requests
import base64

# Read and encode image
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post(
    'http://localhost:8000/api/v1/image/segment',
    json={
        'image': image_data,
        'prompts': [
            {'type': 'text', 'text': 'person'}
        ]
    }
)

result = response.json()
print(f"Found {result['num_masks']} masks")
```

### JavaScript Client Example
```javascript
// Read image file and convert to base64
const reader = new FileReader();
reader.onload = function(event) {
    const base64Image = event.target.result.split(',')[1]; // Remove data:image/jpeg;base64, prefix
    
    fetch('http://localhost:8000/api/v1/image/segment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: base64Image,
            prompts: [
                { type: 'text', text: 'person' }
            ]
        })
    })
    .then(response => response.json())
    .then(data => console.log('Masks:', data.masks.length));
};

reader.readAsDataURL(imageFile);