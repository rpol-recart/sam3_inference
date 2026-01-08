# SAM3 Inference Server - Deployment Guide

## Quick Start (Development)

### 1. Install Dependencies

```bash
cd D:\Projects\Sam_agent\model_inference\sam3

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Server

```bash
# Copy configuration template
cp .env.example .env

# Edit .env (minimum required):
# IMAGE_MODEL_DEVICE=cuda:0
# SAM3_CHECKPOINT=facebook/sam3
```

### 3. Run Server

```bash
# Start server
python server.py

# Server runs on http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

## Testing the Server

### Using Swagger UI (Browser)

1. Open http://localhost:8000/docs
2. Click on `/api/v1/image/segment`
3. Click "Try it out"
4. Paste base64-encoded image
5. Add prompts and execute

### Using Python

```python
import base64
import requests
from PIL import Image
from io import BytesIO

# Encode image
img = Image.open("test.jpg")
buffer = BytesIO()
img.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/api/v1/image/segment",
    json={
        "image": img_b64,
        "prompts": [{"type": "text", "text": "person"}],
        "confidence_threshold": 0.5
    }
)

print(response.json())
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Segment image
IMAGE_B64=$(base64 -w 0 test.jpg)
curl -X POST http://localhost:8000/api/v1/image/segment \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$IMAGE_B64\",
    \"prompts\": [{\"type\": \"text\", \"text\": \"person\"}]
  }"
```

## Docker Deployment

### Build Image

**Option 1: Using GitHub SAM3 (recommended for production)**

Клонирует SAM3 из GitHub внутри контейнера:

```bash
cd D:\Projects\Sam_agent\model_inference\sam3

# Build Docker image
docker build -t sam3-inference:latest .

# Check image
docker images | grep sam3
```

**Option 2: Using Local SAM3 Copy**

Использует локальную копию репозитория SAM3. Сборка из корневой директории проекта:

```bash
cd D:\Projects\Sam_agent

# Build Docker image with local SAM3
docker build -f model_inference/sam3/Dockerfile.local -t sam3-inference:latest .

# Check image
docker images | grep sam3
```

**Примечание**:
- `Dockerfile` (Option 1) клонирует SAM3 из GitHub и может быть запущен из директории `model_inference/sam3`
- `Dockerfile.local` (Option 2) использует локальную копию и должен запускаться из корня проекта `D:\Projects\Sam_agent`

### Run Container

```bash
# Run with GPU support
docker run -d \
  --name sam3-server \
  --gpus all \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/outputs:/tmp/sam3_outputs \
  sam3-inference:latest

# Check logs
docker logs -f sam3-server

# Check status
curl http://localhost:8000/health
```

### Docker Compose

**Option 1: Using GitHub SAM3**

Запуск из директории `model_inference/sam3`:

```bash
cd D:\Projects\Sam_agent\model_inference\sam3
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

**Option 2: Using Local SAM3**

Запуск из корневой директории проекта:

```bash
cd D:\Projects\Sam_agent
docker-compose -f model_inference/sam3/docker-compose.local.yml up -d

# Check logs
docker-compose -f model_inference/sam3/docker-compose.local.yml logs -f

# Stop
docker-compose -f model_inference/sam3/docker-compose.local.yml down
```

**Доступные файлы**:
- `docker-compose.yml` - использует GitHub SAM3 (Dockerfile)
- `docker-compose.local.yml` - использует локальную копию SAM3 (Dockerfile.local)

## Production Deployment

### Requirements

- **GPU**: NVIDIA with 8GB+ VRAM (RTX 3060 Ti minimum)
- **RAM**: 16GB minimum
- **Storage**: 20GB for model + cache
- **CUDA**: 12.6+
- **Python**: 3.12+

### Configuration for Production

Edit `.env`:

```bash
# Server
SERVER_WORKERS=4
LOG_LEVEL=WARNING

# Security
REQUIRE_API_KEY=true
API_KEYS=your-secret-key-1,your-secret-key-2
CORS_ORIGINS=https://yourdomain.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Performance
IMAGE_MODEL_COMPILE=true  # Enable torch.compile
ENABLE_FEATURE_CACHE=true
```

### Nginx Reverse Proxy

```nginx
upstream sam3_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://sam3_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts for long inference
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;

        # File upload size
        client_max_body_size 100M;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/sam3-server.service`:

```ini
[Unit]
Description=SAM3 Inference Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/sam3-server
Environment="PATH=/opt/sam3-server/venv/bin"
ExecStart=/opt/sam3-server/venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sam3-server
sudo systemctl start sam3-server
sudo systemctl status sam3-server
```

## Monitoring

### Prometheus Metrics

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'sam3-server'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Key metrics to monitor:
- `sam3_requests_total` - Total requests
- `sam3_inference_duration_seconds` - Inference latency
- `sam3_gpu_memory_allocated_bytes` - GPU memory usage
- `sam3_cpu_usage_percent` - CPU usage

## Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce image resolution
```bash
IMAGE_MODEL_RESOLUTION=896  # default: 1008
```

**Solution 2**: Reduce batch size
```bash
MAX_CONCURRENT_SESSIONS=5  # default: 10
```

### Slow First Request

**Cause**: Model loading on first request

**Solution**: Add warmup request after startup:
```bash
curl http://localhost:8000/health  # warmup
```

### Model Download Fails

**Solution**: Set HuggingFace cache:
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Import Errors

**Solution**: Ensure SAM3 parent directory is accessible:
```python
# In sam3_image.py, verify path:
SAM3_ROOT = Path(__file__).parent.parent.parent.parent / "sam3"
```

## Performance Tuning

### Enable Torch Compile (PyTorch 2.0+)

```bash
IMAGE_MODEL_COMPILE=true
```

Speedup: 1.5-2x faster after warmup

### Feature Caching

Use `/api/v1/image/cached-features` for multiple prompts on same image.

Speedup: ~10x for 4+ prompts

### Multi-Worker Mode

```bash
SERVER_WORKERS=4  # For multiple CPU cores
```

Note: Each worker loads separate model instance (high memory usage)

## Security Best Practices

1. **Enable API Keys**:
```bash
REQUIRE_API_KEY=true
API_KEYS=long-random-key-1,long-random-key-2
```

2. **Configure CORS**:
```bash
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

3. **Rate Limiting**:
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

4. **Use HTTPS**: Deploy behind Nginx with SSL/TLS

5. **Network Isolation**: Run in private network, expose via reverse proxy

## Backup & Recovery

### Backup Feature Cache

```bash
# Cache stored in memory, but can save to disk
tar -czf cache-backup.tar.gz /tmp/sam3_cache/
```

### Model Checkpoints

Models downloaded to `~/.cache/huggingface/`:
```bash
ls ~/.cache/huggingface/hub/models--facebook--sam3/
```

## Scaling

### Horizontal Scaling

Deploy multiple instances behind load balancer:

```nginx
upstream sam3_cluster {
    server 192.168.1.10:8000;
    server 192.168.1.11:8000;
    server 192.168.1.12:8000;
}
```

### GPU Distribution

Assign different GPUs to different instances:
```bash
# Instance 1
IMAGE_MODEL_DEVICE=cuda:0

# Instance 2
IMAGE_MODEL_DEVICE=cuda:1
```

## Support

- Issues: Create issue in project repository
- Documentation: See [README.md](README.md)
- Spec: [SAM3_INFERENCE_SERVER_SPEC.md](SAM3_INFERENCE_SERVER_SPEC.md)

## Video Inference Deployment

### Configuration

Edit `.env` for video:

```bash
# Video Model
VIDEO_MODEL_ENABLED=true
VIDEO_MODEL_GPUS=0,1,2,3  # Multi-GPU support
VIDEO_MODEL_TEMPORAL_DISAMBIGUATION=true

# Session Management
MAX_CONCURRENT_SESSIONS=10
SESSION_TIMEOUT_SECONDS=3600
```

### Session Management

Video inference uses stateful sessions that maintain tracking state:

```python
import requests

# Start session
response = requests.post(
    "http://localhost:8000/api/v1/video/session/start",
    json={"video_path": "/path/to/video.mp4"}
)
session_id = response.json()["session_id"]

# Add prompt on first frame
requests.post(
    f"http://localhost:8000/api/v1/video/session/{session_id}/prompt",
    json={
        "frame_index": 0,
        "prompts": [{"type": "text", "text": "person"}]
    }
)

# Propagate tracking
results = requests.post(
    f"http://localhost:8000/api/v1/video/session/{session_id}/propagate",
    json={"direction": "both", "max_frames": 100}
).json()

# Close session (important!)
requests.delete(f"http://localhost:8000/api/v1/video/session/{session_id}")
```

### Multi-GPU Configuration

For multi-GPU video processing:

```bash
# Instance 1 (GPUs 0,1)
VIDEO_MODEL_GPUS=0,1
SERVER_PORT=8000

# Instance 2 (GPUs 2,3)
VIDEO_MODEL_GPUS=2,3
SERVER_PORT=8001
```

### WebSocket Streaming

For real-time propagation results:

```python
import websocket
import json

ws = websocket.WebSocket()
ws.connect(f"ws://localhost:8000/api/v1/video/ws/propagate/{session_id}")

# Send propagation request
ws.send(json.dumps({
    "direction": "forward",
    "start_frame_index": 0,
    "max_frames": None
}))

# Receive frame results
while True:
    msg = json.loads(ws.recv())
    if msg["type"] == "frame":
        print(f"Frame {msg['frame_index']}: {len(msg['objects'])} objects")
    elif msg["type"] == "complete":
        print(f"Complete! Total frames: {msg['total_frames']}")
        break
    elif msg["type"] == "error":
        print(f"Error: {msg['error']}")
        break

ws.close()
```

### Session Cleanup

Sessions automatically expire after `SESSION_TIMEOUT_SECONDS` (default: 1 hour).

Manual cleanup:

```bash
# List all sessions
curl http://localhost:8000/api/v1/video/sessions

# Close specific session
curl -X DELETE http://localhost:8000/api/v1/video/session/{session_id}
```

### Video Performance

| GPU | Resolution | FPS (propagation) | GPU Memory |
|-----|------------|-------------------|------------|
| H100 | 1080p | ~30 FPS | 8GB |
| A100 | 1080p | ~20 FPS | 10GB |
| RTX 4090 | 1080p | ~15 FPS | 12GB |

Multi-GPU scaling:
- 2 GPUs: ~1.8x throughput
- 4 GPUs: ~3.2x throughput

---

**Status**: Production Ready for Image & Video Inference ✅
