# SAM3 Inference Server - Спецификация

> **Цель**: Создать FastAPI сервер для inference SAM3 модели с поддержкой изображений и видео
> **Расположение**: `D:\Projects\Sam_agent\model_inference\sam3\`

---

## Обзор Возможностей SAM3

### Работа с Изображениями (Image Mode)

**Основные возможности**:
- ✅ Сегментация по текстовым промптам ("person", "car in red")
- ✅ Сегментация по box prompts (visual exemplars)
- ✅ Комбинированные промпты (text + boxes)
- ✅ Batch processing множества изображений
- ✅ Multi-mask output с confidence scores
- ✅ Interactive refinement (добавление уточняющих промптов)

**Технические характеристики**:
- Input resolution: до 1008x1008 (внутренняя нормализация)
- Precision: bfloat16 (рекомендуется)
- Device: CUDA (single GPU)
- Latency: ~100-200ms на изображение

### Работа с Видео (Video Mode)

**Основные возможности**:
- ✅ Dense object tracking через все кадры
- ✅ Multi-object tracking с уникальными ID
- ✅ Text prompts для видео ("person in blue vest")
- ✅ Point prompts для уточнения (positive/negative clicks)
- ✅ Box prompts (visual exemplars)
- ✅ Forward/backward propagation
- ✅ Session management (stateful processing)
- ✅ Object removal и tracking reset
- ✅ Multi-GPU distributed processing

**Технические характеристики**:
- Input formats: MP4, JPEG sequence
- Multi-GPU: NCCL distributed (до 8 GPU)
- Memory: Держит whole video в session state
- Latency: ~30-50ms на frame при propagation

---

## Архитектура Inference Сервера

### High-Level Design

```
┌─────────────────────────────────────────────────────┐
│              SAM3 Inference Server (FastAPI)         │
│                                                       │
│  ┌───────────────────────────────────────────────┐  │
│  │         API Layer (REST + WebSocket)          │  │
│  │  - Image endpoints (/api/v1/image/*)         │  │
│  │  - Video endpoints (/api/v1/video/*)         │  │
│  │  - Session management (/api/v1/sessions/*)   │  │
│  │  - Health & metrics (/health, /metrics)      │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                              │
│  ┌────────────────────▼──────────────────────────┐  │
│  │           Service Layer                       │  │
│  │  - ImageInferenceService                     │  │
│  │  - VideoInferenceService                     │  │
│  │  - SessionManager                            │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                              │
│  ┌────────────────────▼──────────────────────────┐  │
│  │           Model Layer                         │  │
│  │  - Sam3Processor (image)                     │  │
│  │  - Sam3VideoPredictor (video)                │  │
│  │  - Model loader & cache                      │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌────────┐    ┌─────────┐    ┌─────────┐
   │ GPU 0  │    │  GPU 1  │    │  GPU N  │
   │ (Image)│    │ (Video) │    │ (Video) │
   └────────┘    └─────────┘    └─────────┘
```

---

## API Endpoints Specification

### 1. Image Inference API

#### POST `/api/v1/image/segment`
Сегментация одного изображения

**Request**:
```json
{
  "image": "base64_encoded_image",
  "prompts": [
    {
      "type": "text",
      "text": "person in red shirt"
    },
    {
      "type": "box",
      "box": [0.3, 0.4, 0.5, 0.6],  // [cx, cy, w, h] normalized
      "label": true  // true=positive, false=negative
    }
  ],
  "confidence_threshold": 0.5,
  "return_visualization": false
}
```

**Response**:
```json
{
  "masks": ["rle_encoded_mask_1", "rle_encoded_mask_2"],
  "boxes": [[0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5]],  // XYWH normalized
  "scores": [0.95, 0.87],
  "num_masks": 2,
  "image_size": {"width": 1024, "height": 768},
  "visualization_url": "/api/v1/images/viz/abc123.png"  // if requested
}
```

---

#### POST `/api/v1/image/batch`
Batch processing множества изображений

**Request**:
```json
{
  "images": [
    {
      "id": "img_001",
      "image": "base64_1",
      "prompts": [{"type": "text", "text": "car"}]
    },
    {
      "id": "img_002",
      "image": "base64_2",
      "prompts": [{"type": "text", "text": "person"}]
    }
  ],
  "confidence_threshold": 0.5,
  "max_concurrent": 4
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "img_001",
      "masks": [...],
      "boxes": [...],
      "scores": [...]
    },
    {
      "id": "img_002",
      "masks": [...],
      "boxes": [...],
      "scores": [...]
    }
  ],
  "total_images": 2,
  "successful": 2,
  "failed": 0
}
```

---

#### POST `/api/v1/image/cached-features`
Feature caching для множества промптов на одном изображении

**Request**:
```json
{
  "image": "base64_encoded_image",
  "text_prompts": [
    "person",
    "car",
    "bicycle",
    "traffic sign"
  ],
  "confidence_threshold": 0.5
}
```

**Response**:
```json
{
  "results": [
    {"prompt": "person", "masks": [...], "boxes": [...], "scores": [...]},
    {"prompt": "car", "masks": [...], "boxes": [...], "scores": [...]},
    {"prompt": "bicycle", "masks": [...], "boxes": [...], "scores": [...]},
    {"prompt": "traffic sign", "masks": [...], "boxes": [...], "scores": [...]}
  ],
  "cache_hit": true,
  "inference_time_ms": 450
}
```

---

### 2. Video Inference API

#### POST `/api/v1/video/sessions/start`
Начать новую video session

**Request**:
```json
{
  "video_url": "http://example.com/video.mp4",  // OR
  "video_base64": "base64_encoded_video",  // OR
  "video_path": "/path/to/video.mp4",  // local path
  "session_id": "optional-custom-id",  // auto-generated if not provided
  "gpu_ids": [0, 1, 2]  // для multi-GPU processing
}
```

**Response**:
```json
{
  "session_id": "vid_abc123def456",
  "video_info": {
    "total_frames": 300,
    "fps": 30,
    "resolution": {"width": 1920, "height": 1080},
    "duration_seconds": 10.0
  },
  "status": "ready"
}
```

---

#### POST `/api/v1/video/sessions/{session_id}/prompts`
Добавить промпт в video session

**Request**:
```json
{
  "frame_index": 0,
  "prompts": [
    {
      "type": "text",
      "text": "person in blue vest"
    },
    {
      "type": "point",
      "points": [[0.5, 0.3], [0.6, 0.4]],
      "point_labels": [1, 0]  // 1=positive, 0=negative
    },
    {
      "type": "box",
      "box": [0.3, 0.4, 0.5, 0.6],
      "label": true
    }
  ],
  "obj_id": null  // null=new object, int=refine existing
}
```

**Response**:
```json
{
  "frame_index": 0,
  "obj_id": 0,
  "masks": ["rle_encoded"],
  "boxes": [[0.3, 0.4, 0.5, 0.6]],
  "scores": [0.95],
  "status": "prompt_added"
}
```

---

#### POST `/api/v1/video/sessions/{session_id}/propagate`
Propagate tracking через видео

**Request**:
```json
{
  "direction": "both",  // "forward", "backward", "both"
  "start_frame_index": 0,
  "max_frames": null,  // null = all frames
  "stream": true  // WebSocket streaming if true
}
```

**Response (streaming)**:
```json
// WebSocket stream of frames:
{"frame_index": 0, "objects": [{"id": 0, "mask": "...", "box": [...], "score": 0.95}]}
{"frame_index": 1, "objects": [{"id": 0, "mask": "...", "box": [...], "score": 0.94}]}
...
{"frame_index": 299, "objects": [{"id": 0, "mask": "...", "box": [...], "score": 0.92}]}
{"type": "complete", "total_frames": 300}
```

**Response (non-streaming)**:
```json
{
  "session_id": "vid_abc123def456",
  "results": {
    "0": {"objects": [{"id": 0, "mask": "...", "box": [...]}]},
    "1": {"objects": [{"id": 0, "mask": "...", "box": [...]}]},
    ...
  },
  "total_frames": 300,
  "processing_time_ms": 15000
}
```

---

#### DELETE `/api/v1/video/sessions/{session_id}/objects/{obj_id}`
Удалить объект из tracking

**Request**: Empty body or `{"is_user_action": true}`

**Response**:
```json
{
  "session_id": "vid_abc123def456",
  "obj_id": 0,
  "status": "removed"
}
```

---

#### POST `/api/v1/video/sessions/{session_id}/reset`
Reset video session (сохраняя загруженное видео)

**Response**:
```json
{
  "session_id": "vid_abc123def456",
  "status": "reset",
  "objects_cleared": 3
}
```

---

#### DELETE `/api/v1/video/sessions/{session_id}`
Закрыть и удалить session

**Response**:
```json
{
  "session_id": "vid_abc123def456",
  "status": "closed",
  "memory_freed_mb": 2048
}
```

---

#### GET `/api/v1/video/sessions/{session_id}/status`
Получить статус session

**Response**:
```json
{
  "session_id": "vid_abc123def456",
  "status": "processing",  // "ready", "processing", "error"
  "current_objects": 3,
  "frames_processed": 150,
  "total_frames": 300,
  "gpu_memory_used_mb": 2048
}
```

---

### 3. Session Management API

#### GET `/api/v1/sessions`
Список всех активных sessions

**Response**:
```json
{
  "sessions": [
    {
      "session_id": "vid_abc123",
      "type": "video",
      "created_at": "2026-01-08T10:30:00Z",
      "status": "ready",
      "objects_count": 2
    },
    {
      "session_id": "img_xyz789",
      "type": "image_batch",
      "created_at": "2026-01-08T10:35:00Z",
      "status": "processing",
      "images_processed": 50
    }
  ],
  "total_sessions": 2
}
```

---

### 4. Health & Monitoring API

#### GET `/health`
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-08T10:30:00Z",
  "uptime_seconds": 3600,
  "gpu_available": true,
  "gpu_count": 4,
  "active_sessions": 2
}
```

---

#### GET `/metrics`
Prometheus-style metrics

**Response**:
```
# HELP sam3_requests_total Total number of requests
# TYPE sam3_requests_total counter
sam3_requests_total{endpoint="image_segment",status="success"} 1500
sam3_requests_total{endpoint="video_propagate",status="success"} 300

# HELP sam3_inference_duration_seconds Inference duration
# TYPE sam3_inference_duration_seconds histogram
sam3_inference_duration_seconds_bucket{endpoint="image_segment",le="0.1"} 1200
sam3_inference_duration_seconds_bucket{endpoint="image_segment",le="0.5"} 1500

# HELP sam3_gpu_memory_used_bytes GPU memory used
# TYPE sam3_gpu_memory_used_bytes gauge
sam3_gpu_memory_used_bytes{gpu="0"} 2147483648
```

---

#### GET `/models/info`
Информация о загруженных моделях

**Response**:
```json
{
  "models": {
    "image": {
      "loaded": true,
      "checkpoint": "facebook/sam3",
      "device": "cuda:0",
      "memory_mb": 3500,
      "capabilities": ["text_prompt", "box_prompt", "batch_processing"]
    },
    "video": {
      "loaded": true,
      "checkpoint": "facebook/sam3",
      "devices": ["cuda:1", "cuda:2", "cuda:3"],
      "memory_mb": 8200,
      "capabilities": ["text_prompt", "point_prompt", "box_prompt", "tracking"]
    }
  },
  "server_version": "1.0.0",
  "sam3_version": "1.0.0"
}
```

---

## Структура Проекта

```
model_inference/sam3/
├── README.md                    # Документация сервера
├── requirements.txt             # Зависимости
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Multi-container setup
├── .env.example                 # Пример конфигурации
│
├── server.py                    # FastAPI app entry point
├── config.py                    # Configuration management
│
├── api/                         # API layer
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── image.py             # Image endpoints
│   │   ├── video.py             # Video endpoints
│   │   ├── sessions.py          # Session management
│   │   └── health.py            # Health & metrics
│   ├── schemas/                 # Pydantic models
│   │   ├── __init__.py
│   │   ├── image_schemas.py
│   │   ├── video_schemas.py
│   │   └── common_schemas.py
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py              # API key authentication
│       ├── rate_limit.py        # Rate limiting
│       └── logging.py           # Request logging
│
├── services/                    # Business logic
│   ├── __init__.py
│   ├── image_service.py         # Image inference service
│   ├── video_service.py         # Video inference service
│   ├── session_manager.py       # Session lifecycle
│   └── cache_service.py         # Feature caching
│
├── models/                      # Model layer
│   ├── __init__.py
│   ├── sam3_image.py            # Sam3Processor wrapper
│   ├── sam3_video.py            # Sam3VideoPredictor wrapper
│   ├── model_loader.py          # Model loading & caching
│   └── postprocessing.py        # Mask/box postprocessing
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── image_utils.py           # Image encoding/decoding
│   ├── video_utils.py           # Video processing
│   ├── rle_utils.py             # RLE encoding/decoding
│   └── visualization.py         # Mask visualization
│
├── tests/                       # Tests
│   ├── __init__.py
│   ├── test_image_api.py
│   ├── test_video_api.py
│   ├── test_services.py
│   └── fixtures/
│       ├── sample_image.jpg
│       └── sample_video.mp4
│
└── scripts/                     # Deployment scripts
    ├── start_server.sh
    ├── run_tests.sh
    └── benchmark.py
```

---

## Конфигурация (.env)

```bash
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=4
LOG_LEVEL=INFO

# Model Configuration
SAM3_CHECKPOINT=facebook/sam3
SAM3_BPE_PATH=sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Image Model
IMAGE_MODEL_DEVICE=cuda:0
IMAGE_MODEL_COMPILE=false
IMAGE_MODEL_CONFIDENCE_THRESHOLD=0.5
IMAGE_MODEL_RESOLUTION=1008

# Video Model
VIDEO_MODEL_GPUS=0,1,2,3
VIDEO_MODEL_COMPILE=false
VIDEO_MODEL_TEMPORAL_DISAMBIGUATION=true

# Session Management
MAX_CONCURRENT_SESSIONS=10
SESSION_TIMEOUT_SECONDS=3600
SESSION_CLEANUP_INTERVAL_SECONDS=300

# Cache Configuration
ENABLE_FEATURE_CACHE=true
FEATURE_CACHE_TTL_SECONDS=600
MAX_CACHE_SIZE_MB=4096

# API Keys (optional)
REQUIRE_API_KEY=false
API_KEYS=key1,key2,key3

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Storage
UPLOAD_DIR=/tmp/sam3_uploads
OUTPUT_DIR=/tmp/sam3_outputs
MAX_UPLOAD_SIZE_MB=100

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

---

## Технические Требования

### Hardware

**Минимальные требования** (Image only):
- GPU: NVIDIA with 8GB VRAM (RTX 3060 Ti или выше)
- RAM: 16GB
- Storage: 20GB (для модели + cache)

**Рекомендуемые требования** (Image + Video):
- GPU: 4x NVIDIA A100 40GB или 4x H100 80GB
- RAM: 64GB
- Storage: 100GB SSD

### Software

- Python 3.12+
- CUDA 12.6+
- PyTorch 2.7+
- FastAPI 0.115+
- Docker 24+ (для контейнеризации)

---

## Performance Benchmarks

### Image Inference

| Metric | Single Image | Batch (16 images) |
|--------|-------------|-------------------|
| Latency (GPU H100) | ~100ms | ~800ms (50ms/img) |
| Throughput | 10 img/s | 20 img/s |
| GPU Memory | 3.5GB | 5GB |

### Video Inference

| Metric | 300 frames, 1 object | 300 frames, 5 objects |
|--------|---------------------|----------------------|
| Latency (4x H100) | ~15s | ~25s |
| Throughput | 20 fps | 12 fps |
| GPU Memory | 8GB | 12GB |

---

## Deployment Options

### 1. Local Development
```bash
python server.py --host 0.0.0.0 --port 8000
```

### 2. Docker Single Container
```bash
docker build -t sam3-server .
docker run -p 8000:8000 --gpus all sam3-server
```

### 3. Docker Compose (Multi-GPU)
```bash
docker-compose up -d
```

### 4. Kubernetes
```yaml
# Helm chart для k8s deployment
# С GPU node affinity и autoscaling
```

---

## Security Considerations

1. **API Key Authentication**: Optional but recommended
2. **Rate Limiting**: Предотвращение DDoS
3. **Input Validation**: Pydantic schemas для всех входов
4. **File Upload Limits**: Max 100MB per request
5. **Session Isolation**: Изоляция между sessions
6. **CORS Configuration**: Настраиваемые allowed origins

---

## Monitoring & Observability

### Metrics Collection
- Prometheus endpoint на `/metrics`
- Grafana dashboard для визуализации
- Custom metrics:
  - Request latency по endpoint
  - GPU memory usage
  - Active sessions count
  - Error rates

### Logging
- Structured JSON logging
- Log levels: DEBUG, INFO, WARNING, ERROR
- Request/response logging
- Error stack traces

### Tracing (опционально)
- OpenTelemetry integration
- Distributed tracing для debug

---

## Next Steps

### Phase 1: MVP ✅
- [ ] Базовая FastAPI структура
- [ ] Image inference endpoints
- [ ] Sam3Processor integration
- [ ] Basic health checks

### Phase 2: Video Support
- [ ] Video session management
- [ ] Sam3VideoPredictor integration
- [ ] Propagation endpoints
- [ ] WebSocket streaming

### Phase 3: Production Ready
- [ ] Feature caching
- [ ] API authentication
- [ ] Rate limiting
- [ ] Docker deployment
- [ ] Metrics & monitoring

### Phase 4: Optimization
- [ ] Torch compile optimization
- [ ] Multi-GPU load balancing
- [ ] Connection pooling
- [ ] Response compression

---

## References

- **SAM3 Paper**: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- **SAM3 GitHub**: https://github.com/facebookresearch/sam3
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Project Docs**: `D:\Projects\Sam_agent\docs\SAM3_INTEGRATION.md`

---

**Готово к реализации!** 🚀
