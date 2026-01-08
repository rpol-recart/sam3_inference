# Docker Setup Summary

## Исправлено

Проблема с `COPY ../../sam3 /app/sam3` в Dockerfile решена созданием двух вариантов сборки.

## Доступные файлы

### Dockerfiles

1. **`Dockerfile`** - Клонирует SAM3 из GitHub
   - Запуск: `cd model_inference/sam3 && docker build -t sam3-inference .`
   - Применение: Production, CI/CD
   - Интернет: Требуется при сборке

2. **`Dockerfile.local`** - Использует локальную копию SAM3
   - Запуск: `cd D:\Projects\Sam_agent && docker build -f model_inference/sam3/Dockerfile.local -t sam3-inference .`
   - Применение: Development, offline
   - Интернет: Не требуется

### Docker Compose

1. **`docker-compose.yml`** - Для `Dockerfile`
   ```bash
   cd D:\Projects\Sam_agent\model_inference\sam3
   docker-compose up -d
   ```

2. **`docker-compose.local.yml`** - Для `Dockerfile.local`
   ```bash
   cd D:\Projects\Sam_agent
   docker-compose -f model_inference/sam3/docker-compose.local.yml up -d
   ```

## Quick Start

### Production (GitHub SAM3)
```bash
cd D:\Projects\Sam_agent\model_inference\sam3
docker-compose up -d
curl http://localhost:8000/health
```

### Development (Local SAM3)
```bash
cd D:\Projects\Sam_agent
docker-compose -f model_inference/sam3/docker-compose.local.yml up -d
curl http://localhost:8000/health
```

## Структура

```
D:\Projects\Sam_agent\
├── sam3/                                    # SAM3 репозиторий
│   └── sam3/
│       ├── model/
│       └── assets/
│
└── model_inference/
    └── sam3/                                # Inference сервер
        ├── Dockerfile                       # GitHub SAM3 ✅
        ├── Dockerfile.local                 # Local SAM3 ✅
        ├── docker-compose.yml               # для Dockerfile ✅
        ├── docker-compose.local.yml         # для Dockerfile.local ✅
        ├── DOCKER_BUILD_GUIDE.md            # Детальный гайд ✅
        └── ...
```

## Документация

- **[DOCKER_BUILD_GUIDE.md](DOCKER_BUILD_GUIDE.md)** - Полное руководство по сборке
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[README.md](README.md)** - Основная документация

## Проверка

После запуска:
```bash
# Health check
curl http://localhost:8000/health

# Swagger UI
open http://localhost:8000/docs

# Logs
docker logs sam3-inference
```

Ожидаемый ответ:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "image_model_enabled": true,
  "video_model_enabled": true
}
```
