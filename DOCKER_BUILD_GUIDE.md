# Docker Build Guide - SAM3 Inference Server

## Обзор

Предоставлены два варианта Docker-сборки в зависимости от источника репозитория SAM3:

| Файл | Источник SAM3 | Запуск из директории | Применение |
|------|---------------|---------------------|------------|
| `Dockerfile` | GitHub (клонирует во время сборки) | `model_inference/sam3` | Production, CI/CD |
| `Dockerfile.local` | Локальная копия проекта | Корень проекта | Development, offline |

## Option 1: GitHub SAM3 (Recommended)

### Характеристики
- ✅ Всегда использует актуальную версию SAM3
- ✅ Не зависит от локального репозитория
- ✅ Подходит для production и CI/CD
- ⚠️ Требует интернет-соединения при сборке

### Сборка

```bash
# Перейти в директорию inference сервера
cd D:\Projects\Sam_agent\model_inference\sam3

# Собрать image
docker build -t sam3-inference:latest .

# Проверить
docker images | grep sam3-inference
```

### Запуск

**Вариант A: Docker run**
```bash
docker run -d \
  --name sam3-server \
  --gpus all \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/outputs:/tmp/sam3_outputs \
  sam3-inference:latest
```

**Вариант B: Docker Compose**
```bash
cd D:\Projects\Sam_agent\model_inference\sam3
docker-compose up -d
```

## Option 2: Local SAM3

### Характеристики
- ✅ Использует локальную копию SAM3
- ✅ Работает без интернета
- ✅ Подходит для development
- ⚠️ Требует наличия `D:\Projects\Sam_agent\sam3`

### Предварительные требования

Убедитесь, что SAM3 склонирован в проекте:

```bash
cd D:\Projects\Sam_agent

# Если sam3 ещё не склонирован
git clone https://github.com/facebookresearch/sam3.git sam3

# Проверить структуру
ls sam3/sam3  # Должны быть model/, predictor/, etc.
```

### Сборка

**ВАЖНО**: Сборка должна запускаться из корня проекта!

```bash
# Перейти в корень проекта
cd D:\Projects\Sam_agent

# Собрать image с локальной копией SAM3
docker build -f model_inference/sam3/Dockerfile.local -t sam3-inference:latest .

# Проверить
docker images | grep sam3-inference
```

### Запуск

**Вариант A: Docker run**
```bash
cd D:\Projects\Sam_agent

docker run -d \
  --name sam3-server \
  --gpus all \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/model_inference/sam3/outputs:/tmp/sam3_outputs \
  sam3-inference:latest
```

**Вариант B: Docker Compose**
```bash
cd D:\Projects\Sam_agent

docker-compose -f model_inference/sam3/docker-compose.local.yml up -d
```

## Структура проекта

```
D:\Projects\Sam_agent\
├── sam3/                          # SAM3 репозиторий (для Dockerfile.local)
│   ├── sam3/
│   │   ├── model/
│   │   ├── predictor/
│   │   └── assets/
│   └── setup.py
│
└── model_inference/
    └── sam3/                      # Inference сервер
        ├── Dockerfile             # GitHub SAM3
        ├── Dockerfile.local       # Local SAM3
        ├── docker-compose.yml     # для Dockerfile
        ├── docker-compose.local.yml  # для Dockerfile.local
        ├── server.py
        ├── config.py
        ├── requirements.txt
        └── ...
```

## Проверка работы

После запуска контейнера:

```bash
# Проверить health endpoint
curl http://localhost:8000/health

# Проверить логи
docker logs sam3-server

# Swagger UI
open http://localhost:8000/docs
```

Ожидаемый ответ:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 4,
  "image_model_enabled": true,
  "video_model_enabled": true
}
```

## Troubleshooting

### Ошибка: "COPY ../../sam3 /app/sam3" failed

**Проблема**: Использовали `Dockerfile.local` из неправильной директории.

**Решение**:
```bash
# Правильно
cd D:\Projects\Sam_agent
docker build -f model_inference/sam3/Dockerfile.local -t sam3-inference .

# Неправильно
cd D:\Projects\Sam_agent\model_inference\sam3
docker build -f Dockerfile.local -t sam3-inference .  # ❌ Ошибка!
```

### Ошибка: "git clone failed" в Dockerfile

**Проблема**: Нет интернета или GitHub недоступен.

**Решение**: Используйте `Dockerfile.local` с локальной копией SAM3.

### Ошибка: No module named 'sam3'

**Проблема**: SAM3 не установлен корректно в контейнере.

**Решение**:
1. Проверьте, что SAM3 существует в нужной директории
2. Пересоберите image с флагом `--no-cache`:
   ```bash
   docker build --no-cache -t sam3-inference:latest .
   ```

### CUDA Out of Memory

**Решение**: Уменьшите количество GPUs или разрешение:

```bash
docker run -d \
  --gpus '"device=0"' \
  -e IMAGE_MODEL_RESOLUTION=896 \
  -e VIDEO_MODEL_GPUS=0 \
  -p 8000:8000 \
  sam3-inference:latest
```

## Переменные окружения

Основные переменные для Docker run:

```bash
docker run -d \
  --name sam3-server \
  --gpus all \
  -p 8000:8000 \
  -e SERVER_HOST=0.0.0.0 \
  -e SERVER_PORT=8000 \
  -e IMAGE_MODEL_DEVICE=cuda:0 \
  -e VIDEO_MODEL_ENABLED=true \
  -e VIDEO_MODEL_GPUS=0,1,2,3 \
  -e LOG_LEVEL=INFO \
  -e SAM3_CHECKPOINT=facebook/sam3 \
  sam3-inference:latest
```

Полный список см. в [.env.example](.env.example)

## Production Deployment

Для production рекомендуется:

1. **Использовать GitHub SAM3** (`Dockerfile`) для стабильности
2. **Multi-stage build** для уменьшения размера image
3. **Docker secrets** для API ключей
4. **Health checks** в Kubernetes/Docker Swarm
5. **Volume mounts** для outputs и uploads

См. [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) для деталей.

## Сравнение вариантов

| Критерий | Dockerfile (GitHub) | Dockerfile.local (Local) |
|----------|---------------------|--------------------------|
| Интернет требуется | ✅ Да (при сборке) | ❌ Нет |
| Размер image | ~8GB | ~8GB |
| Время сборки | ~15-20 мин | ~10-15 мин |
| Актуальность SAM3 | Всегда актуален | Зависит от локальной копии |
| Production ready | ✅ Да | ⚠️ Для dev |
| CI/CD friendly | ✅ Да | ❌ Нет |

## Рекомендации

- **Production**: Используйте `Dockerfile` (GitHub SAM3)
- **Development**: Используйте `Dockerfile.local` для быстрой итерации
- **Offline**: Используйте `Dockerfile.local` с предварительно скачанным SAM3

---

**Примечание**: Оба варианта создают идентичный runtime контейнер. Разница только в источнике SAM3 при сборке.
