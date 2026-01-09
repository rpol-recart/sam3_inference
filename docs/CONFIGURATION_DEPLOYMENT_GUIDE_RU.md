# Руководство по конфигурации и развертыванию SAM3 Inference Server

## Оглавление
1. [Конфигурация приложения](#конфигурация-приложения)
2. [Переменные окружения](#переменные-окружения)
3. [Развертывание](#развертывание)
4. [Масштабирование](#масштабирование)
5. [Мониторинг и логирование](#мониторинг-и-логирование)
6. [Безопасность](#безопасность)

## Конфигурация приложения

Конфигурация приложения осуществляется через Pydantic-модель `Settings` в файле `app/core/config.py`. Все настройки могут быть переопределены через переменные окружения.

### Архитектура конфигурации

Класс `Settings` наследует `BaseSettings` из `pydantic_settings` и предоставляет:

- Валидацию значений
- Типизацию параметров
- Описание параметров
- Значения по умолчанию

```python
class Settings(BaseSettings):
    # Серверные настройки
    server_host: str = Field(default="0.0.0.0", description="Хост сервера")
    server_port: int = Field(default=8000, description="Порт сервера")
    # ... другие настройки
```

## Переменные окружения

### Серверные настройки

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `SERVER_HOST` | `0.0.0.0` | IP-адрес, на котором будет запущен сервер |
| `SERVER_PORT` | `8000` | Порт, на котором будет работать сервер |
| `SERVER_WORKERS` | `1` | Количество рабочих процессов Gunicorn |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `RELOAD` | `False` | Включить горячую перезагрузку (для разработки) |

### Настройки моделей

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `SAM3_CHECKPOINT` | `/app/server/sam_weights/sam3.pt` | Путь к чекпоинту модели SAM3 |
| `SAM3_BPE_PATH` | `/app/server/sam_weights/bpe_simple_vocab_16e6.txt.gz` | Путь к файлу токенизатора BPE |

### Настройки модели изображений

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `IMAGE_MODEL_DEVICE` | `cuda:0` | Устройство для модели изображений |
| `IMAGE_MODEL_COMPILE` | `False` | Включить torch.compile оптимизацию |
| `IMAGE_MODEL_CONFIDENCE_THRESHOLD` | `0.5` | Порог уверенности для фильтрации |
| `IMAGE_MODEL_RESOLUTION` | `1008` | Разрешение входного изображения |
| `IMAGE_MODEL_ENABLED` | `True` | Включить модель изображений |

### Настройки модели видео

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `VIDEO_MODEL_ENABLED` | `True` | Включить модель видео |
| `VIDEO_MODEL_REQUIRED` | `False` | Требовать загрузку видео модели |
| `VIDEO_GPU_LIST` | `[0, 1, 2, 3]` | Список GPU ID для видео обработки |
| `VIDEO_MODEL_COMPILE` | `False` | Включить torch.compile для видео |
| `VIDEO_MODEL_TEMPORAL_DISAMBIGUATION` | `True` | Включить временную дизамбигуацию |

### Управление сессиями

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MAX_CONCURRENT_SESSIONS` | `10` | Максимальное количество сессий |
| `SESSION_TIMEOUT_SECONDS` | `3600` | Таймаут сессии в секундах |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `300` | Интервал очистки сессий |

### Кэширование

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ENABLE_FEATURE_CACHE` | `True` | Включить кэширование признаков |
| `FEATURE_CACHE_TTL_SECONDS` | `600` | Время жизни кэша в секундах |
| `MAX_CACHE_SIZE_MB` | `4096` | Максимальный размер кэша в МБ |

### Безопасность

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `REQUIRE_API_KEY` | `False` | Требовать API-ключ для запросов |
| `API_KEYS` | `["your-api-key-1", "your-api-key-2"]` | Список валидных API-ключей |
| `CORS_ORIGINS_LIST` | `["*"]` | Список разрешенных CORS источников |

### Ограничение частоты

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `RATE_LIMIT_ENABLED` | `True` | Включить ограничение частоты |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `100` | Максимальное количество запросов в минуту |

### Хранилище

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `UPLOAD_DIR` | `/tmp/sam3_uploads` | Директория для загрузки файлов |
| `OUTPUT_DIR` | `/tmp/sam3_outputs` | Директория для выходных файлов |
| `MAX_UPLOAD_SIZE_MB` | `100` | Максимальный размер загрузки в МБ |

### Мониторинг

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ENABLE_METRICS` | `True` | Включить метрики |
| `METRICS_PORT` | `9090` | Порт для метрик |

## Развертывание

### Локальное развертывание

#### Требования:
- Python 3.9+
- CUDA (для GPU-ускорения)
- PyTorch с поддержкой CUDA (если используется GPU)

#### Шаги:
1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd sam3_inference
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Подготовьте веса модели:
```bash
mkdir -p sam_weights
# Скопируйте веса модели в папку sam_weights
```

4. Создайте файл `.env` с конфигурацией:
```env
SAM3_CHECKPOINT=/path/to/sam3.pt
SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
IMAGE_MODEL_DEVICE=cuda:0
VIDEO_GPU_LIST=[0]
```

5. Запустите сервер:
```bash
python -m app.main
```

### Docker развертывание

#### Сборка образа:
```bash
docker build -t sam3-inference-server .
```

#### Запуск контейнера:
```bash
docker run -d \
  --name sam3-server \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/weights:/app/server/sam_weights \
  -v /path/to/uploads:/tmp/sam3_uploads \
  -v /path/to/outputs:/tmp/sam3_outputs \
  -e SAM3_CHECKPOINT=/app/server/sam_weights/sam3.pt \
  -e IMAGE_MODEL_DEVICE=cuda:0 \
  sam3-inference-server
```

#### Параметры Docker:
- `--gpus all` - доступ ко всем GPU
- `-p 8000:8000` - проброс порта
- `-v` - монтирование томов для весов и данных
- `-e` - переменные окружения

### Docker Compose

Файл `docker-compose.yml`:

```yaml
version: '3.8'

services:
  sam3-server:
    build: .
    container_name: sam3-server
    ports:
      - "8000:8000"
    volumes:
      - ./sam_weights:/app/server/sam_weights
      - ./uploads:/tmp/sam3_uploads
      - ./outputs:/tmp/sam3_outputs
    environment:
      - SAM3_CHECKPOINT=/app/server/sam_weights/sam3.pt
      - SAM3_BPE_PATH=/app/server/sam_weights/bpe_simple_vocab_16e6.txt.gz
      - IMAGE_MODEL_DEVICE=cuda:0
      - VIDEO_GPU_LIST=[0]
      - MAX_CONCURRENT_SESSIONS=5
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Запуск:
```bash
docker-compose up -d
```

### Kubernetes развертывание

Пример манифеста `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sam3-inference-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sam3-server
  template:
    metadata:
      labels:
        app: sam3-server
    spec:
      containers:
      - name: sam3-server
        image: sam3-inference-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: SAM3_CHECKPOINT
          value: "/app/server/sam_weights/sam3.pt"
        - name: IMAGE_MODEL_DEVICE
          value: "cuda:0"
        - name: VIDEO_GPU_LIST
          value: "[0]"
        volumeMounts:
        - name: weights-volume
          mountPath: /app/server/sam_weights
        - name: uploads-volume
          mountPath: /tmp/sam3_uploads
        - name: outputs-volume
          mountPath: /tmp/sam3_outputs
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
      volumes:
      - name: weights-volume
        persistentVolumeClaim:
          claimName: weights-pvc
      - name: uploads-volume
        persistentVolumeClaim:
          claimName: uploads-pvc
      - name: outputs-volume
        persistentVolumeClaim:
          claimName: outputs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: sam3-service
spec:
  selector:
    app: sam3-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Масштабирование

### Горизонтальное масштабирование

Для увеличения пропускной способности:

1. **Запустите несколько экземпляров**:
```bash
# Запуск нескольких контейнеров
docker run -d --name sam3-server-1 ...
docker run -d --name sam3-server-2 ...
```

2. **Используйте балансировщик нагрузки**:
```nginx
upstream sam3_backend {
    server sam3-server-1:8000;
    server sam3-server-2:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://sam3_backend;
    }
}
```

### Вертикальное масштабирование

Для увеличения производительности одного экземпляра:

1. **Использование нескольких GPU**:
```env
VIDEO_GPU_LIST=[0,1,2,3]
SERVER_WORKERS=4
```

2. **Оптимизация памяти**:
```env
MAX_CONCURRENT_SESSIONS=20
FEATURE_CACHE_TTL_SECONDS=300
MAX_CACHE_SIZE_MB=8192
```

### Ограничения масштабирования

- **Память GPU**: Каждый экземпляр использует значительное количество GPU памяти
- **Сессии**: Видео сессии занимают память до закрытия
- **Модельные ограничения**: SAM3 имеет ограничения по размеру изображений

## Мониторинг и логирование

### Логирование

Сервер использует `loguru` для логирования:

```python
from loguru import logger

logger.info("Сообщение информационного уровня")
logger.error("Сообщение об ошибке")
logger.debug("Отладочное сообщение")
```

#### Уровни логирования:
- `TRACE` - самые подробные сообщения
- `DEBUG` - отладочная информация
- `INFO` - общая информация о работе
- `SUCCESS` - успешные операции
- `WARNING` - предупреждения
- `ERROR` - ошибки
- `CRITICAL` - критические ошибки

### Метрики

Сервер может предоставлять метрики через отдельный порт:

```env
ENABLE_METRICS=True
METRICS_PORT=9090
```

### Мониторинг производительности

Для мониторинга используйте:
- `nvidia-smi` - мониторинг GPU
- `htop` - мониторинг CPU/памяти
- `docker stats` - мониторинг контейнеров

### Логирование запросов

Создайте middleware для логирования HTTP-запросов:

```python
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
        
        return response
```

## Безопасность

### API-ключи

Для защиты API используйте ключи:

```env
REQUIRE_API_KEY=True
API_KEYS=["key1", "key2", "key3"]
```

Используйте заголовок в запросах:
```
X-API-Key: your-api-key
```

### CORS

Настройте разрешенные источники:

```env
CORS_ORIGINS_LIST=["https://yourdomain.com", "https://anotherdomain.com"]
```

### Rate Limiting

Ограничение частоты запросов:

```env
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

### HTTPS

Для продакшена используйте HTTPS через reverse proxy:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Защита от DDoS

- Используйте rate limiting
- Ограничьте размер загружаемых файлов
- Установите таймауты для длительных операций
- Используйте CDN для статических ресурсов

## Рекомендации по развертыванию

### Для разработки:
- Используйте `RELOAD=True` для горячей перезагрузки
- Установите `LOG_LEVEL=DEBUG` для подробного логирования
- Используйте CPU вместо GPU для тестирования

### Для тестирования:
- Используйте небольшие модели для тестирования
- Проверьте работу сессий
- Протестируйте граничные условия

### Для продакшена:
- Используйте GPU для высокой производительности
- Настройте мониторинг и алертинг
- Используйте HTTPS и API-ключи
- Настройте резервное копирование
- Используйте контейнеризацию для изоляции