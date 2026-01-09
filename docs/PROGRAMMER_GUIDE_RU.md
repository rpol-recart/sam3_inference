# Руководство программиста для SAM3 Inference Server

## Оглавление
1. [Структура проекта](#структура-проекта)
2. [Архитектурные шаблоны](#архитектурные-шаблоны)
3. [Разработка новых функций](#разработка-новых-функций)
4. [Тестирование](#тестирование)
5. [Расширение API](#расширение-api)
6. [Обработка ошибок](#обработка-ошибок)
7. [Логирование и мониторинг](#логирование-и-мониторинг)

## Структура проекта

Проект организован по принципам Clean Architecture:

```
sam3_inference/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Точка входа FastAPI приложения
│   ├── core/
│   │   ├── config.py          # Конфигурация приложения
│   │   └── logging.py         # Настройка логирования
│   ├── api/
│   │   ├── dependencies.py    # Зависимости FastAPI
│   │   └── v1/
│   │       └── routes/        # Маршруты API v1
│   ├── services/              # Сервисы бизнес-логики
│   ├── models/                # Обертки для моделей ML
│   ├── schemas/               # Pydantic схемы
│   └── middleware/            # Middleware
├── services/
│   └── session_manager.py     # Управление сессиями
├── api/                       # Альтернативные маршруты (legacy)
├── models/                    # Модели данных
├── docs/                      # Документация
├── tests/                     # Тесты
└── requirements.txt           # Зависимости
```

### Ключевые файлы:

- `app/main.py` - точка входа приложения, инициализация моделей
- `app/core/config.py` - централизованная конфигурация
- `app/api/dependencies.py` - управление зависимостями FastAPI
- `app/services/*.py` - бизнес-логика
- `app/models/*.py` - обертки для моделей SAM3
- `app/schemas/*.py` - валидация данных

## Архитектурные шаблоны

### Dependency Injection

Сервер использует внедрение зависимостей FastAPI для управления сервисами:

```python
def get_image_service(request: Request) -> ImageSegmentationService:
    model = get_image_model(request)
    return ImageSegmentationService(model)
```

### Service Layer Pattern

Бизнес-логика изолирована в сервисах:

```python
class ImageSegmentationService:
    def __init__(self, model: SAM3ImageModel):
        self.model = model

    def segment_image(self, request: ImageSegmentRequest) -> ImageSegmentResponse:
        # Реализация бизнес-логики
        pass
```

### Repository Pattern

Для управления сессиями используется репозиторий:

```python
class SessionManager:
    def create_session(self, session_id: str, ...) -> Dict:
        # Логика управления сессиями
        pass
```

### DTO Pattern

Pydantic схемы используются как DTO для валидации данных:

```python
class ImageSegmentRequest(BaseModel):
    image: str
    prompts: List[Prompt]
    confidence_threshold: float = 0.5
```

## Разработка новых функций

### Добавление нового эндпоинта

1. Создайте схему запроса и ответа:

```python
from pydantic import BaseModel, Field

class NewFeatureRequest(BaseModel):
    param1: str = Field(..., description="Описание параметра")

class NewFeatureResponse(BaseModel):
    result: str
    success: bool = True
```

2. Реализуйте метод в сервисе:

```python
class ImageSegmentationService:
    def new_feature_method(self, request: NewFeatureRequest) -> NewFeatureResponse:
        # Логика обработки
        return NewFeatureResponse(result="результат")
```

3. Создайте маршрут в соответствующем файле:

```python
from fastapi import APIRouter, Depends

router = APIRouter()

@router.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature_endpoint(
    request: NewFeatureRequest,
    service: ImageSegmentationService = Depends(get_image_service)
):
    return service.new_feature_method(request)
```

4. Подключите маршрут в `app/main.py`:

```python
from app.api.v1.routes.image import router as image_router

app.include_router(image_router, prefix="/api/v1/image", tags=["Image"])
```

### Расширение модели SAM3

Чтобы добавить новые возможности в обертку модели:

1. Добавьте новый метод в соответствующий класс модели:

```python
class SAM3ImageModel:
    def new_segmentation_method(self, image: Image.Image, ...):
        # Новая функциональность
        pass
```

2. Обновите сервис, чтобы использовать новый метод:

```python
class ImageSegmentationService:
    def new_method(self, ...):
        return self.model.new_segmentation_method(...)
```

### Добавление нового типа промпта

1. Создайте новую Pydantic-модель для промпта:

```python
class NewPrompt(BaseModel):
    type: Literal["new_type"] = "new_type"
    new_param: str = Field(..., description="Новый параметр")
```

2. Обновите Union-тип для промптов:

```python
Prompt = TextPrompt | PointPrompt | BoxPrompt | NewPrompt
```

3. Добавьте обработку нового типа в сервисе:

```python
if prompt.type == "new_type":
    # Обработка нового типа промпта
    pass
```

## Тестирование

### Unit-тесты

Unit-тесты находятся в директории `tests/`:

```python
import pytest
from app.services.image_service import ImageSegmentationService
from app.models.sam3_image import SAM3ImageModel

def test_segment_image_with_text_prompt():
    # Mock модели для тестирования
    mock_model = Mock(spec=SAM3ImageModel)
    service = ImageSegmentationService(mock_model)
    
    # Подготовка данных
    request = ImageSegmentRequest(...)
    
    # Вызов метода
    response = service.segment_image(request)
    
    # Проверка результатов
    assert response.num_masks > 0
```

### Integration-тесты

Интеграционные тесты проверяют работу эндпоинтов:

```python
from fastapi.testclient import TestClient
from app.main import app

def test_image_segmentation_endpoint():
    client = TestClient(app)
    
    # Подготовка данных
    payload = {
        "image": "base64...",
        "prompts": [{"type": "text", "text": "object"}]
    }
    
    # Вызов эндпоинта
    response = client.post("/api/v1/image/segment", json=payload)
    
    # Проверка ответа
    assert response.status_code == 200
    assert "masks" in response.json()
```

### Запуск тестов

```bash
# Запуск всех тестов
pytest

# Запуск с детальным выводом
pytest -v

# Запуск конкретного файла
pytest tests/test_image_api.py

# Запуск с покрытием кода
pytest --cov=app
```

## Расширение API

### WebSocket-эндпоинты

Для стриминга данных используйте WebSocket:

```python
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws/new-stream/{param}")
async def new_stream_endpoint(websocket: WebSocket, param: str):
    await websocket.accept()
    
    try:
        # Получение данных
        request_data = await websocket.receive_json()
        
        # Обработка
        for data in generate_stream_data():
            await websocket.send_json(data)
    except WebSocketDisconnect:
        # Обработка отключения
        pass
    finally:
        await websocket.close()
```

### Async/await шаблоны

Используйте асинхронные функции для операций ввода-вывода:

```python
async def async_endpoint():
    # Асинхронная обработка
    result = await process_async()
    return result

async def process_async():
    # Асинхронные операции
    await asyncio.sleep(1)
    return "результат"
```

### Batch-обработка

Для обработки нескольких элементов за раз:

```python
async def batch_process(items: List[Item]) -> BatchResponse:
    semaphore = asyncio.Semaphore(settings.max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            return await process_single_item(item)
    
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return BatchResponse(results=results)
```

## Обработка ошибок

### Кастомные исключения

Создавайте специфичные исключения:

```python
class ModelNotLoadedError(Exception):
    def __init__(self):
        super().__init__("Model not loaded")

class SessionNotFoundError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} not found")
        self.detail = f"Session {session_id} not found"
```

### Глобальные обработчики исключений

Добавьте обработчики в `app/main.py`:

```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content={"error": "Model not loaded"}
    )
```

### Валидация входных данных

Pydantic автоматически валидирует входные данные:

```python
class ValidatedRequest(BaseModel):
    # Валидация диапазона
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    
    # Валидация длины списка
    prompts: List[Prompt] = Field(min_length=1)
    
    # Валидация минимального значения
    frame_index: int = Field(ge=0)
```

## Логирование и мониторинг

### Логирование в сервисах

Используйте loguru для логирования:

```python
from loguru import logger

class ImageSegmentationService:
    def segment_image(self, request: ImageSegmentRequest):
        logger.info(f"Processing image of size {request.image[:20]}...")
        
        try:
            result = self.model.segment(...)
            logger.success(f"Segmentation completed: {len(result.masks)} masks")
            return result
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
```

### Middleware для логирования

Создайте middleware для логирования запросов:

```python
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
        
        return response
```

### Метрики производительности

Для мониторинга производительности:

```python
import time

def measure_performance(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
        return result
    return wrapper
```

### Отладка и трассировка

Для отладки сложных проблем:

```python
import traceback

def debug_error_handling(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper
```

## Лучшие практики

### Обработка больших файлов

Для работы с большими файлами:

```python
from tempfile import SpooledTemporaryFile

def handle_large_file(upload_file):
    # Используйте временные файлы для больших данных
    with SpooledTemporaryFile(max_size=1024*1024) as tmp_file:
        # Обработка файла
        pass
```

### Управление памятью

Для эффективного использования памяти:

```python
class SAM3ImageModel:
    def clear_cache(self, cache_key: Optional[str] = None):
        """Очистка кэша для управления памятью"""
        if cache_key:
            self.feature_cache.pop(cache_key, None)
        else:
            self.feature_cache.clear()
```

### Обработка параллельных запросов

Для обработки конкурентных запросов:

```python
import asyncio
from asyncio import Semaphore

class ConcurrentProcessor:
    def __init__(self, max_concurrent: int = 4):
        self.semaphore = Semaphore(max_concurrent)
    
    async def process_with_limit(self, item):
        async with self.semaphore:
            return await self.process_item(item)
```

### Управление сессиями

Для корректного управления сессиями:

```python
class SessionManager:
    def cleanup_expired_sessions(self):
        """Очистка просроченных сессий"""
        current_time = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_accessed > self.timeout
        ]
        
        for sid in expired:
            self.delete_session(sid)
```

Это руководство поможет вам эффективно разрабатывать и расширять функциональность SAM3 Inference Server.