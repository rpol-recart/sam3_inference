"""Health check and monitoring endpoints."""
import psutil
import torch
from fastapi import APIRouter
from pydantic import BaseModel

from config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    gpu_available: bool
    gpu_count: int
    active_sessions: int = 0


class ModelInfo(BaseModel):
    """Model information."""

    loaded: bool
    checkpoint: str
    device: str
    memory_mb: float
    capabilities: list[str]


class ModelsInfoResponse(BaseModel):
    """Models information response."""

    image_model: ModelInfo
    server_version: str
    sam3_version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        active_sessions=0,  # TODO: get from session manager
    )


@router.get("/models/info", response_model=ModelsInfoResponse)
async def models_info():
    """Get information about loaded models."""
    import server

    image_info = ModelInfo(
        loaded=server.image_model is not None,
        checkpoint=settings.sam3_checkpoint,
        device=settings.image_model_device,
        memory_mb=0.0,  # TODO: calculate actual memory
        capabilities=["text_prompt", "box_prompt", "batch_processing", "feature_caching"],
    )

    if server.image_model and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_bytes = torch.cuda.memory_allocated(device=settings.image_model_device)
        image_info.memory_mb = memory_bytes / (1024 * 1024)

    return ModelsInfoResponse(
        image_model=image_info,
        server_version="1.0.0",
        sam3_version="1.0.0",
    )


@router.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint."""
    metrics_text = []

    # GPU metrics
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)

            metrics_text.append(
                f'sam3_gpu_memory_allocated_bytes{{gpu="{i}"}} {memory_allocated}'
            )
            metrics_text.append(
                f'sam3_gpu_memory_reserved_bytes{{gpu="{i}"}} {memory_reserved}'
            )

    # System metrics
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    metrics_text.append(f"sam3_cpu_usage_percent {cpu_percent}")
    metrics_text.append(f"sam3_memory_usage_percent {memory_percent}")

    return "\n".join(metrics_text)
