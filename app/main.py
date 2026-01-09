"""Main FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.middleware.security import SecurityHeadersMiddleware, RateLimitMiddleware
from app.api.v1.routes import (
    health_router,
    image_router,
    video_router
)

# Configure logging
logger = configure_logging(
    log_level=settings.log_level,
)


# Global model instances
image_model = None
video_model = None
session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for model loading/unloading."""
    global image_model, video_model, session_manager

    # Startup: Load models
    logger.info("Starting SAM3 Inference Server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Ensure directories exist
    settings.ensure_directories()

    # Load image model if enabled
    if settings.image_model_enabled:
        try:
            from app.models.sam3_image import SAM3ImageModel

            logger.info("Loading SAM3 image model...")
            image_model = SAM3ImageModel(
                checkpoint=settings.sam3_checkpoint,
                bpe_path=settings.sam3_bpe_path,
                device=settings.image_model_device,
                confidence_threshold=settings.image_model_confidence_threshold,
                resolution=settings.image_model_resolution,
                compile=settings.image_model_compile,
            )
            logger.info("✓ Image model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise

    # Load video model if enabled
    if settings.video_model_enabled:
        try:
            from app.models.sam3_video import SAM3VideoModel
            from app.services.session_manager import SessionManager

            logger.info("Loading SAM3 video model...")
            video_model = SAM3VideoModel(
                checkpoint=settings.sam3_checkpoint,
                bpe_path=settings.sam3_bpe_path,
                gpu_ids=settings.video_gpu_list,
                video_loader_type="cv2",
                async_loading_frames=False,
            )

            # Initialize session manager
            session_manager = SessionManager(
                max_sessions=settings.max_concurrent_sessions,
                session_timeout_seconds=3600,
            )

            logger.info("✓ Video model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            if settings.video_model_required:
                raise
            else:
                logger.warning("Video inference will be disabled")

    # Store models in app state
    app.state.image_model = image_model
    app.state.video_model = video_model
    app.state.session_manager = session_manager

    logger.info("Server startup complete")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down server...")
    if image_model:
        image_model.clear_cache()
    if video_model:
        video_model.shutdown()
    if session_manager:
        session_manager.clear_all_sessions()
    logger.info("Server shutdown complete")


# Add middleware
app = FastAPI(
    title="SAM3 Inference Server",
    description="FastAPI server for SAM3 (Segment Anything 3) image and video inference",
    version="1.0.0",
    lifespan=lifespan,
)

# Security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware (only if enabled in settings)
if settings.rate_limit_enabled:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(image_router, prefix="/api/v1/image", tags=["Image"])
app.include_router(video_router, prefix="/api/v1/video", tags=["Video"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAM3 Inference Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def main():
    """Run the server."""
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.reload,
        workers=settings.server_workers if not settings.reload else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()