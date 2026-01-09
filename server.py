"""SAM3 Inference Server - FastAPI Application."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level,
)

# Global model instances - commented out as they are replaced with app.state
# image_model = None
# video_model = None
# session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading/unloading."""
    # Initialize state variables
    app.state.image_model = None
    app.state.video_model = None
    app.state.session_manager = None

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
            from models.sam3_image import SAM3ImageModel

            logger.info(f"Loading SAM3 image model...{settings.sam3_checkpoint},{settings.sam3_bpe_path}")
            app.state.image_model = SAM3ImageModel(
                checkpoint='/app/server/sam_weights/sam3.pt',
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
            from models.sam3_video import SAM3VideoModel
            from services.session_manager import SessionManager

            logger.info("Loading SAM3 video model...")
            app.state.video_model = SAM3VideoModel(
                checkpoint=settings.sam3_checkpoint,
                bpe_path=settings.sam3_bpe_path,
                gpu_ids=settings.video_gpu_list,
                video_loader_type="cv2",
                async_loading_frames=False,
            )

            # Initialize session manager
            app.state.session_manager = SessionManager(
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

    logger.info("Server startup complete")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down server...")
    if app.state.image_model:
        app.state.image_model.clear_cache()
    if app.state.video_model:
        app.state.video_model.shutdown()
    if app.state.session_manager:
        app.state.session_manager.clear_all_sessions()
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="SAM3 Inference Server",
    description="FastAPI server for SAM3 (Segment Anything 3) image and video inference",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from api.routes import health, image, video

app.include_router(health.router, tags=["Health"])
app.include_router(image.router, prefix="/api/v1/image", tags=["Image"])
app.include_router(video.router, prefix="/api/v1/video", tags=["Video"])

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
        "server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.reload,
        workers=settings.server_workers if not settings.reload else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
