"""Dependency management for FastAPI application."""
from typing import Generator, AsyncGenerator

from fastapi import Depends, HTTPException, Request
from loguru import logger

from app.models.sam3_image import SAM3ImageModel
from app.models.sam3_video import SAM3VideoModel
from app.services.session_manager import SessionManager
from app.services.image_service import ImageSegmentationService
from app.services.video_service import VideoSegmentationService


def get_image_model(request: Request) -> SAM3ImageModel:
    """Get image model from app state."""
    model = request.app.state.image_model
    if model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    return model


def get_video_model(request: Request) -> SAM3VideoModel:
    """Get video model from app state."""
    model = request.app.state.video_model
    if model is None:
        raise HTTPException(status_code=503, detail="Video model not loaded")
    return model


def get_session_manager(request: Request) -> SessionManager:
    """Get session manager from app state."""
    manager = request.app.state.session_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    return manager


def get_image_service(request: Request) -> ImageSegmentationService:
    """Get image segmentation service."""
    model = get_image_model(request)
    return ImageSegmentationService(model)


def get_video_service(request: Request) -> VideoSegmentationService:
    """Get video segmentation service."""
    model = get_video_model(request)
    manager = get_session_manager(request)
    return VideoSegmentationService(model, manager)