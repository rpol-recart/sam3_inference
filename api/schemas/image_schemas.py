"""Pydantic schemas for image inference endpoints."""
from typing import List, Optional

from pydantic import BaseModel, Field

from .common_schemas import ImageSize, Prompt, SegmentationResult


class ImageSegmentRequest(BaseModel):
    """Request for image segmentation."""

    image: str = Field(..., description="Base64-encoded image")
    prompts: List[Prompt] = Field(
        ..., description="List of prompts for segmentation", min_length=1
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold for filtering"
    )
    return_visualization: bool = Field(
        default=False, description="Return visualization image URL"
    )


class ImageSegmentResponse(BaseModel):
    """Response for image segmentation."""

    masks: List[str] = Field(..., description="RLE-encoded masks")
    boxes: List[List[float]] = Field(
        ..., description="Bounding boxes [cx, cy, w, h] normalized"
    )
    scores: List[float] = Field(..., description="Confidence scores")
    num_masks: int
    image_size: ImageSize
    visualization_url: Optional[str] = Field(
        None, description="URL to visualization if requested"
    )
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchImageItem(BaseModel):
    """Single image item for batch processing."""

    id: str = Field(..., description="Unique identifier for this image")
    image: str = Field(..., description="Base64-encoded image")
    prompts: List[Prompt] = Field(..., min_length=1)


class BatchImageRequest(BaseModel):
    """Request for batch image processing."""

    images: List[BatchImageItem] = Field(..., min_length=1)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_concurrent: int = Field(
        default=4, ge=1, le=16, description="Max concurrent processing"
    )


class BatchImageResultItem(BaseModel):
    """Result for single image in batch."""

    id: str
    masks: List[str]
    boxes: List[List[float]]
    scores: List[float]
    num_masks: int
    error: Optional[str] = None


class BatchImageResponse(BaseModel):
    """Response for batch image processing."""

    results: List[BatchImageResultItem]
    total_images: int
    successful: int
    failed: int
    total_time_ms: float


class CachedFeaturesRequest(BaseModel):
    """Request for cached features inference."""

    image: str = Field(..., description="Base64-encoded image")
    text_prompts: List[str] = Field(
        ..., min_length=1, description="List of text prompts to apply"
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CachedFeaturesResultItem(BaseModel):
    """Result for single prompt in cached features."""

    prompt: str
    masks: List[str]
    boxes: List[List[float]]
    scores: List[float]
    num_masks: int


class CachedFeaturesResponse(BaseModel):
    """Response for cached features inference."""

    results: List[CachedFeaturesResultItem]
    cache_hit: bool = Field(..., description="Whether features were cached")
    inference_time_ms: float
