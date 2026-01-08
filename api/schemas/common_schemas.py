"""Common Pydantic schemas used across endpoints."""
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PromptType(str, Enum):
    """Type of prompt for segmentation."""

    TEXT = "text"
    POINT = "point"
    BOX = "box"


class TextPrompt(BaseModel):
    """Text-based prompt."""

    type: Literal[PromptType.TEXT] = PromptType.TEXT
    text: str = Field(..., description="Text description of object to segment")


class PointPrompt(BaseModel):
    """Point-based prompt with positive/negative clicks."""

    type: Literal[PromptType.POINT] = PromptType.POINT
    points: List[List[float]] = Field(
        ...,
        description="List of [x, y] coordinates normalized to [0, 1]",
        min_length=1,
    )
    point_labels: List[int] = Field(
        ...,
        description="Labels for each point: 1=positive, 0=negative",
        min_length=1,
    )


class BoxPrompt(BaseModel):
    """Bounding box prompt."""

    type: Literal[PromptType.BOX] = PromptType.BOX
    box: List[float] = Field(
        ...,
        description="Bounding box [cx, cy, w, h] normalized to [0, 1]",
        min_length=4,
        max_length=4,
    )
    label: bool = Field(
        default=True, description="True=positive exemplar, False=negative"
    )


# Union type for all prompts
Prompt = TextPrompt | PointPrompt | BoxPrompt


class SegmentationResult(BaseModel):
    """Result of segmentation operation."""

    masks: List[str] = Field(..., description="RLE-encoded masks")
    boxes: List[List[float]] = Field(
        ..., description="Bounding boxes in XYWH format, normalized [0,1]"
    )
    scores: List[float] = Field(..., description="Confidence scores for each mask")
    num_masks: int = Field(..., description="Total number of masks")


class ImageSize(BaseModel):
    """Image dimensions."""

    width: int
    height: int


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str
