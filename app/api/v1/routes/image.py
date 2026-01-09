"""Image inference endpoints."""
import base64
import time
from io import BytesIO
from typing import List
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger
from PIL import Image

from app.schemas.image_schemas import (
    ImageSegmentRequest,
    ImageSegmentResponse,
    CachedFeaturesRequest,
    CachedFeaturesResponse,
    CachedFeaturesResultItem,
)
from app.schemas.common_schemas import PromptType
from app.services.image_service import ImageSegmentationService
from app.api.dependencies import get_image_service
from app.exceptions import ModelNotLoadedError

router = APIRouter()


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@router.post("/segment", response_model=ImageSegmentResponse)
async def segment_image(
    request: ImageSegmentRequest,
    image_service: ImageSegmentationService = Depends(get_image_service)
):
    """Segment image with prompts.

    Supports text prompts, box prompts, and combinations.
    """
    try:
        return image_service.segment_image(request)
    except InvalidImageFormatError as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cached-features", response_model=CachedFeaturesResponse)
async def segment_with_cached_features(
    request: CachedFeaturesRequest,
    image_service: ImageSegmentationService = Depends(get_image_service)
):
    """Segment image with multiple text prompts using feature caching.

    This endpoint caches the image features and applies multiple text prompts,
    which is ~10x faster than making separate requests for each prompt.
    """
    try:
        return image_service.segment_with_cached_features(request)
    except InvalidImageFormatError as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except Exception as e:
        logger.error(f"Cached features segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
image_router = router