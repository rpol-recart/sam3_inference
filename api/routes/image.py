"""Image inference endpoints."""
import base64
import hashlib
import time
from io import BytesIO

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from PIL import Image

from api.schemas.common_schemas import PromptType
from api.schemas.image_schemas import (
    CachedFeaturesRequest,
    CachedFeaturesResponse,
    CachedFeaturesResultItem,
    ImageSegmentRequest,
    ImageSegmentResponse,
)

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
async def segment_image(request: ImageSegmentRequest, req: Request):
    """Segment image with prompts.

    Supports text prompts, box prompts, and combinations.
    """
    # Removed import server, using req.app.state instead

    if req.app.state.image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image)
        logger.info(f"Processing image of size {image.size}")

        # Extract prompts by type
        text_prompts = []
        box_prompts = []
        point_prompts = []

        for prompt in request.prompts:
            if prompt.type == PromptType.TEXT:
                text_prompts.append(prompt.text)
            elif prompt.type == PromptType.BOX:
                box_prompts.append((prompt.box, prompt.label))
            elif prompt.type == PromptType.POINT:
                point_prompts.append((prompt.points, prompt.point_labels))

        # Segment with combined prompts
        if text_prompts or box_prompts or point_prompts:
            masks, boxes, scores = req.app.state.image_model.segment_combined(
                image=image,
                text_prompts=text_prompts if text_prompts else None,
                boxes=box_prompts if box_prompts else None,
                points=point_prompts if point_prompts else None,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="At least one text, box, or point prompt is required",
            )

        inference_time = (time.time() - start_time) * 1000

        logger.info(
            f"Segmentation complete: {len(masks)} masks in {inference_time:.1f}ms"
        )

        return ImageSegmentResponse(
            masks=masks,
            boxes=boxes,
            scores=scores,
            num_masks=len(masks),
            image_size={"width": image.size[0], "height": image.size[1]},
            visualization_url=None,  # TODO: implement visualization
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cached-features", response_model=CachedFeaturesResponse)
async def segment_with_cached_features(request: CachedFeaturesRequest, req: Request):
    """Segment image with multiple text prompts using feature caching.

    This endpoint caches the image features and applies multiple text prompts,
    which is ~10x faster than making separate requests for each prompt.
    """
    # Removed import server, using req.app.state instead

    if req.app.state.image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image)

        # Generate cache key from image hash
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        cache_key = hashlib.md5(image_bytes.getvalue()).hexdigest()

        # Cache features if not already cached
        cache_hit = cache_key in req.app.state.image_model.feature_cache
        if not cache_hit:
            req.app.state.image_model.cache_features(image, cache_key)

        # Segment with all text prompts
        results_list = req.app.state.image_model.segment_with_cached_features(
            cache_key, request.text_prompts
        )

        # Format results
        results = []
        for prompt, (masks, boxes, scores) in zip(request.text_prompts, results_list):
            results.append(
                CachedFeaturesResultItem(
                    prompt=prompt,
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    num_masks=len(masks),
                )
            )

        inference_time = (time.time() - start_time) * 1000

        logger.info(
            f"Cached features segmentation: {len(request.text_prompts)} prompts "
            f"in {inference_time:.1f}ms (cache_hit={cache_hit})"
        )

        return CachedFeaturesResponse(
            results=results,
            cache_hit=cache_hit,
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cached features segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
