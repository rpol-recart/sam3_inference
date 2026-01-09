"""Image segmentation service for handling business logic."""
import base64
import hashlib
import time
from io import BytesIO
from typing import List, Tuple, Optional

from loguru import logger
from PIL import Image

from app.models.sam3_image import SAM3ImageModel
from app.schemas.common_schemas import PromptType
from app.schemas.image_schemas import (
    ImageSegmentRequest,
    ImageSegmentResponse,
    CachedFeaturesRequest,
    CachedFeaturesResponse,
    CachedFeaturesResultItem,
)
from app.exceptions import InvalidImageFormatError


class ImageSegmentationService:
    """Service for image segmentation operations."""

    def __init__(self, model: SAM3ImageModel) -> None:
        """Initialize the service with the model instance.
        
        Args:
            model: The SAM3 image model instance to use for segmentation
        """
        self.model = model

    def segment_image(self, request: ImageSegmentRequest) -> ImageSegmentResponse:
        """Perform image segmentation based on the request.
        
        Args:
            request: The segmentation request containing image and prompts
            
        Returns:
            The segmentation results including masks, boxes, and scores
            
        Raises:
            InvalidImageFormatError: If the image format is invalid
            ValueError: If no prompts are provided
        """
        start_time = time.time()

        try:
            # Decode image
            image = self.decode_base64_image(request.image)
            logger.info(f"Processing image of size {image.size}")

            # Extract prompts by type
            text_prompts: List[str] = []
            box_prompts: List[Tuple[List[float], bool]] = []
            point_prompts: List[Tuple[List[List[float]], List[bool]]] = []

            for prompt in request.prompts:
                if prompt.type == PromptType.TEXT:
                    text_prompts.append(prompt.text)
                elif prompt.type == PromptType.BOX:
                    box_prompts.append((prompt.box, prompt.label))
                elif prompt.type == PromptType.POINT:
                    point_prompts.append((prompt.points, prompt.point_labels))

            # Segment with combined prompts
            if text_prompts or box_prompts or point_prompts:
                masks, boxes, scores = self.model.segment_combined(
                    image=image,
                    text_prompts=text_prompts if text_prompts else None,
                    boxes=box_prompts if box_prompts else None,
                    points=point_prompts if point_prompts else None,
                )
            else:
                raise ValueError("At least one text, box, or point prompt is required")

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

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise

    def segment_with_cached_features(self, request: CachedFeaturesRequest) -> CachedFeaturesResponse:
        """Perform segmentation with cached features.
        
        Args:
            request: The cached features request containing image and text prompts
            
        Returns:
            The segmentation results for each text prompt
        """
        start_time = time.time()

        try:
            # Decode image
            image = self.decode_base64_image(request.image)

            # Generate cache key from image hash
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            cache_key = hashlib.md5(image_bytes.getvalue()).hexdigest()

            # Cache features if not already cached
            cache_hit = cache_key in self.model.feature_cache
            if not cache_hit:
                self.model.cache_features(image, cache_key)

            # Segment with all text prompts
            results_list = self.model.segment_with_cached_features(
                cache_key, request.text_prompts
            )

            # Format results
            results: List[CachedFeaturesResultItem] = []
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

        except Exception as e:
            logger.error(f"Cached features segmentation failed: {e}")
            raise

    @staticmethod
    def decode_base64_image(base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image.
        
        Args:
            base64_str: The base64 encoded image string
            
        Returns:
            The decoded PIL Image object
            
        Raises:
            InvalidImageFormatError: If the base64 string is invalid or not an image
        """
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return image.convert("RGB")
        except Exception as e:
            raise InvalidImageFormatError(f"Invalid image data: {str(e)}")