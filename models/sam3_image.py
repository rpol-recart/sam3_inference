"""SAM3 Image Model Wrapper."""
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image

# Add parent sam3 directory to path
SAM3_ROOT = Path(__file__).parent.parent.parent.parent / "sam3"
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.train.masks_ops import rle_encode


class SAM3ImageModel:
    """Wrapper for SAM3 image inference."""

    def __init__(
        self,
        checkpoint: str = "facebook/sam3",
        bpe_path: Optional[str] = None,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        compile: bool = False,
    ):
        """Initialize SAM3 image model.

        Args:
            checkpoint: Model checkpoint path or HuggingFace ID
            bpe_path: Path to BPE tokenizer file
            device: Device to load model on
            confidence_threshold: Confidence threshold for filtering
            resolution: Input image resolution
            compile: Enable torch.compile optimization
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        logger.info(f"Loading SAM3 image model on {device}...")

        # Resolve BPE path
        if bpe_path is None:
            bpe_path = str(SAM3_ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")

        # Build model
        model = build_sam3_image_model(
            checkpoint_path=checkpoint if checkpoint != "facebook/sam3" else None,
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            load_from_HF=checkpoint == "facebook/sam3",
            compile=compile,
        )

        # Create processor
        self.processor = Sam3Processor(
            model=model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        # Feature cache for multiple prompts on same image
        self.feature_cache: Dict[str, Dict] = {}

        logger.info("SAM3 image model loaded successfully")

    def segment_text(
        self, image: Image.Image, text_prompt: str
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """Segment image with text prompt.

        Args:
            image: PIL Image
            text_prompt: Text description of object to segment

        Returns:
            Tuple of (masks, boxes, scores)
            - masks: List of RLE-encoded mask strings
            - boxes: List of [cx, cy, w, h] normalized boxes
            - scores: List of confidence scores
        """
        start_time = time.time()

        # Set image and get features
        state = self.processor.set_image(image)

        # Add text prompt
        state = self.processor.set_text_prompt(state, prompt=text_prompt)

        # Extract results
        masks, boxes, scores = self._extract_results(state, image.size)

        logger.debug(
            f"Text segmentation completed in {(time.time() - start_time)*1000:.1f}ms"
        )

        return masks, boxes, scores

    def segment_box(
        self, image: Image.Image, box: List[float], label: bool = True
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """Segment image with bounding box prompt.

        Args:
            image: PIL Image
            box: Bounding box [cx, cy, w, h] normalized [0, 1]
            label: True for positive exemplar, False for negative

        Returns:
            Tuple of (masks, boxes, scores)
        """
        start_time = time.time()

        # Set image
        state = self.processor.set_image(image)

        # Add box prompt
        # Convert normalized box to pixel coordinates
        orig_w, orig_h = image.size
        box_pixels = [
            box[0] * orig_w,  # cx
            box[1] * orig_h,  # cy
            box[2] * orig_w,  # w
            box[3] * orig_h,  # h
        ]

        state = self.processor.add_geometric_prompt(
            state=state, bounding_box=box_pixels, bounding_box_label=label
        )

        # Extract results
        masks, boxes, scores = self._extract_results(state, image.size)

        logger.debug(
            f"Box segmentation completed in {(time.time() - start_time)*1000:.1f}ms"
        )

        return masks, boxes, scores

    def segment_combined(
        self,
        image: Image.Image,
        text_prompts: Optional[List[str]] = None,
        boxes: Optional[List[Tuple[List[float], bool]]] = None,
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """Segment with combined prompts.

        Args:
            image: PIL Image
            text_prompts: List of text prompts
            boxes: List of (box, label) tuples

        Returns:
            Tuple of (masks, boxes, scores)
        """
        state = self.processor.set_image(image)

        # Add text prompts
        if text_prompts:
            for text in text_prompts:
                state = self.processor.set_text_prompt(state, prompt=text)

        # Add box prompts
        if boxes:
            orig_w, orig_h = image.size
            for box, label in boxes:
                box_pixels = [
                    box[0] * orig_w,
                    box[1] * orig_h,
                    box[2] * orig_w,
                    box[3] * orig_h,
                ]
                state = self.processor.add_geometric_prompt(
                    state=state, bounding_box=box_pixels, bounding_box_label=label
                )

        return self._extract_results(state, image.size)

    def cache_features(self, image: Image.Image, cache_key: str) -> str:
        """Cache image features for reuse with multiple prompts.

        Args:
            image: PIL Image
            cache_key: Unique key for this image

        Returns:
            cache_key
        """
        state = self.processor.set_image(image)
        self.feature_cache[cache_key] = {
            "backbone_out": state["backbone_out"],
            "image_size": image.size,
        }
        return cache_key

    def segment_with_cached_features(
        self, cache_key: str, text_prompts: List[str]
    ) -> List[Tuple[List[str], List[List[float]], List[float]]]:
        """Segment using cached features with multiple text prompts.

        Args:
            cache_key: Key for cached features
            text_prompts: List of text prompts to apply

        Returns:
            List of (masks, boxes, scores) tuples for each prompt
        """
        if cache_key not in self.feature_cache:
            raise ValueError(f"No cached features found for key: {cache_key}")

        cached = self.feature_cache[cache_key]
        results = []

        for prompt in text_prompts:
            # Create state with cached backbone
            state = {"backbone_out": cached["backbone_out"]}

            # Add text prompt
            state = self.processor.set_text_prompt(state, prompt=prompt)

            # Extract results
            masks, boxes, scores = self._extract_results(state, cached["image_size"])
            results.append((masks, boxes, scores))

        return results

    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear feature cache.

        Args:
            cache_key: Specific key to clear, or None to clear all
        """
        if cache_key:
            self.feature_cache.pop(cache_key, None)
        else:
            self.feature_cache.clear()

    def _extract_results(
        self, state: Dict, image_size: Tuple[int, int]
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """Extract and format results from inference state.

        Args:
            state: Inference state dict
            image_size: Original image size (width, height)

        Returns:
            Tuple of (masks, boxes, scores)
        """
        orig_w, orig_h = image_size

        # Normalize boxes to [0, 1]
        boxes_xyxy = torch.stack(
            [
                state["boxes"][:, 0] / orig_w,
                state["boxes"][:, 1] / orig_h,
                state["boxes"][:, 2] / orig_w,
                state["boxes"][:, 3] / orig_h,
            ],
            dim=-1,
        )

        # Convert to XYWH format
        boxes_xywh = box_xyxy_to_xywh(boxes_xyxy).tolist()

        # RLE encode masks
        masks_rle = rle_encode(state["masks"].squeeze(1))
        masks = [m["counts"] for m in masks_rle]

        # Scores
        scores = state["scores"].tolist()

        # Filter by confidence threshold
        filtered_results = []
        for mask, box, score in zip(masks, boxes_xywh, scores):
            if score >= self.confidence_threshold:
                filtered_results.append((mask, box, score))

        if not filtered_results:
            return [], [], []

        masks_filtered, boxes_filtered, scores_filtered = zip(*filtered_results)

        return list(masks_filtered), list(boxes_filtered), list(scores_filtered)

    @property
    def model_info(self) -> Dict:
        """Get model information."""
        return {
            "device": self.device,
            "resolution": self.resolution,
            "confidence_threshold": self.confidence_threshold,
            "cache_size": len(self.feature_cache),
        }
