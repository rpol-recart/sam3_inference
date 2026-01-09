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
        checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        compile: bool = False,
    ):
        """Initialize SAM3 image model.

        Args:
            checkpoint: Model checkpoint path or HuggingFace ID. If None, defaults to local path or HF download.
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

        # Determine if we should load from HuggingFace or local path
        load_from_HF = False
        resolved_checkpoint_path = None
        
        # Check if checkpoint is a HuggingFace ID by looking for common patterns
        # HuggingFace IDs typically contain "/" and don't look like file paths
        if checkpoint is None:
            # Default to local checkpoint if available, otherwise try HF
            local_checkpoint = "/app/server/sam_weights/sam3.pt"
            if Path(local_checkpoint).exists():
                resolved_checkpoint_path = local_checkpoint
            else:
                load_from_HF = True
        elif "/" in checkpoint and not checkpoint.startswith("/") and not Path(checkpoint).is_file():
            # Likely a HuggingFace ID (e.g., "facebook/sam3")
            if checkpoint == "facebook/sam3":
                load_from_HF = True
            else:
                # Custom HuggingFace repo - this case should be handled by the download function
                resolved_checkpoint_path = checkpoint
        else:
            # Local path - check if it's a directory or file
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_dir():
                # If it's a directory, look for sam3.pt inside
                checkpoint_file = checkpoint_path / "sam3.pt"
                if checkpoint_file.exists():
                    resolved_checkpoint_path = str(checkpoint_file)
                else:
                    raise FileNotFoundError(f"Checkpoint file 'sam3.pt' not found in directory: {checkpoint}")
            elif checkpoint_path.is_file():
                # If it's already a file, use it directly
                resolved_checkpoint_path = checkpoint
            else:
                # Path doesn't exist, check if it looks like a HF ID
                load_from_HF = True

        logger.info(f"Using checkpoint path: {resolved_checkpoint_path}, load_from_HF: {load_from_HF}")

        # Build model
        model = build_sam3_image_model(
            checkpoint_path=resolved_checkpoint_path,
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            load_from_HF=load_from_HF,
            compile=compile,
        )
        model = model.to(device)

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
        state = self.processor.set_text_prompt(prompt=text_prompt, state=state)

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
            box=box_pixels, label=label, state=state
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
        points: Optional[List[Tuple[List[List[float]], List[bool]]]] = None,
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """Segment with combined prompts.

        Args:
            image: PIL Image
            text_prompts: List of text prompts
            boxes: List of (box, label) tuples
            points: List of (points, labels) tuples where points is a list of [x, y] coordinates

        Returns:
            Tuple of (masks, boxes, scores)
        """
        state = self.processor.set_image(image)

        # Add text prompts
        if text_prompts:
            for text in text_prompts:
                state = self.processor.set_text_prompt(prompt=text, state=state)

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
                    box=box_pixels, label=label, state=state
                )

        # Add point prompts - if supported by the processor
        if points:
            orig_w, orig_h = image.size
            for point_list, point_labels in points:
                # Convert normalized points to pixel coordinates
                points_pixels = torch.tensor([[x * orig_w, y * orig_h] for x, y in point_list],
                                           device=self.device, dtype=torch.float32).view(-1, 1, 2)
                
                # Convert point labels to tensor (assuming 1 for positive, 0 for negative)
                point_tensor_labels = torch.tensor([1 if label else 0 for label in point_labels],
                                                  device=self.device, dtype=torch.long).view(-1, 1)
                
                # Check if language features exist, if not, initialize with "visual" prompt
                if "language_features" not in state["backbone_out"]:
                    # Add a visual text prompt to the backbone output to allow geometric-only prompting
                    dummy_text_outputs = self.processor.model.backbone.forward_text(["visual"], device=self.device)
                    state["backbone_out"].update(dummy_text_outputs)
                
                # Initialize geometric prompt if not present
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
                
                # Use the append_points method of the geometric prompt
                state["geometric_prompt"].append_points(
                    points=points_pixels,
                    labels=point_tensor_labels
                )
                
                # Run the grounding with the updated prompt
                state = self.processor._forward_grounding(state)

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
            # Create state with cached backbone and image information
            orig_w, orig_h = cached["image_size"]
            state = {
                "backbone_out": cached["backbone_out"],
                "original_height": orig_h,
                "original_width": orig_w,
            }

            # Add text prompt
            state = self.processor.set_text_prompt(prompt=prompt, state=state)

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
        ).to(self.device)

        # Convert to XYWH format
        boxes_xywh = box_xyxy_to_xywh(boxes_xyxy).to(self.device).tolist()

        # RLE encode masks
        masks_rle = rle_encode(state["masks"].squeeze(1).to(self.device))
        masks = [m["counts"] for m in masks_rle]

        # Scores
        scores = state["scores"].to(self.device).tolist()

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
