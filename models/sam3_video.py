"""SAM3 video model wrapper for video segmentation and tracking."""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from sam3.logger import get_logger
from sam3.model.sam3_video_predictor import (
    Sam3VideoPredictor,
    Sam3VideoPredictorMultiGPU,
)
from sam3.model_builder import build_sam3_video_model
from sam3.utils.common_utils import rle_encode

logger = get_logger(__name__)


class SAM3VideoModel:
    """Wrapper for SAM3 video predictor with session management."""

    def __init__(
        self,
        checkpoint: str = "facebook/sam3",
        bpe_path: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        video_loader_type: str = "cv2",
        async_loading_frames: bool = False,
    ):
        """
        Initialize SAM3 video model.

        Args:
            checkpoint: HuggingFace checkpoint or local path
            bpe_path: Path to BPE tokenizer (optional)
            gpu_ids: List of GPU IDs for multi-GPU processing (None=single GPU)
            video_loader_type: "cv2" or "decord"
            async_loading_frames: Enable async frame loading
        """
        self.checkpoint = checkpoint
        self.bpe_path = bpe_path
        self.gpu_ids = gpu_ids or [torch.cuda.current_device()]
        self.video_loader_type = video_loader_type
        self.async_loading_frames = async_loading_frames

        # Initialize predictor
        if len(self.gpu_ids) > 1:
            logger.info(f"Initializing multi-GPU predictor with GPUs: {self.gpu_ids}")
            self.predictor = Sam3VideoPredictorMultiGPU(
                checkpoint_path=checkpoint,
                bpe_path=bpe_path,
                gpus_to_use=self.gpu_ids,
                video_loader_type=video_loader_type,
                async_loading_frames=async_loading_frames,
            )
        else:
            logger.info(f"Initializing single-GPU predictor on GPU: {self.gpu_ids[0]}")
            torch.cuda.set_device(self.gpu_ids[0])
            self.predictor = Sam3VideoPredictor(
                checkpoint_path=checkpoint,
                bpe_path=bpe_path,
                video_loader_type=video_loader_type,
                async_loading_frames=async_loading_frames,
            )

        logger.info("SAM3 video model initialized successfully")

    def start_session(
        self, video_path: str, session_id: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Start a new video inference session.

        Args:
            video_path: Path to video file or directory with frames
            session_id: Optional custom session ID

        Returns:
            Tuple of (session_id, video_info)
        """
        start_time = time.time()

        request = {
            "type": "start_session",
            "resource_path": video_path,
            "session_id": session_id,
        }

        response = self.predictor.handle_request(request)
        session_id = response["session_id"]

        # Get video info from inference state
        session = self.predictor._get_session(session_id)
        inference_state = session["state"]

        video_info = {
            "total_frames": inference_state["num_frames"],
            "resolution": {
                "width": inference_state["video_width"],
                "height": inference_state["video_height"],
            },
            "fps": 30.0,  # Default FPS (SAM3 doesn't store this)
            "duration_seconds": inference_state["num_frames"] / 30.0,
        }

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Started session {session_id} for video with {video_info['total_frames']} frames "
            f"({video_info['resolution']['width']}x{video_info['resolution']['height']}) "
            f"in {elapsed:.1f}ms"
        )

        return session_id, video_info

    def add_prompt(
        self,
        session_id: str,
        frame_index: int,
        text_prompt: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
    ) -> Tuple[int, int, List[str], List[List[float]], List[float]]:
        """
        Add prompt to a specific frame in the video session.

        Args:
            session_id: Session ID
            frame_index: Frame index to add prompt
            text_prompt: Text prompt (e.g., "person")
            points: List of points [[x, y], ...]
            point_labels: Point labels (1=foreground, 0=background)
            boxes: Bounding boxes in XYWH format [[cx, cy, w, h], ...]
            box_labels: Box labels (1=foreground, 0=background)
            obj_id: Object ID to refine (None=new object)

        Returns:
            Tuple of (frame_index, obj_id, masks, boxes, scores)
        """
        start_time = time.time()

        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": text_prompt,
            "points": points,
            "point_labels": point_labels,
            "bounding_boxes": boxes,
            "bounding_box_labels": box_labels,
            "obj_id": obj_id,
        }

        response = self.predictor.handle_request(request)
        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        # Extract results
        masks_rle = []
        boxes_xywh = []
        scores = []

        for obj_id, obj_data in outputs.items():
            # RLE encode masks
            mask_tensor = obj_data["mask"]  # (H, W) binary mask
            mask_rle = rle_encode(mask_tensor.cpu().numpy())
            masks_rle.append(mask_rle[0]["counts"])

            # Boxes in XYWH format
            box = obj_data["bbox"]  # [cx, cy, w, h]
            boxes_xywh.append(box.tolist() if torch.is_tensor(box) else box)

            # Scores
            score = obj_data.get("score", 1.0)
            scores.append(float(score) if torch.is_tensor(score) else score)

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Added prompt to frame {frame_idx} in session {session_id}: "
            f"{len(masks_rle)} objects, {elapsed:.1f}ms"
        )

        # Return the obj_id of the first (or specified) object
        obj_ids = list(outputs.keys())
        result_obj_id = obj_ids[0] if obj_ids else -1

        return frame_idx, result_obj_id, masks_rle, boxes_xywh, scores

    def propagate_in_video(
        self,
        session_id: str,
        direction: str = "both",
        start_frame_index: Optional[int] = None,
        max_frames: Optional[int] = None,
    ):
        """
        Propagate tracking through video frames (generator for streaming).

        Args:
            session_id: Session ID
            direction: "forward", "backward", or "both"
            start_frame_index: Starting frame index (None=start from 0)
            max_frames: Maximum frames to process (None=all frames)

        Yields:
            Dict with frame_index and objects list
        """
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": start_frame_index,
            "max_frame_num_to_track": max_frames,
        }

        start_time = time.time()
        frames_processed = 0

        for response in self.predictor.handle_stream_request(request):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            # Convert outputs to standard format
            objects = []
            for obj_id, obj_data in outputs.items():
                mask_tensor = obj_data["mask"]
                mask_rle = rle_encode(mask_tensor.cpu().numpy())

                box = obj_data["bbox"]
                score = obj_data.get("score", 1.0)

                objects.append(
                    {
                        "id": int(obj_id),
                        "mask": mask_rle[0]["counts"],
                        "box": box.tolist() if torch.is_tensor(box) else box,
                        "score": float(score) if torch.is_tensor(score) else score,
                    }
                )

            frames_processed += 1
            yield {"frame_index": frame_idx, "objects": objects}

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Propagation completed for session {session_id}: "
            f"{frames_processed} frames in {elapsed:.1f}ms "
            f"({elapsed/frames_processed:.1f}ms/frame)"
        )

    def remove_object(self, session_id: str, obj_id: int) -> bool:
        """
        Remove an object from tracking.

        Args:
            session_id: Session ID
            obj_id: Object ID to remove

        Returns:
            Success status
        """
        request = {
            "type": "remove_object",
            "session_id": session_id,
            "obj_id": obj_id,
            "is_user_action": True,
        }

        response = self.predictor.handle_request(request)
        logger.info(f"Removed object {obj_id} from session {session_id}")
        return response["is_success"]

    def reset_session(self, session_id: str) -> bool:
        """
        Reset session to initial state (clear all prompts/objects).

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        request = {"type": "reset_session", "session_id": session_id}

        response = self.predictor.handle_request(request)
        logger.info(f"Reset session {session_id}")
        return response["is_success"]

    def close_session(self, session_id: str) -> bool:
        """
        Close and cleanup session.

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        request = {"type": "close_session", "session_id": session_id}

        response = self.predictor.handle_request(request)
        logger.info(f"Closed session {session_id}")
        return response["is_success"]

    def get_session_info(self, session_id: str) -> Dict:
        """
        Get information about a session.

        Args:
            session_id: Session ID

        Returns:
            Session info dict
        """
        session = self.predictor._get_session(session_id)
        inference_state = session["state"]

        # Count current objects
        num_objects = len(inference_state.get("obj_ids", []))

        # Get GPU memory usage
        gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)

        return {
            "session_id": session_id,
            "num_frames": inference_state["num_frames"],
            "num_objects": num_objects,
            "gpu_memory_mb": gpu_memory_mb,
            "start_time": session["start_time"],
        }

    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self.predictor._ALL_INFERENCE_STATES.keys())

    def shutdown(self):
        """Shutdown predictor and cleanup all sessions."""
        self.predictor.shutdown()
        logger.info("SAM3 video model shutdown complete")
