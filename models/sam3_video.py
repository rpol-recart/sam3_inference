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
from sam3.agent.helpers.rle import rle_encode

logger = get_logger(__name__)


class SAM3VideoModel:
    """Wrapper for SAM3 video predictor with session management."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        video_loader_type: str = "cv2",
        async_loading_frames: bool = False,
     ):
        """
        Initialize SAM3 video model.

        Args:
            checkpoint: HuggingFace checkpoint or local path. If None, defaults to local path or HF download.
            bpe_path: Path to BPE tokenizer (optional)
            gpu_ids: List of GPU IDs for multi-GPU processing (None=single GPU)
            video_loader_type: "cv2" or "decord"
            async_loading_frames: Enable async frame loading
        """
        
        # Determine if we should load from HuggingFace or local path
        load_from_HF = False
        resolved_checkpoint_path = None
        
        # If no checkpoint is specified, use default local path
        if checkpoint is None:
            local_checkpoint = "/app/server/sam_weights/sam3.pt"
            if Path(local_checkpoint).exists():
                resolved_checkpoint_path = local_checkpoint
            else:
                load_from_HF = True
        else:
            # A checkpoint was specified, check if it's a local file or directory
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_file():
                # Direct file path
                resolved_checkpoint_path = checkpoint
            elif checkpoint_path.is_dir():
                # Directory, look for sam3.pt inside
                checkpoint_file = checkpoint_path / "sam3.pt"
                if checkpoint_file.exists():
                    resolved_checkpoint_path = str(checkpoint_file)
                else:
                    raise FileNotFoundError(f"Checkpoint file 'sam3.pt' not found in directory: {checkpoint}")
            else:
                # Path doesn't exist - might be a HuggingFace ID or invalid path
                # Check if it looks like a HuggingFace ID (contains '/' but not a local path)
                if "/" in checkpoint and not checkpoint.startswith(("/", ".")):
                    # Likely a HuggingFace ID (e.g., "facebook/sam3")
                    # First try to see if local path exists as fallback
                    local_checkpoint = "/app/server/sam_weights/sam3.pt"
                    if Path(local_checkpoint).exists():
                        logger.info(f"Using local checkpoint instead of HuggingFace: {local_checkpoint}")
                        resolved_checkpoint_path = local_checkpoint
                    else:
                        load_from_HF = True
                else:
                    # Invalid local path, fallback to default
                    local_checkpoint = "/app/server/sam_weights/sam3.pt"
                    if Path(local_checkpoint).exists():
                        resolved_checkpoint_path = local_checkpoint
                    else:
                        load_from_HF = True

        logger.info(f"Using video checkpoint path: {resolved_checkpoint_path}, load_from_HF: {load_from_HF}")

        self.checkpoint = resolved_checkpoint_path
        self.bpe_path = bpe_path
        self.gpu_ids = gpu_ids or [0]  # Default to GPU 0 if none specified
        self.video_loader_type = video_loader_type
        self.async_loading_frames = async_loading_frames

        # Ensure we don't exceed available GPUs
        available_gpus = torch.cuda.device_count()
        filtered_gpu_ids = [gpu_id for gpu_id in self.gpu_ids if gpu_id < available_gpus]
        if not filtered_gpu_ids:
            filtered_gpu_ids = [0]  # Fallback to GPU 0 if no valid GPU IDs
        
        self.gpu_ids = filtered_gpu_ids

        # Initialize predictor
        if len(self.gpu_ids) > 1:
            logger.info(f"Initializing multi-GPU predictor with GPUs: {self.gpu_ids}")
            self.predictor = Sam3VideoPredictorMultiGPU(
                checkpoint_path=resolved_checkpoint_path,
                bpe_path=bpe_path,
                gpus_to_use=self.gpu_ids,
                video_loader_type=video_loader_type,
                async_loading_frames=async_loading_frames,
            )
        else:
            logger.info(f"Initializing single-GPU predictor on GPU: {self.gpu_ids[0]}")
            torch.cuda.set_device(self.gpu_ids[0])
            self.predictor = Sam3VideoPredictor(
                checkpoint_path=resolved_checkpoint_path,
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

        # FIX: Convert text_ids from list to tensor to fix numpy indexing error
        # SAM3 library initializes text_ids as Python lists, but later tries to use
        # tensor ellipsis indexing [...] which only works on tensors
        try:
            num_frames = inference_state.get("num_frames", 0)
            input_batch = inference_state.get("input_batch")

            if input_batch is not None and hasattr(input_batch, "find_inputs"):
                for t in range(num_frames):
                    if t < len(input_batch.find_inputs):
                        find_input = input_batch.find_inputs[t]
                        if hasattr(find_input, "text_ids"):
                            text_ids = find_input.text_ids
                            # Convert list to tensor if needed
                            if isinstance(text_ids, list):
                                find_input.text_ids = torch.tensor(
                                    text_ids, dtype=torch.long, device=self.predictor.device
                                )
                                logger.debug(f"Converted text_ids to tensor for frame {t}")
        except Exception as e:
            logger.warning(f"Could not patch text_ids in inference_state: {e}")
            # Continue anyway, the error might not occur

        # Extract video properties from the inference state
        # Different versions of SAM3 may have different attribute names
        width = 0
        height = 0
        
        # Look for video dimensions in various possible locations in the inference state
        if hasattr(inference_state, 'get'):
            width = (inference_state.get("video_width") or
                    inference_state.get("width") or
                    inference_state.get("_video_width", 0))
            height = (inference_state.get("video_height") or
                     inference_state.get("height") or
                     inference_state.get("_video_height", 0))
        
        # If dimensions are still 0, get them from the video file itself
        if width == 0 or height == 0:
            import cv2
            # Try to get the video path from the session or use the original video_path parameter
            session_video_path = session.get("video_path", "")
            if session_video_path and session_video_path != "":
                cap = cv2.VideoCapture(session_video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
            else:
                # If we still don't have the video path in session, use the original video_path parameter
                cap = cv2.VideoCapture(video_path)  # video_path is a parameter to this method
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

        video_info = {
            "total_frames": inference_state.get("num_frames", 0),
            "resolution": {
                "width": width,
                "height": height,
            },
            "fps": 30.0,  # Default FPS (SAM3 doesn't store this)
            "duration_seconds": inference_state.get("num_frames", 0) / 30.0,
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
    ) -> Tuple[int, List[int], List[str], List[List[float]], List[float]]:
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
        if not isinstance(frame_index, int):
            logger.warning(f"frame_index is not int: {type(frame_index)} = {frame_index}")
            frame_index = int(frame_index) if hasattr(frame_index, '__int__') else frame_index[0]
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
        logger.info(request)
        response = self.predictor.handle_request(request)
        logger.info('Handle req sucsesfull')
        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        masks_rle = []
        boxes_xywh = []
        scores = []

        obj_ids = outputs["out_obj_ids"]
        probs = outputs["out_probs"]
        boxes = outputs["out_boxes_xywh"]
        masks = outputs["out_binary_masks"]  # numpy array of shape (N, H, W), dtype=bool

        for i in range(len(obj_ids)):
            # 1. Преобразуем numpy → torch, добавляем batch dim: (H, W) → (1, H, W)
            mask_tensor = torch.from_numpy(masks[i]).unsqueeze(0)  # (1, H, W), dtype=torch.bool

            # 2. RLE encode
            mask_rle = rle_encode(mask_tensor)  # возвращает список из 1 элемента
            masks_rle.append(mask_rle[0]["counts"])

            # 3. Box
            boxes_xywh.append(boxes[i].tolist())

            # 4. Score
            scores.append(float(probs[i]))
        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Added prompt to frame {frame_idx} in session {session_id}: "
            f"{len(masks_rle)} objects, {elapsed:.1f}ms"
        )

        # Return the obj_id of the first (or specified) object
        #obj_ids = list(outputs.keys())
        #result_obj_id = obj_ids[0] if obj_ids else -1

        return frame_idx, list(obj_ids), masks_rle, boxes_xywh, scores

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
        logger.info(f'Input {request}')
        start_time = time.time()
        frames_processed = 0

        for response in self.predictor.handle_stream_request(request):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]
            
            # Extract arrays
            obj_ids = outputs["out_obj_ids"]          # shape: (N,)
            probs = outputs["out_probs"]              # shape: (N,)
            boxes_xywh = outputs["out_boxes_xywh"]    # shape: (N, 4)
            binary_masks = outputs["out_binary_masks"]  # shape: (N, H, W)

            # Convert outputs to standard format
            objects = []
            for i in range(len(obj_ids)):
                obj_id = obj_ids[i]
                mask_np = binary_masks[i]  # This is a NumPy array of dtype=bool
                #mask_tensor = torch.from_numpy(masks[i]).unsqueeze(0)  # (1, H, W), dtype=torch.bool
                # Convert to torch tensor with correct dtype for rle_encode
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).bool()  # Ensure torch.bool

                # Encode RLE
                mask_rle = rle_encode(mask_tensor)  # pycocotools expects NumPy bool array

                # Convert box from XYWH to XYXY if needed? (not required if your system uses XYWH)
                box = boxes_xywh[i].tolist()
                score = float(probs[i])

                objects.append({
                    "id": int(obj_id),
                    "mask": mask_rle[0]["counts"],  # assuming rle_encode returns list of dicts
                    "box": box,
                    "score": score,
                })

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
