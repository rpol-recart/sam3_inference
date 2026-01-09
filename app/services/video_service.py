"""Video segmentation service for handling business logic."""
import base64
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger

from app.models.sam3_video import SAM3VideoModel
from app.services.session_manager import SessionManager
from app.schemas.video_schemas import (
    StartSessionRequest,
    StartSessionResponse,
    AddPromptRequest,
    AddPromptResponse,
    PropagateRequest,
    PropagateResponse,
    VideoSessionStatus,
    FrameResult,
    SessionStatusResponse,
    RemoveObjectResponse,
    ResetSessionResponse,
    CloseSessionResponse,
    SessionListResponse,
)
from app.exceptions import SessionNotFoundError, InvalidVideoSourceError


class VideoSegmentationService:
    """Service for video segmentation operations."""

    def __init__(self, model: SAM3VideoModel, session_manager: SessionManager):
        """Initialize the service with the model and session manager instances.
        
        Args:
            model: The SAM3 video model instance to use for segmentation
            session_manager: The session manager for managing video sessions
        """
        self.model = model
        self.session_manager = session_manager

    def save_video_from_request(self, request: StartSessionRequest) -> str:
        """
        Save video from request to temporary file.

        Args:
            request: StartSessionRequest with video source

        Returns:
            Path to saved video file

        Raises:
            InvalidVideoSourceError: If no valid video source provided
        """
        if request.video_path and os.path.exists(request.video_path):
            return request.video_path

        if request.video_url:
            # Download video from URL
            import urllib.request
            
            try:
                # Create temporary file to store downloaded video
                temp_dir = Path(tempfile.gettempdir()) / "sam3_videos"
                temp_dir.mkdir(exist_ok=True)
                
                # Generate a unique filename based on URL
                url_hash = hashlib.md5(request.video_url.encode()).hexdigest()
                parsed_url = urlparse(request.video_url)
                ext = os.path.splitext(parsed_url.path)[1] or ".mp4"
                temp_video_path = temp_dir / f"downloaded_{url_hash}{ext}"
                
                # Download the video file
                urllib.request.urlretrieve(request.video_url, str(temp_video_path))
                
                # Verify that the file was downloaded
                if not temp_video_path.exists():
                    raise InvalidVideoSourceError()
                    
                return str(temp_video_path)
            except Exception as e:
                raise InvalidVideoSourceError()

        if request.video_base64:
            # Decode base64 and save to temp file
            try:
                video_bytes = base64.b64decode(request.video_base64)
                temp_dir = Path(tempfile.gettempdir()) / "sam3_videos"
                temp_dir.mkdir(exist_ok=True)

                # Generate filename from hash
                video_hash = hashlib.md5(video_bytes).hexdigest()
                video_path = temp_dir / f"{video_hash}.mp4"

                with open(video_path, "wb") as f:
                    f.write(video_bytes)

                return str(video_path)
            except Exception as e:
                raise InvalidVideoSourceError()

        raise InvalidVideoSourceError()

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        """Start a new video inference session.
        
        Args:
            request: The start session request containing video source and session ID
            
        Returns:
            The session response with session ID, video info, and status
        """
        # Save video to file
        video_path = self.save_video_from_request(request)

        # Start session with SAM3
        session_id, video_info = self.model.start_session(
            video_path=video_path, session_id=request.session_id
        )

        # Register session in manager
        self.session_manager.create_session(
            session_id=session_id, session_type="video", video_info=video_info
        )

        return StartSessionResponse(
            session_id=session_id,
            video_info=video_info,
            status=VideoSessionStatus.READY,
        )

    def add_prompt_to_frame(self, session_id: str, request: AddPromptRequest) -> AddPromptResponse:
        """Add prompts to a specific frame in the video.
        
        Args:
            session_id: The session ID
            request: The add prompt request containing frame index and prompts
            
        Returns:
            The response with frame index, object IDs, masks, boxes, and scores
        """
        # Check session exists
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        try:
            self.session_manager.update_session_status(
                session_id, VideoSessionStatus.PROCESSING
            )

            # Extract prompts
            text_prompts = [p.text for p in request.prompts if hasattr(p, "text")]
            text_prompt = text_prompts[0] if text_prompts else None

            box_prompts = [p for p in request.prompts if hasattr(p, "box")]
            boxes = [p.box for p in box_prompts] if box_prompts else None
            box_labels = [p.label for p in box_prompts] if box_prompts else None

            point_prompts = [p for p in request.prompts if hasattr(p, "point")]
            points = [p.point for p in point_prompts] if point_prompts else None
            point_labels = [p.label for p in point_prompts] if point_prompts else None

            # Add prompt to SAM3
            logger.info(f"Adding prompt to frame {request.frame_index}...")
            frame_idx, obj_ids, masks, boxes_out, scores = self.model.add_prompt(
                session_id=session_id,
                frame_index=request.frame_index,
                text_prompt=text_prompt,
                points=points,
                point_labels=point_labels,
                boxes=boxes,
                box_labels=box_labels,
                obj_id=request.obj_id,
            )

            # Update session stats
            session_info = self.model.get_session_info(session_id)
            self.session_manager.update_session_stats(
                session_id, objects_count=session_info["num_objects"]
            )
            self.session_manager.update_session_status(
                session_id, VideoSessionStatus.READY
            )

            return AddPromptResponse(
                frame_index=frame_idx,
                obj_id=obj_ids,
                masks=masks,
                boxes=boxes_out,
                scores=scores,
            )

        except Exception as e:
            self.session_manager.update_session_status(
                session_id, VideoSessionStatus.ERROR, error=str(e)
            )
            raise

    def propagate_tracking(self, session_id: str, request: PropagateRequest) -> PropagateResponse:
        """Propagate object tracking through video frames.
        
        Args:
            session_id: The session ID
            request: The propagate request with direction and frame range
            
        Returns:
            The response with tracking results for each frame
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        try:
            self.session_manager.update_session_status(
                session_id, VideoSessionStatus.PROCESSING
            )

            start_time = time.time()
            results: Dict[int, FrameResult] = {}

            # Propagate through video
            for frame_data in self.model.propagate_in_video(
                session_id=session_id,
                direction=request.direction,
                start_frame_index=request.start_frame_index,
                max_frames=request.max_frames,
            ):
                frame_idx = frame_data["frame_index"]
                objects = frame_data["objects"]

                results[frame_idx] = FrameResult(frame_index=frame_idx, objects=objects)

            elapsed_ms = (time.time() - start_time) * 1000

            # Update session stats
            self.session_manager.update_session_stats(
                session_id, frames_processed=len(results)
            )
            self.session_manager.update_session_status(
                session_id, VideoSessionStatus.READY
            )

            return PropagateResponse(
                session_id=session_id,
                results=results,
                total_frames=len(results),
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            self.session_manager.update_session_status(
                session_id, VideoSessionStatus.ERROR, error=str(e)
            )
            raise

    def get_session_status(self, session_id: str) -> SessionStatusResponse:
        """Get current status of a video session.
        
        Args:
            session_id: The session ID
            
        Returns:
            The session status response
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        session_info = self.model.get_session_info(session_id)

        return SessionStatusResponse(
            session_id=session_id,
            status=session["status"],
            current_objects=session_info["num_objects"],
            frames_processed=session["frames_processed"],
            total_frames=session_info["num_frames"],
            gpu_memory_used_mb=session_info["gpu_memory_mb"],
        )

    def remove_object_from_tracking(self, session_id: str, obj_id: int) -> RemoveObjectResponse:
        """Remove an object from tracking in the session.
        
        Args:
            session_id: The session ID
            obj_id: The object ID to remove
            
        Returns:
            The remove response
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        self.model.remove_object(session_id, obj_id)

        # Update stats
        session_info = self.model.get_session_info(session_id)
        self.session_manager.update_session_stats(
            session_id, objects_count=session_info["num_objects"]
        )

        return RemoveObjectResponse(session_id=session_id, obj_id=obj_id)

    def reset_video_session(self, session_id: str) -> ResetSessionResponse:
        """Reset session to initial state (clears all prompts and objects).
        
        Args:
            session_id: The session ID
            
        Returns:
            The reset response
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        self.model.reset_session(session_id)

        # Clear stats
        objects_cleared = session["objects_count"]
        self.session_manager.update_session_stats(
            session_id, objects_count=0, frames_processed=0
        )
        self.session_manager.update_session_status(
            session_id, VideoSessionStatus.READY
        )

        return ResetSessionResponse(
            session_id=session_id, objects_cleared=objects_cleared
        )

    def close_video_session(self, session_id: str) -> CloseSessionResponse:
        """Close and cleanup video session.
        
        Args:
            session_id: The session ID
            
        Returns:
            The close response
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Get memory before closing
        session_info = self.model.get_session_info(session_id)
        memory_mb = session_info["gpu_memory_mb"]

        # Close SAM3 session
        self.model.close_session(session_id)

        # Remove from manager
        self.session_manager.delete_session(session_id)

        return CloseSessionResponse(
            session_id=session_id, memory_freed_mb=memory_mb
        )

    def list_video_sessions(self) -> SessionListResponse:
        """List all active video sessions.
        
        Returns:
            The session list response
        """
        sessions = self.session_manager.list_sessions()
        return SessionListResponse(sessions=sessions, total_sessions=len(sessions))