"""Video inference API routes."""
import asyncio
import base64
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import JSONResponse

from app.schemas.video_schemas import (
    AddPromptRequest,
    AddPromptResponse,
    CloseSessionResponse,
    FrameResult,
    PropagateRequest,
    PropagateResponse,
    RemoveObjectResponse,
    ResetSessionResponse,
    SessionListResponse,
    SessionStatusResponse,
    StartSessionRequest,
    StartSessionResponse,
    StreamFrameMessage,
    VideoSessionStatus,
)
from app.schemas.video_schemas import VideoObject  # Import VideoObject for type hints
from app.services.video_service import VideoSegmentationService
from app.api.dependencies import get_video_service
from app.exceptions import ModelNotLoadedError, SessionNotFoundError, InvalidVideoSourceError

router = APIRouter()


def _save_video_from_request(request: StartSessionRequest) -> str:
    """
    Save video from request to temporary file.

    Args:
        request: StartSessionRequest with video source

    Returns:
        Path to saved video file

    Raises:
        HTTPException: If no valid video source provided
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
                raise HTTPException(
                    status_code=500, detail="Failed to download video from URL"
                )
                
            return str(temp_video_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to download video from URL: {str(e)}"
            )

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
            raise HTTPException(
                status_code=400, detail=f"Failed to decode video: {str(e)}"
            )

    raise HTTPException(
        status_code=400,
        detail="No valid video source provided (video_path, video_url, or video_base64 required)",
    )


@router.post("/session/start", response_model=StartSessionResponse)
async def start_video_session(
    request: StartSessionRequest,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """
    Start a new video inference session.

    Creates a session for video processing. The session maintains state
    for tracking objects across frames.
    """
    try:
        return video_service.start_session(request)
    except InvalidVideoSourceError:
        raise HTTPException(status_code=400, detail="No valid video source provided (video_path, video_url, or video_base64 required)")
    except Exception as e:
        print(f"Failed to start session: {e}")  # Using print as placeholder
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/prompt", response_model=AddPromptResponse)
async def add_prompt_to_frame(
    session_id: str,
    request: AddPromptRequest,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """
    Add prompts to a specific frame in the video.

    This initializes or refines object tracking for a particular frame.
    The prompts will be used as reference for propagation.
    """
    try:
        return video_service.add_prompt_to_frame(session_id, request)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except Exception as e:
        print(f"Failed to add prompt: {e}")  # Using print as placeholder for logger
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/propagate", response_model=PropagateResponse)
async def propagate_tracking(
    session_id: str,
    request: PropagateRequest,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """
    Propagate object tracking through video frames (non-streaming).

    Use this endpoint for batch processing. For real-time streaming,
    use the WebSocket endpoint at /ws/propagate/{session_id}.
    """
    if request.stream:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Use WebSocket endpoint /ws/propagate/{session_id} for streaming"
            },
        )

    try:
        return video_service.propagate_tracking(session_id, request)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except Exception as e:
        print(f"Propagation failed: {e}")  # Using print as placeholder for logger
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/propagate/{session_id}")
async def propagate_tracking_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming video propagation results.

    Sends frame results as they're processed for real-time feedback.
    """
    await websocket.accept()

    # Access app state via websocket scope for WebSocket connections
    app_state = websocket.scope["app"].state
    
    video_model = app_state.video_model
    session_manager = app_state.session_manager
    
    if video_model is None:
        await websocket.send_json(
            StreamFrameMessage(
                type="error", error="Video inference not enabled"
            ).dict()
        )
        await websocket.close()
        return

    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json(
            StreamFrameMessage(
                type="error", error=f"Session {session_id} not found"
            ).dict()
        )
        await websocket.close()
        return

    try:
        # Receive propagation request
        request_data = await websocket.receive_json()
        direction = request_data.get("direction", "both")
        start_frame_index = request_data.get("start_frame_index", None)
        max_frames = request_data.get("max_frames", None)

        session_manager.update_session_status(
            session_id, VideoSessionStatus.PROCESSING
        )

        frames_sent = 0

        # Stream frame results
        for frame_data in video_model.propagate_in_video(
            session_id=session_id,
            direction=direction,
            start_frame_index=start_frame_index,
            max_frames=max_frames,
        ):
            message = StreamFrameMessage(
                type="frame",
                frame_index=frame_data["frame_index"],
                objects=frame_data["objects"],
            )
            await websocket.send_json(message.dict())
            frames_sent += 1

        # Send completion message
        await websocket.send_json(
            StreamFrameMessage(type="complete", total_frames=frames_sent).dict()
        )

        session_manager.update_session_stats(
            session_id, frames_processed=frames_sent
        )
        session_manager.update_session_status(
            session_id, VideoSessionStatus.READY
        )

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")  # Using print as placeholder for logger
    except Exception as e:
        print(f"WebSocket error: {e}")  # Using print as placeholder for logger
        try:
            await websocket.send_json(
                StreamFrameMessage(type="error", error=str(e)).dict()
            )
        except:
            pass
        session_manager.update_session_status(
            session_id, VideoSessionStatus.ERROR, error=str(e)
        )
    finally:
        await websocket.close()


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """Get current status of a video session."""
    try:
        return video_service.get_session_status(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except Exception as e:
        print(f"Failed to get session status: {e}")  # Using print as placeholder for logger
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}/object/{obj_id}", response_model=RemoveObjectResponse)
async def remove_object_from_tracking(
    session_id: str,
    obj_id: int,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """Remove an object from tracking in the session."""
    try:
        return video_service.remove_object_from_tracking(session_id, obj_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except Exception as e:
        print(f"Failed to remove object: {e}")  # Using print as placeholder for logger
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/reset", response_model=ResetSessionResponse)
async def reset_video_session(
    session_id: str,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """Reset session to initial state (clears all prompts and objects)."""
    try:
        return video_service.reset_video_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except Exception as e:
        print(f"Failed to reset session: {e}")  # Using print as placeholder for logger
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}", response_model=CloseSessionResponse)
async def close_video_session(
    session_id: str,
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """Close and cleanup video session."""
    try:
        return video_service.close_video_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except Exception as e:
        print(f"Failed to close session: {e}")  # Using print as placeholder for logger
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_video_sessions(
    video_service: VideoSegmentationService = Depends(get_video_service)
):
    """List all active video sessions."""
    return video_service.list_video_sessions()


# Export router
video_router = router