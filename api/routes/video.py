"""Video inference API routes."""
import asyncio
import base64
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

# Removed import server to break circular dependency
from api.schemas.video_schemas import (
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
from sam3.logger import get_logger

logger = get_logger(__name__)
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
        from urllib.parse import urlparse
        
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
async def start_video_session(request: StartSessionRequest, req: Request):
    """
    Start a new video inference session.

    Creates a session for video processing. The session maintains state
    for tracking objects across frames.
    """
    if req.app.state.video_model is None:
        raise HTTPException(
            status_code=503, detail="Video inference is not enabled on this server"
        )

    try:
        # Save video to file
        video_path = _save_video_from_request(request)

        # Start session with SAM3
        session_id, video_info = req.app.state.video_model.start_session(
            video_path=video_path, session_id=request.session_id
        )

        # Register session in manager
        req.app.state.session_manager.create_session(
            session_id=session_id, session_type="video", video_info=video_info
        )

        return StartSessionResponse(
            session_id=session_id,
            video_info=video_info,
            status=VideoSessionStatus.READY,
        )

    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/prompt", response_model=AddPromptResponse)
async def add_prompt_to_frame(session_id: str, request: AddPromptRequest, req: Request):
    """
    Add prompts to a specific frame in the video.

    This initializes or refines object tracking for a particular frame.
    The prompts will be used as reference for propagation.
    """
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    # Check session exists
    session = req.app.state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        req.app.state.session_manager.update_session_status(
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
        logger.info(f"Frame index is  {request.frame_index}...")
        frame_idx, obj_ids, masks, boxes_out, scores = req.app.state.video_model.add_prompt(
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
        session_info = req.app.state.video_model.get_session_info(session_id)
        req.app.state.session_manager.update_session_stats(
            session_id, objects_count=session_info["num_objects"]
        )
        req.app.state.session_manager.update_session_status(
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
        req.app.state.session_manager.update_session_status(
            session_id, VideoSessionStatus.ERROR, error=str(e)
        )
        logger.error(f"Failed to add prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/propagate", response_model=PropagateResponse)
async def propagate_tracking(session_id: str, request: PropagateRequest, req: Request):
    """
    Propagate object tracking through video frames (non-streaming).

    Use this endpoint for batch processing. For real-time streaming,
    use the WebSocket endpoint at /ws/propagate/{session_id}.
    """
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    session = req.app.state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if request.stream:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Use WebSocket endpoint /ws/propagate/{session_id} for streaming"
            },
        )

    try:
        req.app.state.session_manager.update_session_status(
            session_id, VideoSessionStatus.PROCESSING
        )

        start_time = time.time()
        results: Dict[int, FrameResult] = {}

        # Propagate through video
        for frame_data in req.app.state.video_model.propagate_in_video(
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
        req.app.state.session_manager.update_session_stats(
            session_id, frames_processed=len(results)
        )
        req.app.state.session_manager.update_session_status(
            session_id, VideoSessionStatus.READY
        )

        return PropagateResponse(
            session_id=session_id,
            results=results,
            total_frames=len(results),
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        req.app.state.session_manager.update_session_status(
            session_id, VideoSessionStatus.ERROR, error=str(e)
        )
        logger.error(f"Propagation failed: {e}")
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
    
    if app_state.video_model is None:
        await websocket.send_json(
            StreamFrameMessage(
                type="error", error="Video inference not enabled"
            ).model_dump()
        )
        await websocket.close()
        return

    session = app_state.session_manager.get_session(session_id)
    if not session:
        await websocket.send_json(
            StreamFrameMessage(
                type="error", error=f"Session {session_id} not found"
            ).model_dump()
        )
        await websocket.close()
        return

    try:
        # Receive propagation request
        request_data = await websocket.receive_json()
        direction = request_data.get("direction", "both")
        start_frame_index = request_data.get("start_frame_index", None)
        max_frames = request_data.get("max_frames", None)

        app_state.session_manager.update_session_status(
            session_id, VideoSessionStatus.PROCESSING
        )

        frames_sent = 0

        # Stream frame results
        for frame_data in app_state.video_model.propagate_in_video(
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
            await websocket.send_json(message.model_dump())
            frames_sent += 1

        # Send completion message
        await websocket.send_json(
            StreamFrameMessage(type="complete", total_frames=frames_sent).model_dump()
        )

        app_state.session_manager.update_session_stats(
            session_id, frames_processed=frames_sent
        )
        app_state.session_manager.update_session_status(
            session_id, VideoSessionStatus.READY
        )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json(
                StreamFrameMessage(type="error", error=str(e)).model_dump()
            )
        except:
            pass
        app_state.session_manager.update_session_status(
            session_id, VideoSessionStatus.ERROR, error=str(e)
        )
    finally:
        await websocket.close()


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str, req: Request):
    """Get current status of a video session."""
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    session = req.app.state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        session_info = req.app.state.video_model.get_session_info(session_id)

        return SessionStatusResponse(
            session_id=session_id,
            status=session["status"],
            current_objects=session_info["num_objects"],
            frames_processed=session["frames_processed"],
            total_frames=session_info["num_frames"],
            gpu_memory_used_mb=session_info["gpu_memory_mb"],
        )
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}/object/{obj_id}", response_model=RemoveObjectResponse)
async def remove_object_from_tracking(session_id: str, obj_id: int, req: Request):
    """Remove an object from tracking in the session."""
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    session = req.app.state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        req.app.state.video_model.remove_object(session_id, obj_id)

        # Update stats
        session_info = req.app.state.video_model.get_session_info(session_id)
        req.app.state.session_manager.update_session_stats(
            session_id, objects_count=session_info["num_objects"]
        )

        return RemoveObjectResponse(session_id=session_id, obj_id=obj_id)

    except Exception as e:
        logger.error(f"Failed to remove object: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/reset", response_model=ResetSessionResponse)
async def reset_video_session(session_id: str, req: Request):
    """Reset session to initial state (clears all prompts and objects)."""
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    session = req.app.state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        req.app.state.video_model.reset_session(session_id)

        # Clear stats
        objects_cleared = session["objects_count"]
        req.app.state.session_manager.update_session_stats(
            session_id, objects_count=0, frames_processed=0
        )
        req.app.state.session_manager.update_session_status(
            session_id, VideoSessionStatus.READY
        )

        return ResetSessionResponse(
            session_id=session_id, objects_cleared=objects_cleared
        )

    except Exception as e:
        logger.error(f"Failed to reset session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}", response_model=CloseSessionResponse)
async def close_video_session(session_id: str, req: Request):
    """Close and cleanup video session."""
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    session = req.app.state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        # Get memory before closing
        session_info = req.app.state.video_model.get_session_info(session_id)
        memory_mb = session_info["gpu_memory_mb"]

        # Close SAM3 session
        req.app.state.video_model.close_session(session_id)

        # Remove from manager
        req.app.state.session_manager.delete_session(session_id)

        return CloseSessionResponse(
            session_id=session_id, memory_freed_mb=memory_mb
        )

    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_video_sessions(req: Request):
    """List all active video sessions."""
    if req.app.state.video_model is None:
        raise HTTPException(status_code=503, detail="Video inference not enabled")

    sessions = req.app.state.session_manager.list_sessions()

    return SessionListResponse(sessions=sessions, total_sessions=len(sessions))
