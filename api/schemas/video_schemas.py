"""Pydantic schemas for video inference endpoints."""
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .common_schemas import Prompt


class VideoSessionStatus(str, Enum):
    """Status of video session."""

    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    CLOSED = "closed"


class PropagationDirection(str, Enum):
    """Direction for propagation."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


class VideoInfo(BaseModel):
    """Video metadata."""

    total_frames: int
    fps: float
    resolution: Dict[str, int]  # {"width": int, "height": int}
    duration_seconds: float


class StartSessionRequest(BaseModel):
    """Request to start video session."""

    video_url: Optional[str] = Field(
        None, description="URL to video file (http/https)"
    )
    video_base64: Optional[str] = Field(None, description="Base64-encoded video")
    video_path: Optional[str] = Field(None, description="Local path to video")
    session_id: Optional[str] = Field(
        None, description="Custom session ID (auto-generated if not provided)"
    )
    gpu_ids: Optional[List[int]] = Field(
        None, description="GPU IDs for multi-GPU processing"
    )


class StartSessionResponse(BaseModel):
    """Response for starting video session."""

    session_id: str
    video_info: VideoInfo
    status: VideoSessionStatus


class AddPromptRequest(BaseModel):
    """Request to add prompt to video session."""

    frame_index: int = Field(..., ge=0, description="Frame index to add prompt")
    prompts: List[Prompt] = Field(..., min_length=1)
    obj_id: Optional[int] = Field(
        None, description="Object ID to refine (None = new object)"
    )


class VideoObject(BaseModel):
    """Single object in video frame."""

    id: int = Field(..., description="Unique object ID")
    mask: str = Field(..., description="RLE-encoded mask")
    box: List[float] = Field(..., description="Bounding box [cx, cy, w, h]")
    score: float = Field(..., description="Confidence score")


class AddPromptResponse(BaseModel):
    """Response for adding prompt."""

    frame_index: int
    obj_id: List[int]
    masks: List[str]
    boxes: List[List[float]]
    scores: List[float]
    status: str = "prompt_added"


class PropagateRequest(BaseModel):
    """Request to propagate tracking through video."""

    direction: PropagationDirection = PropagationDirection.BOTH
    start_frame_index: int = Field(default=0, ge=0)
    max_frames: Optional[int] = Field(None, description="Max frames to process (None=all)")
    stream: bool = Field(
        default=False, description="Use WebSocket streaming if true"
    )


class FrameResult(BaseModel):
    """Result for single video frame."""

    frame_index: int
    objects: List[VideoObject]


class PropagateResponse(BaseModel):
    """Response for propagation (non-streaming)."""

    session_id: str
    results: Dict[int, FrameResult]  # frame_index -> FrameResult
    total_frames: int
    processing_time_ms: float


class StreamFrameMessage(BaseModel):
    """WebSocket message for streaming frame result."""

    type: Literal["frame", "complete", "error"] = "frame"
    frame_index: Optional[int] = None
    objects: Optional[List[VideoObject]] = None
    total_frames: Optional[int] = None
    error: Optional[str] = None


class SessionStatusResponse(BaseModel):
    """Response for session status."""

    session_id: str
    status: VideoSessionStatus
    current_objects: int
    frames_processed: int
    total_frames: int
    gpu_memory_used_mb: float


class RemoveObjectResponse(BaseModel):
    """Response for removing object."""

    session_id: str
    obj_id: int
    status: str = "removed"


class ResetSessionResponse(BaseModel):
    """Response for resetting session."""

    session_id: str
    status: str = "reset"
    objects_cleared: int


class CloseSessionResponse(BaseModel):
    """Response for closing session."""

    session_id: str
    status: str = "closed"
    memory_freed_mb: float


class SessionListItem(BaseModel):
    """Single session in list."""

    session_id: str
    type: Literal["video", "image_batch"]
    created_at: str  # ISO format timestamp
    status: VideoSessionStatus
    objects_count: Optional[int] = None
    images_processed: Optional[int] = None


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: List[SessionListItem]
    total_sessions: int
