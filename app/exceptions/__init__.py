"""Custom exceptions for the SAM3 Inference Server."""

from fastapi import HTTPException


class ModelNotLoadedError(HTTPException):
    """Raised when a model is not loaded but is required for an operation."""
    
    def __init__(self, model_type: str = "unknown"):
        super().__init__(
            status_code=503,
            detail=f"{model_type.capitalize()} model not loaded"
        )


class SessionNotFoundError(HTTPException):
    """Raised when a session is not found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            status_code=404,
            detail=f"Session {session_id} not found"
        )


class InvalidVideoSourceError(HTTPException):
    """Raised when no valid video source is provided."""
    
    def __init__(self):
        super().__init__(
            status_code=400,
            detail="No valid video source provided (video_path, video_url, or video_base64 required)"
        )


class InvalidImageFormatError(HTTPException):
    """Raised when an invalid image format is provided."""
    
    def __init__(self, message: str = "Invalid image data"):
        super().__init__(
            status_code=400,
            detail=message
        )


class MaxSessionsExceededError(HTTPException):
    """Raised when maximum number of sessions is exceeded."""
    
    def __init__(self, max_sessions: int):
        super().__init__(
            status_code=429,
            detail=f"Maximum sessions ({max_sessions}) exceeded. Please close inactive sessions."
        )