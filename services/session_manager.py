"""Session manager for video inference sessions."""
import time
from datetime import datetime
from typing import Dict, List, Optional

from sam3.logger import get_logger

from api.schemas.video_schemas import VideoSessionStatus

logger = get_logger(__name__)


class SessionManager:
    """Manages video inference sessions with metadata and lifecycle."""

    def __init__(self, max_sessions: int = 100, session_timeout_seconds: int = 3600):
        """
        Initialize session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions
            session_timeout_seconds: Session timeout in seconds (default: 1 hour)
        """
        self.max_sessions = max_sessions
        self.session_timeout_seconds = session_timeout_seconds
        self._sessions: Dict[str, Dict] = {}

    def create_session(
        self,
        session_id: str,
        session_type: str = "video",
        video_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a new session with metadata.

        Args:
            session_id: Unique session ID
            session_type: "video" or "image_batch"
            video_info: Video metadata (total_frames, resolution, etc.)

        Returns:
            Session metadata dict

        Raises:
            RuntimeError: If max sessions exceeded
        """
        if len(self._sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.max_sessions}) exceeded. "
                    f"Please close inactive sessions."
                )

        session_data = {
            "session_id": session_id,
            "type": session_type,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_accessed": time.time(),
            "status": VideoSessionStatus.READY,
            "video_info": video_info or {},
            "objects_count": 0,
            "frames_processed": 0,
            "error": None,
        }

        self._sessions[session_id] = session_data
        logger.info(f"Created session {session_id} (type: {session_type})")
        return session_data

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session metadata.

        Args:
            session_id: Session ID

        Returns:
            Session metadata or None if not found
        """
        session = self._sessions.get(session_id)
        if session:
            session["last_accessed"] = time.time()
        return session

    def update_session_status(
        self, session_id: str, status: VideoSessionStatus, error: Optional[str] = None
    ):
        """
        Update session status.

        Args:
            session_id: Session ID
            status: New status
            error: Error message if status is ERROR
        """
        if session_id in self._sessions:
            self._sessions[session_id]["status"] = status
            self._sessions[session_id]["last_accessed"] = time.time()
            if error:
                self._sessions[session_id]["error"] = error
            logger.debug(f"Session {session_id} status updated to {status}")

    def update_session_stats(
        self,
        session_id: str,
        objects_count: Optional[int] = None,
        frames_processed: Optional[int] = None,
    ):
        """
        Update session statistics.

        Args:
            session_id: Session ID
            objects_count: Number of tracked objects
            frames_processed: Number of processed frames
        """
        if session_id in self._sessions:
            if objects_count is not None:
                self._sessions[session_id]["objects_count"] = objects_count
            if frames_processed is not None:
                self._sessions[session_id]["frames_processed"] = frames_processed
            self._sessions[session_id]["last_accessed"] = time.time()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session metadata.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def list_sessions(self) -> List[Dict]:
        """
        List all active sessions.

        Returns:
            List of session metadata dicts
        """
        return list(self._sessions.values())

    def get_session_count(self) -> int:
        """
        Get total number of active sessions.

        Returns:
            Session count
        """
        return len(self._sessions)

    def _cleanup_expired_sessions(self):
        """Remove expired sessions based on timeout."""
        current_time = time.time()
        expired = []

        for session_id, session in self._sessions.items():
            if (
                current_time - session["last_accessed"]
            ) > self.session_timeout_seconds:
                expired.append(session_id)

        for session_id in expired:
            self.delete_session(session_id)
            logger.info(f"Removed expired session {session_id}")

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def clear_all_sessions(self):
        """Clear all sessions (for shutdown)."""
        count = len(self._sessions)
        self._sessions.clear()
        logger.info(f"Cleared all {count} sessions")
