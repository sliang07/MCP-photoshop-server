"""
Session manager for per-connection document state.
"""

import uuid
from typing import Dict, Optional

from canvas import Canvas


class SessionManager:
    """Manages active documents per session."""

    def __init__(self):
        self._sessions: Dict[str, Canvas] = {}

    def get_or_create(self, session_id: str) -> Canvas:
        """Get existing canvas or create a new one for this session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = Canvas()
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Canvas]:
        """Get canvas for session, or None if not exists."""
        return self._sessions.get(session_id)

    def create(self, session_id: str, width: int = 1024, height: int = 1024,
               bg_color: tuple = (255, 255, 255)) -> Canvas:
        """Create a new canvas for this session, replacing any existing one."""
        self._sessions[session_id] = Canvas(width, height, bg_color)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        """Delete a session's canvas."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_default_session(self) -> Canvas:
        """Get or create a default session (for single-user setups)."""
        return self.get_or_create("default")