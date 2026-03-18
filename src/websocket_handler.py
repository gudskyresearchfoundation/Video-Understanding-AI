"""
WebSocket connection manager.
Enforces the 10-concurrent-user limit and handles per-session lifecycle.
"""

import asyncio
import logging
import os
import uuid
from typing import Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

MAX_CONNECTIONS: int = int(os.getenv("MAX_WS_CONNECTIONS", "10"))


class SessionState:
    """Holds state for a single connected user."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id:  str       = session_id
        self.websocket:   WebSocket = websocket
        self.video_path:  Optional[str] = None
        self.history:     list      = []


class ConnectionManager:
    """
    Async-safe WebSocket manager with a hard cap of MAX_CONNECTIONS sessions.
    Cleans up temp video files on disconnect.
    """

    def __init__(self):
        self._lock     = asyncio.Lock()
        self.sessions: Dict[str, SessionState] = {}

    async def connect(self, websocket: WebSocket) -> Optional[str]:
        await websocket.accept()
        async with self._lock:
            if len(self.sessions) >= MAX_CONNECTIONS:
                await websocket.send_json({
                    "type":    "error",
                    "code":    "CAPACITY",
                    "message": (
                        f"Server is at capacity ({MAX_CONNECTIONS} active users). "
                        "Please try again in a moment."
                    ),
                })
                await websocket.close(code=1013)
                logger.warning("[WS] Rejected — at capacity")
                return None

            session_id = str(uuid.uuid4())
            self.sessions[session_id] = SessionState(session_id, websocket)
            logger.info(
                f"[WS] +connected  id={session_id[:8]}  "
                f"total={len(self.sessions)}/{MAX_CONNECTIONS}"
            )
            return session_id

    async def disconnect(self, session_id: str) -> None:
        async with self._lock:
            session = self.sessions.pop(session_id, None)
        if session and session.video_path:
            try:
                import os as _os
                if _os.path.exists(session.video_path):
                    _os.remove(session.video_path)
            except Exception:
                pass
        logger.info(
            f"[WS] -disconnected id={session_id[:8]}  "
            f"total={len(self.sessions)}/{MAX_CONNECTIONS}"
        )

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self.sessions)

    async def send(self, session_id: str, payload: dict) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        try:
            await session.websocket.send_json(payload)
        except Exception as exc:
            logger.debug(f"[WS] send failed {session_id[:8]}: {exc}")


_manager: Optional[ConnectionManager] = None


def get_manager() -> ConnectionManager:
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
