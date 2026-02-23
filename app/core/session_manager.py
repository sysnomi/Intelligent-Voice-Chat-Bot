"""
core/session_manager.py — In-memory session store for voice conversations.

Each WebSocket connection gets a unique session_id. The session stores the
full ConversationState (LangGraph state) for that connection's lifetime.

v2 upgrade path: Replace with Redis for distributed multi-server sessions.
"""
import time
import uuid
from typing import Dict

from app.brain.state import ConversationState
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Global in-memory store: session_id → (state, last_active_timestamp)
_sessions: Dict[str, tuple[ConversationState, float]] = {}


def create_session() -> str:
    """Create a new session, return its session_id."""
    _cleanup_expired()

    if len(_sessions) >= settings.max_sessions:
        raise RuntimeError(
            f"Max concurrent sessions reached ({settings.max_sessions}). "
            "Try again later."
        )

    session_id = str(uuid.uuid4())
    initial_state: ConversationState = {
        "session_id": session_id,
        "conversation_history": [],
        "current_transcript": "",
        "extracted_entities": [],
        "relationships": [],
        "sentiment": "Neutral",
        "intent": "other",
        "missing_info": [],
        "response_audio_text": "",
        "trigger_clarification": False,
        "pending_clarification": None,
        "turn_count": 0,
        "is_interrupted": False,
    }
    _sessions[session_id] = (initial_state, time.time())
    logger.info("session_created", session_id=session_id, total=len(_sessions))
    return session_id


def get_session(session_id: str) -> ConversationState | None:
    """Retrieve session state, or None if not found / expired."""
    entry = _sessions.get(session_id)
    if entry is None:
        return None
    state, _ = entry
    _sessions[session_id] = (state, time.time())  # Touch timestamp
    return state


def update_session(session_id: str, new_state: ConversationState) -> None:
    """Overwrite session state after a LangGraph run."""
    if session_id not in _sessions:
        logger.warning("session_update_miss", session_id=session_id)
        return
    _sessions[session_id] = (new_state, time.time())


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    if existed:
        logger.info("session_deleted", session_id=session_id, remaining=len(_sessions))
    return existed


def list_sessions() -> list[dict]:
    """Return summary info for all active sessions."""
    now = time.time()
    return [
        {
            "session_id": sid,
            "turn_count": state.get("turn_count", 0),
            "sentiment": state.get("sentiment", ""),
            "idle_seconds": round(now - ts, 1),
        }
        for sid, (state, ts) in _sessions.items()
    ]


def _cleanup_expired() -> None:
    """Remove sessions that haven't been active within the timeout window."""
    timeout = settings.session_timeout_seconds
    now = time.time()
    expired = [
        sid for sid, (_, ts) in _sessions.items()
        if now - ts > timeout
    ]
    for sid in expired:
        _sessions.pop(sid, None)
    if expired:
        logger.info("sessions_expired", count=len(expired))
