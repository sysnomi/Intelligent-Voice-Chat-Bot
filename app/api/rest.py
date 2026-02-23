"""
api/rest.py — REST endpoints for health checks and session management.
"""
from fastapi import APIRouter, HTTPException
from app.core.session_manager import (
    create_session,
    get_session,
    delete_session,
    list_sessions,
)
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check — confirms server is running and shows active config."""
    return {
        "status": "ok",
        "stt_provider": settings.stt_provider,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.get_llm_model(),
        "tts_provider": settings.tts_provider,
        "active_sessions": len(list_sessions()),
    }


@router.post("/sessions")
async def start_session():
    """Create a new conversation session. Returns the session_id."""
    try:
        session_id = create_session()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"session_id": session_id}


@router.get("/sessions")
async def get_all_sessions():
    """List all active sessions with summary info."""
    return {"sessions": list_sessions()}


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get the full state of a specific session."""
    state = get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": state["session_id"],
        "turn_count": state["turn_count"],
        "sentiment": state["sentiment"],
        "intent": state["intent"],
        "extracted_entities": state["extracted_entities"],
        "relationships": state["relationships"],
        "missing_info": state["missing_info"],
        "conversation_history": state["conversation_history"],
    }


@router.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """Terminate and delete a session."""
    existed = delete_session(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}
