"""
tests/test_session_manager.py — Tests for in-memory session store.
"""
import pytest
from app.core.session_manager import (
    create_session,
    get_session,
    update_session,
    delete_session,
    list_sessions,
)


def test_create_and_get_session():
    sid = create_session()
    state = get_session(sid)
    assert state is not None
    assert state["session_id"] == sid
    assert state["turn_count"] == 0
    assert state["conversation_history"] == []
    delete_session(sid)


def test_update_session():
    sid = create_session()
    state = get_session(sid)
    state["turn_count"] = 5
    state["sentiment"] = "Excited"
    update_session(sid, state)
    updated = get_session(sid)
    assert updated["turn_count"] == 5
    assert updated["sentiment"] == "Excited"
    delete_session(sid)


def test_delete_session():
    sid = create_session()
    assert delete_session(sid) is True
    assert get_session(sid) is None
    assert delete_session(sid) is False  # Already gone


def test_list_sessions():
    sid1 = create_session()
    sid2 = create_session()
    sessions = list_sessions()
    ids = [s["session_id"] for s in sessions]
    assert sid1 in ids
    assert sid2 in ids
    delete_session(sid1)
    delete_session(sid2)


def test_get_nonexistent_session():
    state = get_session("00000000-0000-0000-0000-000000000000")
    assert state is None
