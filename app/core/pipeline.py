"""
core/pipeline.py — Orchestrates the full STT → Brain → TTS pipeline.

This is the core processing loop for one voice turn:
  1. Receive audio chunks from WebSocket
  2. Stream to STT (partial transcripts sent back to client)
  3. Run LangGraph brain on final transcript
  4. Stream TTS audio back to client chunk by chunk

All communication back to the client goes through a WebSocket send callback.
"""
import asyncio
import base64
import json
from typing import Callable, AsyncGenerator

from app.brain.graph import run_brain
from app.brain.state import ConversationState
from app.core.session_manager import get_session, update_session
from app.stt.factory import get_stt_provider
from app.tts.factory import get_tts_provider
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level singletons (created once, reused)
_stt = None
_tts = None


def _get_stt():
    global _stt
    if _stt is None:
        _stt = get_stt_provider()
    return _stt


def _get_tts():
    global _tts
    if _tts is None:
        _tts = get_tts_provider()
    return _tts


async def process_voice_turn(
    session_id: str,
    audio_generator: AsyncGenerator[bytes, None],
    send_json: Callable[[dict], None],
    is_interrupted: Callable[[], bool] | None = None,
) -> None:
    """
    Run one full voice turn for a session.

    Args:
        session_id:     The active session ID.
        audio_generator: Yields raw PCM audio byte chunks.
        send_json:      Async callable to send a dict as JSON to the client.
        is_interrupted: Optional callable; if returns True, abort TTS mid-stream.
    """
    stt = _get_stt()
    tts = _get_tts()
    state: ConversationState = get_session(session_id)

    if state is None:
        await send_json({"type": "error", "message": f"Session {session_id} not found"})
        return

    # ── 1. STT Phase ──────────────────────────────────────────────────────
    logger.info("pipeline_stt_start", session_id=session_id)

    async def on_partial(text: str):
        await send_json({"type": "transcript_partial", "text": text})

    async def on_final(text: str):
        await send_json({"type": "transcript_final", "text": text})

    # Bridge sync callbacks from STT → async send
    loop = asyncio.get_event_loop()

    def _on_partial_sync(text: str):
        asyncio.run_coroutine_threadsafe(on_partial(text), loop)

    def _on_final_sync(text: str):
        asyncio.run_coroutine_threadsafe(on_final(text), loop)

    transcript = await stt.transcribe_stream(
        audio_chunk_generator=audio_generator,
        on_partial=_on_partial_sync,
        on_final=_on_final_sync,
    )

    if not transcript.strip():
        await send_json({"type": "info", "message": "No speech detected"})
        return

    logger.info("pipeline_stt_done", transcript=transcript[:80])

    # ── 2. Brain Phase (LangGraph) ────────────────────────────────────────
    state["current_transcript"] = transcript
    logger.info("pipeline_brain_start", session_id=session_id)

    updated_state = await run_brain(state)
    update_session(session_id, updated_state)

    # Send extraction result to client
    await send_json({
        "type": "extraction",
        "entities": updated_state.get("extracted_entities", []),
        "relationships": updated_state.get("relationships", []),
        "sentiment": updated_state.get("sentiment", ""),
        "intent": updated_state.get("intent", ""),
        "missing_info": updated_state.get("missing_info", []),
        "trigger_clarification": updated_state.get("trigger_clarification", False),
        "clarification_question": updated_state.get("pending_clarification"),
        "turn_count": updated_state.get("turn_count", 0),
    })

    response_text = updated_state.get("response_audio_text", "")
    if not response_text.strip():
        return

    logger.info("pipeline_tts_start", session_id=session_id, chars=len(response_text))

    # ── 3. TTS Phase ──────────────────────────────────────────────────────
    sequence = 0
    async for audio_chunk in tts.synthesize_stream(response_text):
        # Check for interruption before sending each chunk
        if is_interrupted and is_interrupted():
            logger.info("pipeline_tts_interrupted", session_id=session_id)
            await send_json({"type": "interrupt"})
            return

        encoded = base64.b64encode(audio_chunk).decode("utf-8")
        await send_json({
            "type": "audio_chunk",
            "data": encoded,
            "sequence": sequence,
            "format": "mp3",
        })
        sequence += 1

    await send_json({"type": "audio_done", "sequence_total": sequence})
    logger.info("pipeline_done", session_id=session_id, chunks=sequence)
