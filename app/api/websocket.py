"""
api/websocket.py — WebSocket endpoint: ws://host/ws/voice/{session_id}

Protocol:
  Client → Server : Binary frames (raw PCM audio, 16kHz 16-bit mono)
                    OR JSON text frame: {"type": "interrupt"}
  Server → Client : JSON text frames (see MessageType enum below)

The endpoint manages the full lifecycle:
  connect → create/load session → stream audio → run pipeline → stream audio back
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.pipeline import process_voice_turn
from app.core.session_manager import create_session, get_session, delete_session
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """
    Main WebSocket endpoint for a voice session.

    Path param session_id: Pass "new" to auto-create a session, or an existing UUID.
    """
    await websocket.accept()
    logger.info("ws_connected", session_id=session_id)

    # Auto-create session if "new" is passed
    if session_id == "new":
        try:
            session_id = create_session()
        except RuntimeError as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1013)
            return

    # Validate existing session
    if get_session(session_id) is None:
        # Auto-create if session doesn't exist (client reconnect)
        from app.core.session_manager import _sessions
        session_id = create_session()

    await websocket.send_json({
        "type": "session_ready",
        "session_id": session_id,
        "message": "Connected. Send PCM audio chunks to begin.",
    })

    # ── Interruption flag shared between receiver and TTS streamer ──────
    _interrupted = False

    def is_interrupted() -> bool:
        return _interrupted

    # Async callable to send JSON back to client
    async def send_json(data: dict):
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(data)
            except Exception:
                pass  # Client disconnected mid-stream

    # ── Audio chunk buffer (filled by receiver, consumed by pipeline) ────
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def audio_generator():
        """Yield audio chunks from the queue until None sentinel."""
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            yield chunk

    try:
        while True:
            # Receive the next message (binary audio OR JSON control)
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── JSON control messages ────────────────────────────────
            if "text" in message:
                try:
                    control = json.loads(message["text"])
                    msg_type = control.get("type", "")

                    if msg_type == "audio_end":
                        # Client signals end of utterance → trigger pipeline
                        audio_queue.put_nowait(None)  # Sentinel
                        _interrupted = False
                        await process_voice_turn(
                            session_id=session_id,
                            audio_generator=audio_generator(),
                            send_json=send_json,
                            is_interrupted=is_interrupted,
                        )
                        # Reset queue for next utterance
                        audio_queue = asyncio.Queue()

                    elif msg_type == "interrupt":
                        _interrupted = True
                        await send_json({"type": "interrupt_ack"})

                    elif msg_type == "ping":
                        await send_json({"type": "pong"})

                except json.JSONDecodeError:
                    pass

            # ── Binary audio chunks ──────────────────────────────────
            elif "bytes" in message:
                _interrupted = False  # Reset on new audio
                audio_queue.put_nowait(message["bytes"])

    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("ws_error", session_id=session_id, error=str(e))
        await send_json({"type": "error", "message": str(e)})
    finally:
        logger.info("ws_cleanup", session_id=session_id)
        # Note: We intentionally do NOT delete the session here so the client
        # can reconnect and resume the conversation.
