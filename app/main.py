"""
main.py — FastAPI application entry point.

Start the server:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
  Swagger UI  → http://localhost:8000/docs
  ReDoc       → http://localhost:8000/redoc
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import router as ws_router
from app.api.rest import router as rest_router
from app.utils.logging import setup_logging, get_logger
from app.config import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    setup_logging()
    logger.info(
        "voice_ai_startup",
        stt=settings.stt_provider,
        llm=f"{settings.llm_provider}/{settings.get_llm_model()}",
        tts=settings.tts_provider,
        host=settings.app_host,
        port=settings.app_port,
    )
    yield
    logger.info("voice_ai_shutdown")


app = FastAPI(
    title="Voice AI Insight Engine",
    description=(
        "Real-time Voice AI assistant with STT → LLM Entity/Sentiment Extraction → TTS. "
        "WebSocket endpoint for audio streaming. REST endpoints for session management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow all origins for POC; restrict in production) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(ws_router, tags=["WebSocket"])
app.include_router(rest_router, prefix="/api", tags=["Sessions"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "Voice AI Insight Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "ws://localhost:8000/ws/voice/new",
        "providers": {
            "stt": settings.stt_provider,
            "llm": f"{settings.llm_provider}/{settings.get_llm_model()}",
            "tts": settings.tts_provider,
        },
    }
