"""
tts/kokoro_tts.py — Kokoro-82M self-hosted TTS (free, human-level quality).

Requires: pip install kokoro soundfile
The Kokoro model runs locally — no API key needed.
First run will download the model weights (~330MB).

Uncomment TTS_PROVIDER=kokoro in .env to use this instead of ElevenLabs.
"""
import asyncio
import io
from typing import AsyncGenerator

import soundfile as sf

from app.tts.base import BaseTTS
from app.utils.logging import get_logger

logger = get_logger(__name__)


class KokoroTTS(BaseTTS):
    def __init__(self, voice: str = "af_heart"):
        self._voice = voice
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
                self._pipeline = KPipeline(lang_code="a")  # 'a' = American English
            except ImportError:
                raise ImportError(
                    "Kokoro is not installed. Run: pip install kokoro\n"
                    "Or switch TTS_PROVIDER=elevenlabs in your .env"
                )
        return self._pipeline

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio via Kokoro and yield WAV byte chunks."""
        if not text or not text.strip():
            return

        logger.info("kokoro_tts_start", chars=len(text), voice=self._voice)

        def _generate():
            pipeline = self._get_pipeline()
            results = []
            # Kokoro returns (graphemes, phonemes, audio_array) tuples
            for _, _, audio in pipeline(text, voice=self._voice, speed=1.0):
                buf = io.BytesIO()
                sf.write(buf, audio, samplerate=24000, format="WAV", subtype="PCM_16")
                buf.seek(0)
                results.append(buf.read())
            return results

        chunks = await asyncio.to_thread(_generate)
        for chunk in chunks:
            yield chunk

        logger.info("kokoro_tts_done", chunks=len(chunks))

    async def close(self) -> None:
        self._pipeline = None
