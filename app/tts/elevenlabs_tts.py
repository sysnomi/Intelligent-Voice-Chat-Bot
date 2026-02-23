"""
tts/elevenlabs_tts.py — ElevenLabs streaming TTS (default for POC).

Uses the ElevenLabs Python SDK with streaming enabled.
Yields raw MP3 audio byte chunks for low-latency playback.

Free tier: 10,000 characters/month
"""
import asyncio
from typing import AsyncGenerator

from app.tts.base import BaseTTS
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ElevenLabsTTS(BaseTTS):
    def __init__(self, api_key: str, voice_id: str):
        self._api_key = api_key
        self._voice_id = voice_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs
                self._client = ElevenLabs(api_key=self._api_key)
            except ImportError:
                raise ImportError("Install: pip install elevenlabs")
        return self._client

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream MP3 audio chunks from ElevenLabs."""
        if not text or not text.strip():
            return

        logger.info("elevenlabs_tts_start", chars=len(text), voice=self._voice_id)

        def _generate_chunks():
            client = self._get_client()
            audio_stream = client.text_to_speech.convert_as_stream(
                voice_id=self._voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",   # Lowest latency model
                output_format="mp3_44100_128",
            )
            return list(audio_stream)  # Collect in thread, yield async outside

        chunks = await asyncio.to_thread(_generate_chunks)

        chunk_count = 0
        for chunk in chunks:
            if chunk:
                chunk_count += 1
                yield chunk

        logger.info("elevenlabs_tts_done", chunks=chunk_count)

    async def close(self) -> None:
        self._client = None
