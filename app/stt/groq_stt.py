"""
stt/groq_stt.py — Groq Whisper STT (batch per utterance, ultra-fast LPU inference).

Note: Groq does not support true streaming partial results.
Instead we buffer all audio chunks, write to a temp WAV file, and send
the whole utterance once VAD signals end-of-speech. This still achieves
sub-500ms turnaround thanks to Groq's LPU hardware.
"""
import asyncio
import io
import tempfile
import os
from typing import AsyncGenerator, Callable

import soundfile as sf
import numpy as np

from app.stt.base import BaseSTT
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GroqSTT(BaseSTT):
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                raise ImportError("Install: pip install groq")
        return self._client

    async def transcribe_stream(
        self,
        audio_chunk_generator: AsyncGenerator[bytes, None],
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
    ) -> str:
        # Buffer all PCM chunks
        pcm_buffer = bytearray()
        async for chunk in audio_chunk_generator:
            pcm_buffer.extend(chunk)

        if not pcm_buffer:
            return ""

        # Convert PCM bytes → in-memory WAV
        audio_np = np.frombuffer(bytes(pcm_buffer), dtype=np.int16).astype(np.float32) / 32768.0
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_np, samplerate=16000, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)

        logger.info("groq_stt_sending", audio_seconds=round(len(audio_np) / 16000, 2))

        def _transcribe():
            client = self._get_client()
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", wav_buffer, "audio/wav"),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en",
            )
            return transcription

        transcript = await asyncio.to_thread(_transcribe)
        transcript = transcript.strip()

        # Groq is batch-only — fire final immediately
        if on_partial:
            on_partial(transcript)
        if on_final:
            on_final(transcript)

        logger.info("groq_stt_done", transcript=transcript[:80])
        return transcript

    async def close(self) -> None:
        self._client = None
