"""
stt/azure_stt.py — Azure AI Speech continuous recognition (streaming partials).

Requires:
  AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env

The Azure SDK uses callbacks. We bridge them to asyncio by posting results
onto an asyncio.Queue from a background thread.
"""
import asyncio
from typing import AsyncGenerator, Callable

from app.stt.base import BaseSTT
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AzureSTT(BaseSTT):
    def __init__(self, speech_key: str, speech_region: str):
        self._key = speech_key
        self._region = speech_region
        self._recognizer = None

    def _build_recognizer(self, stream):
        """Build an Azure SpeechRecognizer from a push-stream."""
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-speech is not installed. "
                "Run: pip install azure-cognitiveservices-speech"
            )

        speech_config = speechsdk.SpeechConfig(
            subscription=self._key, region=self._region
        )
        speech_config.speech_recognition_language = "en-US"
        # Enable detailed output for confidence scores
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        audio_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        return recognizer, audio_stream

    async def transcribe_stream(
        self,
        audio_chunk_generator: AsyncGenerator[bytes, None],
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
    ) -> str:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            raise ImportError("Install: pip install azure-cognitiveservices-speech")

        loop = asyncio.get_event_loop()
        result_queue: asyncio.Queue[str | None] = asyncio.Queue()
        final_parts: list[str] = []

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        speech_config = speechsdk.SpeechConfig(
            subscription=self._key, region=self._region
        )
        speech_config.speech_recognition_language = "en-US"

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # ── Callbacks (called from SDK thread) ──────────────────────────
        def on_recognizing(evt):
            text = evt.result.text
            if text and on_partial:
                loop.call_soon_threadsafe(on_partial, text)

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text
                if text:
                    final_parts.append(text)
                    if on_final:
                        loop.call_soon_threadsafe(on_final, text)

        def on_session_stopped(evt):
            loop.call_soon_threadsafe(result_queue.put_nowait, None)

        def on_canceled(evt):
            logger.warning("azure_stt_canceled", reason=str(evt.result.reason))
            loop.call_soon_threadsafe(result_queue.put_nowait, None)

        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_session_stopped)
        recognizer.canceled.connect(on_canceled)

        recognizer.start_continuous_recognition()
        logger.info("azure_stt_started")

        try:
            # Pump audio chunks into the push stream
            async for chunk in audio_chunk_generator:
                push_stream.write(chunk)

            push_stream.close()  # Signal end-of-audio
            await result_queue.get()  # Wait for session_stopped
        finally:
            recognizer.stop_continuous_recognition()

        full_transcript = " ".join(final_parts).strip()
        logger.info("azure_stt_done", transcript_length=len(full_transcript))
        return full_transcript

    async def close(self) -> None:
        pass  # SDK resources are cleaned up per-call
