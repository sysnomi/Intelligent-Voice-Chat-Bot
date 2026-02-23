"""
tts/factory.py — Return the correct TTS provider based on settings.
"""
from app.config import settings
from app.tts.base import BaseTTS


def get_tts_provider() -> BaseTTS:
    provider = settings.tts_provider.lower()

    if provider == "elevenlabs":
        from app.tts.elevenlabs_tts import ElevenLabsTTS
        return ElevenLabsTTS(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
        )

    if provider == "kokoro":
        from app.tts.kokoro_tts import KokoroTTS
        return KokoroTTS(voice=settings.kokoro_voice)

    raise ValueError(
        f"Unknown TTS_PROVIDER='{provider}'. Valid options: elevenlabs | kokoro"
    )
