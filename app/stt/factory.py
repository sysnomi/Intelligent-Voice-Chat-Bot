"""
stt/factory.py — Return the correct STT provider based on settings.
"""
from app.config import settings
from app.stt.base import BaseSTT


def get_stt_provider() -> BaseSTT:
    provider = settings.stt_provider.lower()

    if provider == "azure":
        from app.stt.azure_stt import AzureSTT
        return AzureSTT(
            speech_key=settings.azure_speech_key,
            speech_region=settings.azure_speech_region,
        )

    if provider == "groq":
        from app.stt.groq_stt import GroqSTT
        return GroqSTT(api_key=settings.groq_api_key)

    raise ValueError(
        f"Unknown STT_PROVIDER='{provider}'. Valid options: azure | groq"
    )
