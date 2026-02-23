"""
config.py — Centralised settings loaded from .env
All provider selections and API keys live here.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ───────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    max_sessions: int = 50
    session_timeout_seconds: int = 300

    # ── STT ───────────────────────────────────
    stt_provider: str = Field(default="azure")  # azure | groq
    azure_speech_key: str = ""
    azure_speech_region: str = "eastus"

    # ── LLM ───────────────────────────────────
    llm_provider: str = Field(default="openai")  # openai | gemini | groq | azure_openai
    llm_model: str = ""  # Empty = use provider default

    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"

    # ── TTS ───────────────────────────────────
    tts_provider: str = Field(default="elevenlabs")  # elevenlabs | kokoro
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    kokoro_voice: str = "af_heart"

    def get_llm_model(self) -> str:
        """Return the effective LLM model name based on provider."""
        if self.llm_model:
            return self.llm_model
        defaults = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "groq": "llama-3.3-70b-versatile",
            "azure_openai": self.azure_openai_deployment,
        }
        return defaults.get(self.llm_provider, "gpt-4o-mini")


settings = Settings()
