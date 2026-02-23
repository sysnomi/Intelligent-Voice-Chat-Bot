"""
tts/base.py — Abstract interface all TTS providers must implement.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseTTS(ABC):
    """
    Abstract Text-to-Speech provider.

    Subclasses must implement:
      - synthesize_stream: yields raw audio bytes chunks from input text
    """

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Convert text to speech and yield audio bytes chunks progressively.
        Each chunk is a small WAV/MP3 segment suitable for streaming playback.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any SDK clients or connections."""
        ...
