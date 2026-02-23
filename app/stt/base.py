"""
stt/base.py — Abstract interface all STT providers must implement.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable


class BaseSTT(ABC):
    """
    Abstract Speech-to-Text provider.

    Subclasses must implement:
      - transcribe_stream: feeds PCM chunks, yields partial/final texts
    """

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunk_generator: AsyncGenerator[bytes, None],
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
    ) -> str:
        """
        Consume an async generator of raw PCM chunks.
        Call on_partial(text) for interim results.
        Call on_final(text) for the committed final result.
        Return the full final transcript string.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any open connections or SDK resources."""
        ...
