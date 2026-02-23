"""
utils/audio.py — PCM audio helpers: conversion, resampling, base64 encoding.
"""
import base64
import io
import numpy as np
import soundfile as sf


def pcm_bytes_to_numpy(pcm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert raw 16-bit PCM bytes to a float32 numpy array."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert a float32 numpy array to WAV bytes (in-memory)."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    """Base64-encode WAV bytes for JSON transport."""
    return base64.b64encode(wav_bytes).decode("utf-8")


def chunk_audio_bytes(audio_bytes: bytes, chunk_size: int = 4096):
    """Yield audio_bytes in fixed-size chunks."""
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i : i + chunk_size]
