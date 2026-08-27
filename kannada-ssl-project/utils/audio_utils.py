"""
Audio Utilities for Kannada Speech Processing
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 for wav2vec2)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        logger.info(f"Loaded audio: {file_path}, duration={len(audio)/sr:.2f}s, sr={sr}")
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        raise


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio from bytes (for web upload handling)."""
    import io
    import soundfile as sf
    import librosa

    try:
        buf = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buf)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32), target_sr
    except Exception as e:
        logger.error(f"Failed to load audio from bytes: {e}")
        raise


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def trim_silence(audio: np.ndarray, sr: int = SAMPLE_RATE, top_db: int = 20) -> np.ndarray:
    """Trim leading/trailing silence from audio."""
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except Exception:
        return audio


def split_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_duration: float = 10.0,
    overlap: float = 0.5,
) -> list:
    """
    Split long audio into overlapping chunks for processing.

    Args:
        audio: Audio array
        sr: Sample rate
        max_duration: Maximum chunk duration in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        List of audio chunks
    """
    max_samples = int(max_duration * sr)
    overlap_samples = int(overlap * sr)
    step = max_samples - overlap_samples

    if len(audio) <= max_samples:
        return [audio]

    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        chunks.append(audio[start:end])
        start += step
        if end == len(audio):
            break

    return chunks


def compute_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 40,
) -> np.ndarray:
    """Compute MFCC features for audio."""
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def get_audio_stats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Get basic statistics about an audio signal."""
    return {
        "duration_seconds": len(audio) / sr,
        "sample_rate": sr,
        "num_samples": len(audio),
        "rms_energy": float(np.sqrt(np.mean(audio ** 2))),
        "max_amplitude": float(np.max(np.abs(audio))),
        "min_amplitude": float(np.min(np.abs(audio))),
    }


def list_kannada_audio_files(audio_dir: str) -> list:
    """List all WAV files in the Kannada dataset directory."""
    if not os.path.exists(audio_dir):
        logger.warning(f"Audio directory not found: {audio_dir}")
        return []

    wav_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    ]
    wav_files.sort()
    logger.info(f"Found {len(wav_files)} WAV files in {audio_dir}")
    return wav_files
