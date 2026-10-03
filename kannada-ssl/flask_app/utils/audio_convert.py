"""
Audio Format Conversion

Browsers record microphone audio as WebM/Opus (via MediaRecorder), but neither
librosa/soundfile (used for the Kannada wav2vec2 model) nor the `SpeechRecognition`
package's AudioFile reader (used for other languages) can read WebM directly —
SpeechRecognition only accepts WAV/AIFF/FLAC, and soundfile needs an audioread
fallback that isn't reliable across environments.

This module normalizes ANY uploaded audio (webm, mp3, ogg, m4a, wav, ...) into a
16kHz mono WAV file using ffmpeg before it's handed to either ASR path.
"""

import os
import shutil
import subprocess
import uuid
import logging

logger = logging.getLogger(__name__)


class AudioConversionError(Exception):
    pass


_ffmpeg_path_cache = None


def _find_ffmpeg() -> str:
    """
    Resolve a usable ffmpeg binary.

    1. A system-installed ffmpeg on PATH (fastest, used if present).
    2. The static binary bundled by the `imageio-ffmpeg` pip package. This means
       users do NOT need to `sudo apt install` / `brew install` ffmpeg — installing
       Python requirements is enough. imageio-ffmpeg downloads a small platform
       binary on first use and caches it locally.

    Returns "" if neither is available.
    """
    global _ffmpeg_path_cache
    if _ffmpeg_path_cache is not None:
        return _ffmpeg_path_cache

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _ffmpeg_path_cache = system_ffmpeg
        return _ffmpeg_path_cache

    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            _ffmpeg_path_cache = bundled
            return _ffmpeg_path_cache
    except Exception as e:
        logger.warning(f"imageio-ffmpeg fallback unavailable: {e}")

    _ffmpeg_path_cache = ""
    return _ffmpeg_path_cache


def ffmpeg_available() -> bool:
    return bool(_find_ffmpeg())


def ensure_wav(input_path: str, output_dir: str = None, sample_rate: int = 16000) -> str:
    """
    Convert an audio file to 16kHz mono WAV using ffmpeg, if it isn't already one.

    Args:
        input_path: Path to the source audio file (any format ffmpeg supports)
        output_dir: Directory to write the converted WAV into (defaults to input's dir)
        sample_rate: Target sample rate

    Returns:
        Path to a WAV file (either the original, if it was already a valid WAV, or
        a newly converted one).

    Raises:
        AudioConversionError: if ffmpeg is missing or the conversion fails.
    """
    output_dir = output_dir or os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    # If it's already a .wav, still re-encode to a known-good 16kHz mono PCM WAV —
    # browsers/phones sometimes produce WAV containers ffmpeg-decodable but that
    # soundfile/wave still choke on (odd sample rates, float PCM, etc).
    out_path = os.path.join(output_dir, f"conv_{uuid.uuid4().hex[:10]}.wav")

    ffmpeg_bin = _find_ffmpeg()
    if not ffmpeg_bin:
        raise AudioConversionError(
            "ffmpeg is not available. Run 'pip install -r requirements.txt' so the "
            "bundled imageio-ffmpeg fallback is installed, or install ffmpeg system-wide "
            "(e.g. 'sudo apt install ffmpeg' on Linux, 'brew install ffmpeg' on macOS, "
            "or download a build for Windows and add it to PATH)."
        )

    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # sample rate
        "-sample_fmt", "s16",       # 16-bit PCM
        out_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AudioConversionError("Audio conversion timed out.")
    except FileNotFoundError:
        raise AudioConversionError("ffmpeg is not installed or not on PATH.")

    if result.returncode != 0 or not os.path.exists(out_path):
        stderr_tail = (result.stderr or "").strip().splitlines()[-1] if result.stderr else "unknown error"
        logger.error(f"ffmpeg conversion failed: {stderr_tail}")
        raise AudioConversionError(
            f"Could not process the uploaded audio file (it may be empty, corrupted, "
            f"or in an unsupported format). Details: {stderr_tail}"
        )

    return out_path
