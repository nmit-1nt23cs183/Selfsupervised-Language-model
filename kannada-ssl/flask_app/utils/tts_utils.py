"""
Text-to-Speech Utilities
Synthesizes speech audio from text for any of the app's supported languages.
Used by the Speech-to-Speech pipeline and the "Listen" playback fallback.
"""

import os
import uuid
import logging

logger = logging.getLogger(__name__)

# gTTS language codes match Config.SUPPORTED_LANGUAGES almost exactly.
# A few need remapping to what gTTS/Google expects.
_GTTS_LANG_OVERRIDES = {
    "zh-CN": "zh-CN",
}


def _resolve_gtts_lang(lang_code: str) -> str:
    return _GTTS_LANG_OVERRIDES.get(lang_code, lang_code)


def synthesize_speech(text: str, lang: str = "en", output_dir: str = None) -> dict:
    """
    Convert text to a speech audio file (MP3) using gTTS.

    Args:
        text: Text to synthesize
        lang: Language code (e.g. 'kn', 'en', 'hi') — must match Config.SUPPORTED_LANGUAGES
        output_dir: Directory to save the generated MP3 file into

    Returns:
        dict with success flag, file path/name, and any error
    """
    if not text or not text.strip():
        return {"success": False, "error": "No text provided for speech synthesis"}

    try:
        from gtts import gTTS
    except ImportError:
        return {
            "success": False,
            "error": "gTTS not installed",
            "hint": "Run:  pip install gTTS",
        }

    output_dir = output_dir or os.path.join(os.getcwd(), "uploads", "tts")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"tts_{uuid.uuid4().hex[:12]}.mp3"
    filepath = os.path.join(output_dir, filename)

    try:
        gtts_lang = _resolve_gtts_lang(lang)
        tts = gTTS(text=text, lang=gtts_lang)
        tts.save(filepath)
        logger.info(f"Synthesized speech ({lang}, {len(text)} chars) -> {filename}")
        return {
            "success": True,
            "filename": filename,
            "filepath": filepath,
            "language": lang,
        }
    except Exception as e:
        logger.error(f"Speech synthesis failed for lang={lang}: {e}")
        # gTTS doesn't support every language/dialect equally well — fall back to English
        # so the pipeline still returns *some* audio instead of hard failing.
        if lang != "en":
            try:
                tts = gTTS(text=text, lang="en")
                tts.save(filepath)
                return {
                    "success": True,
                    "filename": filename,
                    "filepath": filepath,
                    "language": "en",
                    "warning": f"'{lang}' not supported by TTS engine, fell back to English",
                }
            except Exception as e2:
                return {"success": False, "error": str(e2)}
        return {"success": False, "error": str(e)}


def cleanup_old_audio(output_dir: str, max_age_seconds: int = 3600) -> int:
    """Delete generated TTS files older than max_age_seconds. Returns number removed."""
    import time

    if not os.path.exists(output_dir):
        return 0

    removed = 0
    now = time.time()
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        try:
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
                os.remove(fpath)
                removed += 1
        except OSError:
            continue
    return removed
