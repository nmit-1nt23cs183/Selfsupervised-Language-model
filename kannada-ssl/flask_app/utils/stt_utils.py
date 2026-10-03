"""
Generic Speech-to-Text Utility (non-Kannada languages)

The project's own wav2vec2 model (models/wav2vec2_model.py) is trained specifically
for Kannada ASR. For the Speech-to-Speech feature, the *source* audio can be in any
of the app's supported languages, so we fall back to the `SpeechRecognition` package
(Google Web Speech API) for anything that isn't Kannada, or when no fine-tuned
Kannada checkpoint has been trained yet.
"""

import logging

logger = logging.getLogger(__name__)

# SpeechRecognition's Google Web Speech API expects BCP-47 style codes.
_SR_LANG_MAP = {
    "en": "en-IN", "kn": "kn-IN", "hi": "hi-IN", "te": "te-IN", "ta": "ta-IN",
    "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "bn": "bn-IN", "ur": "ur-IN",
    "fr": "fr-FR", "de": "de-DE", "es": "es-ES", "zh-CN": "zh-CN",
    "ar": "ar-SA", "ja": "ja-JP", "ko": "ko-KR", "ru": "ru-RU",
    "pt": "pt-PT", "it": "it-IT",
}


def transcribe_generic(audio_path: str, lang: str = "en") -> dict:
    """
    Transcribe an audio file using the SpeechRecognition package.

    Args:
        audio_path: Path to a WAV file (mono PCM works best)
        lang: Our internal language code (e.g. 'en', 'hi')

    Returns:
        dict with success flag, transcription text, and any error
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return {
            "success": False,
            "error": "SpeechRecognition not installed",
            "hint": "Run:  pip install SpeechRecognition",
        }

    recognizer = sr.Recognizer()
    sr_lang = _SR_LANG_MAP.get(lang, "en-IN")

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=sr_lang)
        logger.info(f"Generic STT ({sr_lang}): transcribed {len(text)} chars")
        return {"success": True, "transcription": text, "language": lang}
    except sr.UnknownValueError:
        return {"success": False, "error": "Could not understand audio (silence or unclear speech)"}
    except sr.RequestError as e:
        return {"success": False, "error": f"Speech recognition service error: {e}"}
    except Exception as e:
        logger.error(f"Generic STT failed: {e}")
        return {"success": False, "error": str(e)}
