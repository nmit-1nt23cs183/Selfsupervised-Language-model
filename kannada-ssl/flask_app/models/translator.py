"""
Multilingual Translation Module
Handles translation between any language and Kannada using deep-translator.
"""

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import logging

import time
from functools import lru_cache

logger = logging.getLogger(__name__)


def _gtx_translate(text: str, target: str, source: str = "auto") -> str:
    """Fallback: Google's other public endpoint (no key). Uses only the standard library."""
    import json
    import urllib.parse
    import urllib.request
    url = ("https://translate.googleapis.com/translate_a/single?client=gtx"
           f"&sl={source}&tl={target}&dt=t&q={urllib.parse.quote(text)}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(part[0] for part in data[0] if part and part[0])


@lru_cache(maxsize=512)
def _google_translate(text: str, target: str, source: str = "auto") -> str:
    """Translate with deep-translator; on rate-limit/failure fall back to the gtx endpoint."""
    first_err = None
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        first_err = e
        logger.warning(f"deep-translator failed ({e}); trying fallback endpoint")
    try:
        return _gtx_translate(text, target, source)
    except Exception as e2:
        raise RuntimeError(f"deep-translator: {first_err} | fallback: {e2}")


def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        return lang
    except LangDetectException:
        logger.warning("Could not detect language, defaulting to English")
        return "en"


def translate_to_kannada(text: str, source_lang: str = "auto") -> dict:
    """
    Translate any text to Kannada.
    Returns dict with translated text and detected source language.
    """
    try:
        if source_lang == "auto":
            source_lang = detect_language(text)

        if source_lang == "kn":
            return {
                "translated_text": text,
                "source_language": "kn",
                "was_translated": False,
                "success": True,
            }

        translated = _google_translate(text, "kn")
        logger.info(f"Translated '{text[:50]}...' from {source_lang} to Kannada")
        return {
            "translated_text": translated,
            "source_language": source_lang,
            "was_translated": True,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Translation to Kannada failed: {e}")
        return {
            "translated_text": text,
            "source_language": source_lang,
            "was_translated": False,
            "success": False,
            "error": str(e),
        }


def translate_from_kannada(text: str, target_lang: str = "en") -> dict:
    """
    Translate Kannada text to the target language.
    Returns dict with translated text.
    """
    try:
        if target_lang == "kn":
            return {
                "translated_text": text,
                "target_language": "kn",
                "was_translated": False,
                "success": True,
            }

        translated = _google_translate(text, target_lang, "kn")
        logger.info(f"Translated from Kannada to {target_lang}")
        return {
            "translated_text": translated,
            "target_language": target_lang,
            "was_translated": True,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Translation from Kannada failed: {e}")
        return {
            "translated_text": text,
            "target_language": target_lang,
            "was_translated": False,
            "success": False,
            "error": str(e),
        }


def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> dict:
    """
    Generic any-language → any-language translation (not routed through Kannada).
    Used by the speech-to-speech pipeline and the chatbot, where we sometimes need
    a direct translation rather than the full Kannada-routed pipeline.
    """
    try:
        if source_lang == "auto":
            source_lang = detect_language(text)

        if source_lang == target_lang:
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "was_translated": False,
                "success": True,
            }

        translated = _google_translate(text, target_lang)
        return {
            "translated_text": translated,
            "source_language": source_lang,
            "target_language": target_lang,
            "was_translated": True,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Direct translation failed ({source_lang}->{target_lang}): {e}")
        return {
            "translated_text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "was_translated": False,
            "success": False,
            "error": str(e),
        }


def full_pipeline(
    input_text: str,
    target_language: str = "en",
    kannada_processor=None,
) -> dict:
    """
    Complete multilingual pipeline:
    1. Detect input language
    2. Translate to Kannada
    3. Process through Kannada SSL model
    4. Translate result to target language

    Args:
        input_text: Text in any language
        target_language: Desired output language code (e.g., 'en', 'hi', 'te')
        kannada_processor: Optional Kannada model for text processing

    Returns:
        dict with all intermediate steps and final output
    """
    result = {
        "input_text": input_text,
        "target_language": target_language,
        "steps": [],
        "success": False,
    }

    # Step 1: Detect language
    detected_lang = detect_language(input_text)
    result["detected_language"] = detected_lang
    result["steps"].append({
        "step": 1,
        "name": "Language Detection",
        "output": f"Detected: {detected_lang}",
    })

    if kannada_processor is None:
        # No Kannada model loaded: the Kannada hop adds nothing, and it doubles the Google
        # requests (which is what triggers "too many requests"). Translate directly.
        direct = translate_text(input_text, source_lang=detected_lang, target_lang=target_language)
        result["final_output"] = direct.get("translated_text", input_text)
        result["steps"].append({
            "step": 2,
            "name": f"Direct translation to {target_language.upper()}",
            "output": result["final_output"],
            "was_translated": direct.get("was_translated", False),
        })
        to_kannada = from_kannada = direct
    else:
        to_kannada = translate_to_kannada(input_text, source_lang=detected_lang)
        result["kannada_text"] = to_kannada.get("translated_text", input_text)
        result["steps"].append({
            "step": 2, "name": "Translation to Kannada",
            "output": result["kannada_text"],
            "was_translated": to_kannada.get("was_translated", False),
        })
        processed_kannada = result["kannada_text"]
        try:
            processed_result = kannada_processor.process(result["kannada_text"])
            processed_kannada = processed_result.get("output", result["kannada_text"])
            result["model_output"] = processed_kannada
            result["steps"].append({"step": 3, "name": "Kannada SSL Model Processing",
                                    "output": processed_kannada, "model_used": True})
        except Exception as e:
            logger.warning(f"Model processing failed, using translated text: {e}")
        from_kannada = translate_from_kannada(processed_kannada, target_lang=target_language)
        result["final_output"] = from_kannada.get("translated_text", processed_kannada)
        result["steps"].append({
            "step": 4, "name": f"Translation to {target_language.upper()}",
            "output": result["final_output"],
            "was_translated": from_kannada.get("was_translated", False),
        })

    failed = [r for r in (to_kannada, from_kannada) if not r.get("success", False)]
    if failed:
        result["success"] = False
        result["error"] = "Translation failed: " + failed[0].get("error", "unknown error")
        result["hint"] = ("Check internet access (Google Translate must be reachable) and that "
                          "deep-translator is installed: pip install deep-translator langdetect")
    else:
        result["success"] = True
    return result
