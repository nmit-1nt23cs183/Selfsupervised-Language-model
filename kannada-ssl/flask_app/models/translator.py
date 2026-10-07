"""
Multilingual Translation Module
Handles translation between any language and Kannada using deep-translator.
"""

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import logging

logger = logging.getLogger(__name__)


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

        translator = GoogleTranslator(source=source_lang, target="kn")
        translated = translator.translate(text)
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

        translator = GoogleTranslator(source="kn", target=target_lang)
        translated = translator.translate(text)
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

        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
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

    # Step 2: Translate to Kannada
    to_kannada = translate_to_kannada(input_text, source_lang=detected_lang)
    result["kannada_text"] = to_kannada.get("translated_text", input_text)
    result["steps"].append({
        "step": 2,
        "name": "Translation to Kannada",
        "output": result["kannada_text"],
        "was_translated": to_kannada.get("was_translated", False),
    })

    # Step 3: Process through Kannada SSL model (if available)
    processed_kannada = result["kannada_text"]
    if kannada_processor is not None:
        try:
            processed_result = kannada_processor.process(result["kannada_text"])
            processed_kannada = processed_result.get("output", result["kannada_text"])
            result["model_output"] = processed_kannada
            result["steps"].append({
                "step": 3,
                "name": "Kannada SSL Model Processing",
                "output": processed_kannada,
                "model_used": True,
            })
        except Exception as e:
            logger.warning(f"Model processing failed, using translated text: {e}")
            processed_kannada = result["kannada_text"]
            result["steps"].append({
                "step": 3,
                "name": "Kannada SSL Model Processing",
                "output": f"Passthrough (model not ready): {processed_kannada}",
                "model_used": False,
            })
    else:
        result["steps"].append({
            "step": 3,
            "name": "Kannada SSL Model Processing",
            "output": f"Model not loaded — using translated text: {processed_kannada}",
            "model_used": False,
        })

    # Step 4: Translate from Kannada to target language
    from_kannada = translate_from_kannada(processed_kannada, target_lang=target_language)
    result["final_output"] = from_kannada.get("translated_text", processed_kannada)
    result["steps"].append({
        "step": 4,
        "name": f"Translation to {target_language.upper()}",
        "output": result["final_output"],
        "was_translated": from_kannada.get("was_translated", False),
    })

    result["success"] = True
    return result
