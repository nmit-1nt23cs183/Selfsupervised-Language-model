"""
Speech-to-Speech Translation Pipeline

Source audio
    ↓
Speech-to-Text
    ↓
Translation
    ↓
Text-to-Speech
    ↓
Target audio

This version uses the generic speech recognizer for all supported
source languages. The custom Kannada wav2vec2 model is not used here
until a properly fine-tuned Kannada checkpoint is available.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechToSpeechPipeline:
    """Connects Speech-to-Text, Translation and Text-to-Speech."""

    def __init__(
        self,
        wav2vec2_checkpoint: Optional[str] = None,
        tts_output_dir: Optional[str] = None
    ):
        self.wav2vec2_checkpoint = wav2vec2_checkpoint
        self.tts_output_dir = tts_output_dir

    def _transcribe(self, audio_path: str, source_lang: str) -> dict:
        """
        Convert speech to text.

        Uses the generic speech recognition system for all supported
        source languages.
        """

        try:
            from utils.stt_utils import transcribe_generic

            result = transcribe_generic(
                audio_path,
                lang=source_lang
            )

            return result

        except Exception as e:
            logger.exception("Speech recognition failed")
            return {
                "success": False,
                "error": str(e)
            }

    def translate_speech(
        self,
        audio_path: str,
        source_lang: str = "en",
        target_lang: str = "kn",
    ) -> dict:
        """
        Complete speech-to-speech pipeline.

        Audio
          → Speech-to-Text
          → Translation
          → Text-to-Speech

        Returns:
            Dictionary containing:
            - transcription
            - translated_text
            - output audio filename
            - processing steps
        """

        result = {
            "success": False,
            "source_language": source_lang,
            "target_language": target_lang,
            "steps": [],
        }

        # ---------------------------------------------------------
        # STEP 0: Convert uploaded audio to WAV
        # ---------------------------------------------------------

        try:
            from utils.audio_convert import (
                ensure_wav,
                AudioConversionError
            )

            wav_path = ensure_wav(
                audio_path,
                output_dir=self.tts_output_dir
            )

        except AudioConversionError as e:
            result["error"] = f"Audio conversion failed: {str(e)}"
            return result

        except Exception as e:
            result["error"] = f"Audio conversion failed: {str(e)}"
            return result

        # ---------------------------------------------------------
        # STEP 1: Speech-to-Text
        # ---------------------------------------------------------

        try:
            stt_result = self._transcribe(
                wav_path,
                source_lang
            )

        except Exception as e:
            stt_result = {
                "success": False,
                "error": str(e)
            }

        # Remove temporary WAV file after transcription
        if wav_path != audio_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        # Check STT result
        if not stt_result.get("success"):
            result["error"] = (
                "Speech-to-text failed: "
                + str(stt_result.get("error", "Unknown error"))
            )
            return result

        transcribed_text = stt_result.get(
            "transcription",
            ""
        ).strip()

        if not transcribed_text:
            result["error"] = (
                "Speech-to-text returned empty text. "
                "Please speak clearly and try again."
            )
            return result

        result["transcribed_text"] = transcribed_text

        result["steps"].append({
            "step": 1,
            "name": "Speech-to-Text",
            "output": transcribed_text
        })

        # ---------------------------------------------------------
        # STEP 2: Translation
        # ---------------------------------------------------------

        try:
            from models.translator import translate_text

            translation = translate_text(
                transcribed_text,
                source_lang=source_lang,
                target_lang=target_lang
            )

        except Exception as e:
            result["error"] = (
                f"Translation failed: {str(e)}"
            )
            return result

        translated_text = translation.get(
            "translated_text",
            ""
        ).strip()

        if not translated_text:
            translated_text = transcribed_text

        result["translated_text"] = translated_text

        result["steps"].append({
            "step": 2,
            "name": "Translation",
            "output": translated_text
        })

        # ---------------------------------------------------------
        # STEP 3: Text-to-Speech
        # ---------------------------------------------------------

        try:
            from utils.tts_utils import synthesize_speech

            tts_result = synthesize_speech(
                translated_text,
                lang=target_lang,
                output_dir=self.tts_output_dir
            )

        except Exception as e:
            result["error"] = (
                f"Text-to-speech failed: {str(e)}"
            )
            return result

        if not tts_result.get("success"):
            result["error"] = (
                "Text-to-speech failed: "
                + str(tts_result.get("error", "Unknown error"))
            )
            return result

        result["audio_filename"] = tts_result.get(
            "filename"
        )

        result["steps"].append({
            "step": 3,
            "name": "Text-to-Speech",
            "output": tts_result.get("filename")
        })

        # Include warning if TTS returned one
        if tts_result.get("warning"):
            result["warning"] = tts_result["warning"]

        # ---------------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------------

        result["success"] = True

        return result