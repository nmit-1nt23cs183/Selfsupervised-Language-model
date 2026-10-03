"""
wav2vec2 Model for Kannada Speech Processing
Self-supervised learning model wrapper for Kannada ASR.
"""

import os
import logging
import torch
import numpy as np
from typing import Optional, Union

logger = logging.getLogger(__name__)


class KannadaWav2Vec2Model:
    """
    Wrapper around Hugging Face wav2vec2 for Kannada speech recognition.
    Supports both pre-trained and fine-tuned Kannada checkpoints.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.model = None
        self.processor = None
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        logger.info(f"KannadaWav2Vec2Model initialized. Device: {self.device}")

    def load(self, force_base: bool = False) -> bool:
        """Load the wav2vec2 model."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            # Try to load fine-tuned checkpoint first, fall back to base model
            model_path = None
            if not force_base and self.checkpoint_path and os.path.exists(self.checkpoint_path):
                model_path = self.checkpoint_path
                logger.info(f"Loading fine-tuned Kannada model from {model_path}")
            else:
                model_path = "facebook/wav2vec2-base"
                logger.info(f"Loading base wav2vec2 model: {model_path}")

            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("wav2vec2 model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load wav2vec2 model: {e}")
            self.is_loaded = False
            return False

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Transcribe audio to Kannada text.

        Args:
            audio_array: Audio as numpy array (mono, float32)
            sample_rate: Sample rate (model expects 16000 Hz)

        Returns:
            dict with transcription and confidence
        """
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}

        try:
            import librosa

            # Resample if needed
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

            # Process through model
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return {
                "success": True,
                "transcription": transcription,
                "language": "kn",
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"success": False, "error": str(e)}

    def extract_features(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Extract self-supervised features from audio.
        Returns hidden states from wav2vec2 encoder.
        """
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}

        try:
            from transformers import Wav2Vec2Model as HFWav2Vec2Model
            import librosa

            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.wav2vec2(**inputs)
                hidden_states = outputs.last_hidden_state

            return {
                "success": True,
                "features": hidden_states.cpu().numpy(),
                "feature_dim": hidden_states.shape[-1],
                "sequence_length": hidden_states.shape[1],
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {"success": False, "error": str(e)}

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_type": "wav2vec2",
            "task": "Kannada ASR (Self-Supervised Learning)",
            "device": self.device,
            "is_loaded": self.is_loaded,
            "checkpoint": self.checkpoint_path or "facebook/wav2vec2-base",
            "framework": "Hugging Face Transformers",
        }
