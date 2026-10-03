"""
Kannada Text Classifier
Multi-class text classification for Kannada documents using fine-tuned transformers.
"""

import os
import logging
import torch
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)

# Default Kannada BERT model
KANNADA_BERT_MODEL = "l3cube-pune/kannada-bert"
FALLBACK_MODEL = "bert-base-multilingual-cased"


class KannadaTextClassifier:
    """
    Text classifier for Kannada language using BERT-based models.
    Supports fine-tuning on custom Kannada datasets.
    """

    LABEL_NAMES = [
        "Politics",
        "Sports",
        "Entertainment",
        "Technology",
        "Business",
    ]

    def __init__(
        self,
        num_labels: int = 5,
        checkpoint_path: Optional[str] = None,
        base_model: str = FALLBACK_MODEL,
    ):
        self.num_labels = num_labels
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.label_names = self.LABEL_NAMES[:num_labels]

    def load(self) -> bool:
        """Load the text classification model."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model_path = None
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                model_path = self.checkpoint_path
                logger.info(f"Loading fine-tuned classifier from {model_path}")
            else:
                model_path = self.base_model
                logger.info(f"Loading base model: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("Text classifier loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load text classifier: {e}")
            self.is_loaded = False
            return False

    def classify(self, text: str) -> dict:
        """
        Classify Kannada text into categories.

        Args:
            text: Kannada text to classify

        Returns:
            dict with predicted label, confidence, and all class scores
        """
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(probs))
            predicted_label = self.label_names[predicted_idx] if predicted_idx < len(self.label_names) else f"Class {predicted_idx}"

            all_scores = {
                self.label_names[i] if i < len(self.label_names) else f"Class {i}": float(probs[i])
                for i in range(len(probs))
            }

            return {
                "success": True,
                "predicted_label": predicted_label,
                "confidence": float(probs[predicted_idx]),
                "all_scores": all_scores,
                "text_length": len(text),
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {"success": False, "error": str(e)}

    def batch_classify(self, texts: List[str]) -> List[dict]:
        """Classify a batch of Kannada texts."""
        return [self.classify(text) for text in texts]

    def get_model_info(self) -> dict:
        return {
            "model_type": "BERT-based Text Classifier",
            "task": "Kannada Text Classification",
            "num_labels": self.num_labels,
            "label_names": self.label_names,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "base_model": self.base_model,
        }
