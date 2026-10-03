"""
Data Preparation for Kannada SSL Training
Loads and preprocesses Kannada audio + text datasets for model training.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.audio_utils import load_audio, normalize_audio, trim_silence, list_kannada_audio_files
from utils.kannada_utils import (
    load_padakosha,
    load_classification_dataset,
    load_large_documents_dataset,
    prepare_train_val_split,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KannadaAudioDataset:
    """Dataset class for Kannada audio files (for wav2vec2 training)."""

    def __init__(self, audio_dir: str, sample_rate: int = 16000, max_duration: float = 10.0):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.file_list = list_kannada_audio_files(audio_dir)
        logger.info(f"KannadaAudioDataset: {len(self.file_list)} files")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_list[idx]
        try:
            audio, sr = load_audio(file_path, target_sr=self.sample_rate)
            audio = normalize_audio(audio)
            audio = trim_silence(audio, sr=sr)
            # Clip/pad to max duration
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            return {
                "input_values": audio,
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "duration": len(audio) / self.sample_rate,
            }
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {
                "input_values": np.zeros(self.max_samples, dtype=np.float32),
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "duration": 0.0,
                "error": str(e),
            }

    def get_stats(self) -> dict:
        """Compute dataset statistics."""
        durations = []
        errors = 0
        for i in range(min(len(self), 50)):  # Sample first 50 for quick stats
            item = self[i]
            if "error" not in item:
                durations.append(item["duration"])
            else:
                errors += 1

        return {
            "total_files": len(self.file_list),
            "sampled": len(durations),
            "errors": errors,
            "avg_duration_sec": float(np.mean(durations)) if durations else 0.0,
            "total_duration_min": float(sum(durations) / 60) if durations else 0.0,
        }


class KannadaTextDataset:
    """Dataset class for Kannada text classification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        label_column: str,
        tokenizer=None,
        max_length: int = 512,
    ):
        self.df = dataframe.dropna(subset=[text_column, label_column]).reset_index(drop=True)
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Encode labels
        unique_labels = sorted(self.df[label_column].unique())
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        logger.info(f"KannadaTextDataset: {len(self.df)} samples, {len(self.label2id)} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.df.iloc[idx][self.text_column])
        label = self.df.iloc[idx][self.label_column]
        label_id = self.label2id.get(label, 0)

        item = {"text": text, "label": label_id, "label_name": label}

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoding["input_ids"].squeeze()
            item["attention_mask"] = encoding["attention_mask"].squeeze()

        return item


def prepare_audio_data(config: Config) -> Optional[KannadaAudioDataset]:
    """Prepare Kannada audio dataset for wav2vec2 training."""
    audio_dir = config.AUDIO_DATASET_DIR
    if not os.path.exists(audio_dir):
        logger.warning(f"Audio dataset directory not found: {audio_dir}")
        logger.info("Please place your Kannada WAV files in: data/dataset_kannada/")
        return None

    dataset = KannadaAudioDataset(audio_dir, sample_rate=config.AUDIO_SAMPLE_RATE)
    stats = dataset.get_stats()
    logger.info(f"Audio dataset stats: {stats}")
    return dataset


def prepare_text_data(config: Config) -> Optional[Dict[str, KannadaTextDataset]]:
    """Prepare Kannada text classification dataset."""
    df = load_classification_dataset(config.CLASSIFICATION_DATASET)
    if df is None:
        logger.warning("Classification dataset not found. Skipping text data prep.")
        return None

    # Auto-detect text and label columns
    text_col = None
    label_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["text", "content", "sentence", "document"]):
            text_col = col
        if any(k in col_lower for k in ["label", "category", "class", "tag"]):
            label_col = col

    if not text_col or not label_col:
        # Assume first column is text, last is label
        text_col = df.columns[0]
        label_col = df.columns[-1]
        logger.warning(f"Auto-detected: text='{text_col}', label='{label_col}'")

    splits = prepare_train_val_split(df, text_col, label_col)

    datasets = {
        "train": KannadaTextDataset(splits["train"], text_col, label_col),
        "val": KannadaTextDataset(splits["val"], text_col, label_col),
        "full": KannadaTextDataset(df, text_col, label_col),
    }

    return datasets


if __name__ == "__main__":
    config = Config()
    logger.info("Preparing Kannada datasets...")
    audio_ds = prepare_audio_data(config)
    text_ds = prepare_text_data(config)

    if audio_ds:
        logger.info(f"Audio dataset ready: {len(audio_ds)} samples")
    if text_ds:
        logger.info(f"Text train: {len(text_ds['train'])}, Val: {len(text_ds['val'])}")
    logger.info("Data preparation complete.")
