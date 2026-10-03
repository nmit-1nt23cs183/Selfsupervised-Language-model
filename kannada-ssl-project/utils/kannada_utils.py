"""
Kannada-specific Utilities
Handles vocabulary, dataset loading, and Kannada text processing.
"""

import os
import logging
import pandas as pd
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def load_padakosha(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load the Kannada Padakosha (vocabulary dictionary).

    Args:
        file_path: Path to 'Kannada padakosha.xlsx'

    Returns:
        DataFrame with vocabulary entries, or None if not found
    """
    if not os.path.exists(file_path):
        logger.warning(f"Padakosha file not found: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded Padakosha: {len(df)} entries, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load Padakosha: {e}")
        return None


def load_classification_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load the Kannada text classification dataset.

    Args:
        file_path: Path to 'KannadaKasturi_dataset_300_5label.xlsx'

    Returns:
        DataFrame with text and label columns
    """
    if not os.path.exists(file_path):
        logger.warning(f"Classification dataset not found: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded classification dataset: {len(df)} samples, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load classification dataset: {e}")
        return None


def load_large_documents_dataset(file_path: str, nrows: int = 1000) -> Optional[pd.DataFrame]:
    """
    Load the large Kannada documents dataset (Individual label version).
    Loads in chunks to avoid memory issues.

    Args:
        file_path: Path to 'Kannada_Documents_Dataset - Individual_label.xlsx'
        nrows: Number of rows to load (None = all)

    Returns:
        DataFrame with document text and labels
    """
    if not os.path.exists(file_path):
        logger.warning(f"Documents dataset not found: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path, nrows=nrows)
        logger.info(f"Loaded documents dataset: {len(df)} samples (nrows={nrows})")
        return df
    except Exception as e:
        logger.error(f"Failed to load documents dataset: {e}")
        return None


def get_dataset_stats(df: pd.DataFrame, label_column: Optional[str] = None) -> dict:
    """Get statistics about a dataset."""
    stats = {
        "total_samples": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
    }

    if label_column and label_column in df.columns:
        label_dist = df[label_column].value_counts().to_dict()
        stats["label_distribution"] = label_dist
        stats["num_classes"] = len(label_dist)

    return stats


def prepare_train_val_split(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset into train and validation sets.

    Args:
        df: Input DataFrame
        text_column: Column name for text
        label_column: Column name for labels
        val_size: Fraction for validation (default 0.2)
        random_state: Random seed

    Returns:
        Dict with 'train' and 'val' DataFrames
    """
    from sklearn.model_selection import train_test_split

    df_clean = df[[text_column, label_column]].dropna()
    train_df, val_df = train_test_split(
        df_clean,
        test_size=val_size,
        random_state=random_state,
        stratify=df_clean[label_column] if df_clean[label_column].nunique() > 1 else None,
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    return {"train": train_df, "val": val_df}


def build_vocabulary(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """
    Build a simple vocabulary from a list of texts.

    Args:
        texts: List of Kannada text strings
        min_freq: Minimum frequency to include a word

    Returns:
        Dictionary mapping word to index
    """
    from collections import Counter
    all_words = []
    for text in texts:
        all_words.extend(text.strip().split())

    word_freq = Counter(all_words)
    vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
    for word, freq in word_freq.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    logger.info(f"Built vocabulary: {len(vocab)} tokens")
    return vocab
