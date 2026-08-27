"""
Text Processing Utilities for Kannada NLP
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Kannada Unicode range: \u0C80-\u0CFF
KANNADA_UNICODE_RANGE = re.compile(r'[\u0C80-\u0CFF]')


def is_kannada_text(text: str) -> bool:
    """Check if text contains Kannada characters."""
    return bool(KANNADA_UNICODE_RANGE.search(text))


def clean_text(text: str) -> str:
    """Basic text cleaning — remove extra whitespace, special chars."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0C80-\u0CFF.,!?;:\'\"-]', '', text)
    return text


def tokenize_kannada(text: str) -> List[str]:
    """Simple whitespace tokenizer for Kannada text."""
    return text.strip().split()


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return re.sub(r'[^\w\s\u0C80-\u0CFF]', '', text)


def count_kannada_chars(text: str) -> int:
    """Count the number of Kannada characters in text."""
    return len(KANNADA_UNICODE_RANGE.findall(text))


def get_language_ratio(text: str) -> dict:
    """Estimate proportion of Kannada vs. Latin characters."""
    kannada_count = count_kannada_chars(text)
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    total = len(text.replace(' ', ''))
    return {
        "kannada_ratio": kannada_count / total if total else 0,
        "latin_ratio": latin_count / total if total else 0,
        "total_chars": total,
        "kannada_chars": kannada_count,
        "latin_chars": latin_count,
    }


def truncate_text(text: str, max_chars: int = 512) -> str:
    """Truncate text to maximum character limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '...'


def split_into_sentences(text: str) -> List[str]:
    """Split Kannada/English text into sentences."""
    sentences = re.split(r'[।.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]
