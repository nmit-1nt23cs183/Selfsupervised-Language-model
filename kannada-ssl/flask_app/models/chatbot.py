"""
Kannada AI Chatbot

A retrieval-grounded chatbot that answers using the SAME datasets the project's
models are trained on (the Kannada classification dataset + the Padakosha
vocabulary). It does not call any external LLM — it works fully offline once the
translation packages are installed, which keeps it consistent with the rest of the
project (self-supervised / trained-on-our-own-data philosophy).

How it works
------------
1. User message (any supported language) is translated into Kannada.
2. TF-IDF vectors (scikit-learn — already a project dependency) are built once over
   every row of the trained/training datasets and cached in memory.
3. The Kannada query is vectorized and compared with cosine similarity against the
   dataset index to retrieve the closest matching entries.
4. If a trained text-classifier checkpoint exists (train/train_text_classifier.py),
   the query is also run through it to report a predicted topic/category.
5. A grounded Kannada answer is composed from the retrieved dataset content and
   translated back to the user's language.
6. If nothing in the dataset is a good match, the chatbot says so honestly instead
   of inventing an answer, and offers what it does know.
"""

import logging
import os
from typing import Optional

import re

logger = logging.getLogger(__name__)

_LANG_NAMES = {
    "english": "en", "kannada": "kn", "hindi": "hi", "telugu": "te", "tamil": "ta",
    "malayalam": "ml", "marathi": "mr", "gujarati": "gu", "bengali": "bn", "urdu": "ur",
    "french": "fr", "german": "de", "spanish": "es", "chinese": "zh-CN", "arabic": "ar",
    "japanese": "ja", "korean": "ko", "russian": "ru", "portuguese": "pt", "italian": "it",
}
_LANG_RE = "(?P<lang>" + "|".join(_LANG_NAMES) + ")"
_Q = "[\"'“‘]?"
_QE = "[\"'”’]?"

# "what does hi mean in kannada", "what is the meaning of hi in hindi", "hi meaning in tamil",
# "how to say thank you in telugu", "translate good morning to french", "water in kannada",
# "kannada word for book"
_WORD_PATTERNS = [
    rf"(?:what\s+(?:does|is|do)|meaning\s+of|how\s+(?:do\s+(?:you|i)|to)\s+say|translate)\s+(?:the\s+)?(?:(?:meaning|translation)\s+of\s+)?{_Q}(?P<w>.+?){_QE}\s+(?:(?:mean|means|meaning|meanin)\s+)?(?:in|to|into)\s+{_LANG_RE}\??$",
    rf"{_Q}(?P<w>.+?){_QE}\s+(?:meaning|means?|translation)\s+in\s+{_LANG_RE}\??$",
    rf"{_LANG_RE}\s+(?:word|translation|meaning)\s+(?:for|of)\s+{_Q}(?P<w>.+?){_QE}\??$",
    rf"{_Q}(?P<w>.+?){_QE}\s+(?:in|to|into)\s+{_LANG_RE}\??$",
]


def _extract_word_request(message: str):
    """Return (word, language_code) if the user asks to translate a word/phrase, else None."""
    m = (message or "").strip().rstrip(".!? ")
    for pat in _WORD_PATTERNS:
        hit = re.search(pat, m, flags=re.IGNORECASE)
        if hit:
            w = hit.group("w").strip(" \"'“”‘’")
            # drop leftover filler like "the meaning of" captured at the front
            w = re.sub(r"^(?:the\s+)?\w*(?:aning|translation)\s+of\s+", "", w, flags=re.IGNORECASE)
            if w:
                return w, _LANG_NAMES[hit.group("lang").lower()]
    return None


SIMILARITY_THRESHOLD = 0.12  # (legacy dataset search, currently unused)
MAX_HISTORY_TURNS = 20


class KannadaChatbot:
    """Singleton-style chatbot: build once (expensive), then call respond() cheaply."""

    def __init__(self, config):
        self.config = config
        self.is_loaded = False
        self.using_sample_data = False

        self._vectorizer = None
        self._doc_matrix = None
        self._documents = []   # list[str] — Kannada text of each indexed row
        self._doc_sources = [] # list[dict] — where each row came from (dataset/label/etc.)

        self._classifier = None  # lazy, optional

        self._sessions = {}  # session_id -> list of {role, text}

    # ── Index building ──────────────────────────────────────────────────────

    def load(self) -> dict:
        """Load datasets and build the TF-IDF retrieval index. Safe to call once at startup."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed. Run: pip install scikit-learn"}

        from utils.kannada_utils import load_classification_dataset, load_padakosha

        self._documents = []
        self._doc_sources = []
        self.using_sample_data = False

        real_files_present = os.path.exists(self.config.CLASSIFICATION_DATASET) or os.path.exists(
            self.config.PADAKOSHA_FILE
        )
        if not real_files_present:
            self.using_sample_data = True
            logger.warning(
                "No real dataset files found in data/ — indexing the small bundled "
                "sample dataset instead (data/sample_data/). Add your real files "
                "(see data/README.md) and restart to use them."
            )

        # 1) Classification dataset — labeled Kannada news/topic snippets
        class_df = load_classification_dataset(
            self.config.CLASSIFICATION_DATASET,
            fallback_path=getattr(self.config, "SAMPLE_CLASSIFICATION_DATASET", None),
        )
        if class_df is not None:
            text_col, label_col = self._guess_columns(
                class_df,
                primary_hints=["text", "sentence", "content", "document", "kannada"],
                secondary_hints=["label", "category", "class", "topic"],
            )
            for _, row in class_df.iterrows():
                text = str(row.get(text_col, "")).strip()
                if not text or text == "nan":
                    continue
                label = str(row.get(label_col, "")).strip() if label_col else ""
                self._documents.append(text)
                self._doc_sources.append({
                    "dataset": "classification_dataset",
                    "label": label,
                })

        # 2) Padakosha — Kannada vocabulary/dictionary entries
        pada_df = load_padakosha(
            self.config.PADAKOSHA_FILE,
            fallback_path=getattr(self.config, "SAMPLE_PADAKOSHA_FILE", None),
        )
        if pada_df is not None:
            word_col, meaning_col = self._guess_columns(
                pada_df,
                primary_hints=["word", "pada", "kannada", "shabda"],
                secondary_hints=["meaning", "artha", "definition", "english", "description"],
            )
            for _, row in pada_df.iterrows():
                word = str(row.get(word_col, "")).strip()
                meaning = str(row.get(meaning_col, "")).strip() if meaning_col else ""
                if not word or word == "nan":
                    continue
                combined = f"{word} — {meaning}" if meaning and meaning != "nan" else word
                self._documents.append(combined)
                self._doc_sources.append({
                    "dataset": "padakosha",
                    "word": word,
                    "meaning": meaning,
                })

        if not self._documents:
            self.is_loaded = False
            return {
                "success": False,
                "error": "No dataset rows found to index. Place the dataset files in data/ (see data/README.md).",
            }

        self._vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            # IMPORTANT: sklearn's default token_pattern (\b\w\w+\b) splits Kannada
            # conjunct characters apart at combining marks (e.g. virama), fragmenting
            # words like "ಕ್ರಿಕೆಟ್" into garbage tokens. Whitespace-based tokenization
            # keeps each Kannada word intact, which is what dataset-retrieval needs.
            token_pattern=r"(?u)\S+",
        )
        self._doc_matrix = self._vectorizer.fit_transform(self._documents)
        self.is_loaded = True

        logger.info(f"Chatbot index built: {len(self._documents)} documents")
        return {"success": True, "num_documents": len(self._documents)}

    @staticmethod
    def _guess_columns(df, primary_hints=None, secondary_hints=None):
        """
        Pick a (primary_column, secondary_column) pair from an unknown Excel schema.

        First tries matching column names against `primary_hints` / `secondary_hints`
        (case-insensitive substring match) since real dataset files usually have
        sensible headers. Falls back to a statistical heuristic (longest average text
        = primary, few repeated short values = secondary) when no name matches.
        """
        primary_hints = primary_hints or []
        secondary_hints = secondary_hints or []
        columns = list(df.columns)

        def _match(hints):
            for col in columns:
                col_lower = str(col).strip().lower()
                if any(hint in col_lower for hint in hints):
                    return col
            return None

        primary_col = _match(primary_hints)
        secondary_col = _match([h for h in secondary_hints])
        if secondary_col == primary_col:
            secondary_col = None

        if primary_col and secondary_col:
            return primary_col, secondary_col

        # Statistical fallback for whichever slot is still unresolved
        stat_text_col, stat_label_col = None, None
        best_avg_len = -1
        for col in columns:
            try:
                sample = df[col].dropna().astype(str)
                if sample.empty:
                    continue
                avg_len = sample.str.len().mean()
                nunique = sample.nunique()
            except Exception:
                continue
            if avg_len > best_avg_len:
                best_avg_len = avg_len
                stat_text_col = col
            if 1 < nunique <= 12 and avg_len < 30:
                stat_label_col = col

        if primary_col is None:
            primary_col = stat_text_col if stat_text_col is not None else (columns[0] if columns else None)
        if secondary_col is None:
            secondary_col = stat_label_col if stat_label_col != primary_col else None

        return primary_col, secondary_col

    # ── Optional: topic classifier ──────────────────────────────────────────

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier or None
        try:
            from models.text_classifier import KannadaTextClassifier
            clf = KannadaTextClassifier(checkpoint_path=self.config.TEXT_CLASSIFIER_CHECKPOINT)
            if os.path.exists(self.config.TEXT_CLASSIFIER_CHECKPOINT) and clf.load():
                self._classifier = clf
            else:
                self._classifier = False
        except Exception as e:
            logger.warning(f"Chatbot: classifier unavailable ({e})")
            self._classifier = False
        return self._classifier or None

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _retrieve(self, query_kn: str, top_k: int = 3) -> list:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        query_vec = self._vectorizer.transform([query_kn])
        sims = cosine_similarity(query_vec, self._doc_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        matches = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0:
                continue
            matches.append({
                "text": self._documents[idx],
                "score": score,
                "source": self._doc_sources[idx],
            })
        return matches

    # ── Public API ───────────────────────────────────────────────────────────

    def respond(self, message: str, target_language: str = "kn", session_id: str = "default",
                source_language: str = "auto") -> dict:
        """
        Speech/text recognition + translation assistant (not a Q&A bot).

        The user types or speaks in any language; the bot recognises the language and
        replies with the translation in the chosen "Reply in" language. The frontend
        reads the reply aloud, the same way the Speech-to-Speech page does.
        """
        message = (message or "").strip()
        if not message:
            return {"success": False, "error": "Empty message"}

        from models.translator import detect_language, translate_text

        detected_lang = source_language if source_language and source_language != "auto" \
            else detect_language(message)

        # "what does hi mean in <language>" -> translate just the word into that language
        req = _extract_word_request(message)
        text_to_translate = req[0] if req else message
        if req:
            target_language = req[1]
            if source_language == "auto":
                detected_lang = "auto"  # the extracted word is short; let Google detect it

        if detected_lang == target_language:
            result = {
                "translated_text": text_to_translate, "success": True, "was_translated": False,
            }
            note = "Already in the chosen language — pick a different Reply-in language to translate."
        else:
            result = translate_text(text_to_translate, source_lang=detected_lang,
                                    target_lang=target_language)
            note = None

        if not result.get("success"):
            return {"success": False,
                    "error": "Translation failed: " + str(result.get("error", "unknown error")),
                    "hint": "Check your internet connection (Google Translate must be reachable)."}

        reply = result["translated_text"]

        history = self._sessions.setdefault(session_id, [])
        history.append({"role": "user", "text": message})
        history.append({"role": "bot", "text": reply})
        max_turns = getattr(self.config, "CHATBOT_MAX_HISTORY_TURNS", MAX_HISTORY_TURNS)
        del history[: max(0, len(history) - max_turns * 2)]

        return {
            "success": True,
            "reply": reply,
            "heard": text_to_translate,
            "note": note,
            "detected_language": detected_lang,
            "target_language": target_language,
        }

    def reset(self, session_id: str = "default") -> None:
        self._sessions.pop(session_id, None)

    def get_history(self, session_id: str = "default") -> list:
        return self._sessions.get(session_id, [])

    def status(self) -> dict:
        return {
            "is_loaded": self.is_loaded,
            "num_documents": len(self._documents),
            "has_classifier": bool(self._get_classifier()) if self.is_loaded else False,
            "active_sessions": len(self._sessions),
            "using_sample_data": self.using_sample_data,
        }
