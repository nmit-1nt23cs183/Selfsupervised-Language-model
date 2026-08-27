import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "kannada-ssl-secret-2024")
    DEBUG = os.environ.get("DEBUG", "True") == "True"

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    # Dataset files (place your Excel/WAV files here)
    PADAKOSHA_FILE = os.path.join(DATA_DIR, "Kannada padakosha.xlsx")
    CLASSIFICATION_DATASET = os.path.join(DATA_DIR, "KannadaKasturi_dataset_300_5label.xlsx")
    AUDIO_DATASET_DIR = os.path.join(DATA_DIR, "dataset_kannada")

    # Model config
    WAV2VEC2_BASE_MODEL = "facebook/wav2vec2-base"
    WAV2VEC2_KANNADA_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "wav2vec2_kannada")
    TEXT_CLASSIFIER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "text_classifier")

    # Training config
    AUDIO_SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 10  # seconds
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4

    # Supported languages for translation
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "kn": "Kannada",
        "hi": "Hindi",
        "te": "Telugu",
        "ta": "Tamil",
        "ml": "Malayalam",
        "mr": "Marathi",
        "gu": "Gujarati",
        "bn": "Bengali",
        "ur": "Urdu",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "zh-CN": "Chinese",
        "ar": "Arabic",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "pt": "Portuguese",
        "it": "Italian",
    }

    # Max file upload size (16 MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
