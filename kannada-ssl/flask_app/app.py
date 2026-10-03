"""
Kannada SSL - Flask Web Application
Self-Supervised Learning for Low Resource Kannada Language Processing

Run:  python app.py
Then: open http://localhost:5000
"""

import os
import json
import logging
import threading

import uuid

from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Optional CORS — only needed if calling the API from a different origin
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass  # Not required for local use

# ── Global training state ────────────────────────────────────────────────────
_training_status = {
    "is_training": False,
    "current_task": None,
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "accuracy": None,
    "message": "Idle",
    "history": [],
}

# ── Global chatbot instance (built lazily on first request, reused after) ──
_chatbot = None
_chatbot_lock = threading.Lock()


def _get_chatbot():
    """Return the shared chatbot instance, building its dataset index on first use."""
    global _chatbot
    with _chatbot_lock:
        if _chatbot is None:
            from models.chatbot import KannadaChatbot
            _chatbot = KannadaChatbot(Config)
    return _chatbot


# ── Helper: safe translator import ──────────────────────────────────────────

def _get_translator():
    """Import translator lazily — only when a request actually needs it."""
    try:
        from models.translator import (
            full_pipeline,
            translate_to_kannada,
            translate_from_kannada,
            detect_language,
        )
        return full_pipeline, translate_to_kannada, translate_from_kannada, detect_language
    except ImportError as e:
        return None, None, None, None


def _missing_package_response(pkg):
    return jsonify({
        "success": False,
        "error": f"Required package not installed: {pkg}",
        "hint": f"Run:  pip install {pkg}",
    }), 503


# ── Page routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", languages=Config.SUPPORTED_LANGUAGES)


@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/inference")
def inference_page():
    return render_template("inference.html", languages=Config.SUPPORTED_LANGUAGES)


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/speech-to-speech")
def speech_to_speech_page():
    return render_template("speech_to_speech.html", languages=Config.SUPPORTED_LANGUAGES)


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html", languages=Config.SUPPORTED_LANGUAGES)


# ── API: Translation pipeline ────────────────────────────────────────────────

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """Any language → Kannada model → target language."""
    data = request.get_json(silent=True) or {}
    input_text = data.get("text", "").strip()
    target_language = data.get("target_language", "en")

    if not input_text:
        return jsonify({"success": False, "error": "No text provided"}), 400
    if len(input_text) > 5000:
        return jsonify({"success": False, "error": "Text too long (max 5000 chars)"}), 400

    full_pipeline, _, _, _ = _get_translator()
    if full_pipeline is None:
        return _missing_package_response("deep-translator langdetect")

    try:
        result = full_pipeline(input_text, target_language=target_language)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Translation pipeline error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/detect-language", methods=["POST"])
def api_detect_language():
    """Detect the language of input text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    _, _, _, detect_language = _get_translator()
    if detect_language is None:
        return _missing_package_response("langdetect")

    lang = detect_language(text)
    lang_name = Config.SUPPORTED_LANGUAGES.get(lang, lang.upper())
    return jsonify({"success": True, "language_code": lang, "language_name": lang_name})


@app.route("/api/translate-to-kannada", methods=["POST"])
def api_translate_to_kannada():
    """Translate any text to Kannada."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    source_lang = data.get("source_language", "auto")

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    _, translate_to_kannada, _, _ = _get_translator()
    if translate_to_kannada is None:
        return _missing_package_response("deep-translator langdetect")

    result = translate_to_kannada(text, source_lang=source_lang)
    return jsonify(result)


# ── API: Audio transcription ─────────────────────────────────────────────────

@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Upload a WAV file → Kannada transcription → translated output."""
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    target_language = request.form.get("target_language", "en")

    if audio_file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"success": False, "error": f"Unsupported audio format: {ext}"}), 400

    try:
        filename = secure_filename(audio_file.filename)
        upload_path = os.path.join(Config.UPLOAD_DIR, filename)
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        audio_file.save(upload_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"File save failed: {e}"}), 500

    try:
        from models.wav2vec2_model import KannadaWav2Vec2Model
        from utils.audio_utils import load_audio
    except ImportError as e:
        return _missing_package_response(str(e))

    try:
        model = KannadaWav2Vec2Model(checkpoint_path=Config.WAV2VEC2_KANNADA_CHECKPOINT)
        if not model.load():
            return jsonify({
                "success": False,
                "error": "Model not ready. Train the wav2vec2 model first.",
                "hint": "Go to the Training page and run wav2vec2 training.",
            }), 503

        audio_array, sr = load_audio(upload_path)
        transcription_result = model.transcribe(audio_array, sample_rate=sr)

        if not transcription_result["success"]:
            return jsonify(transcription_result), 500

        kannada_text = transcription_result["transcription"]
        _, _, translate_from_kannada, _ = _get_translator()
        translation_text = kannada_text
        if translate_from_kannada:
            t = translate_from_kannada(kannada_text, target_lang=target_language)
            translation_text = t.get("translated_text", kannada_text)

        return jsonify({
            "success": True,
            "kannada_transcription": kannada_text,
            "translation": translation_text,
            "target_language": target_language,
        })
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── API: Speech-to-Speech ────────────────────────────────────────────────────

@app.route("/api/speech-to-speech", methods=["POST"])
def api_speech_to_speech():
    """Upload audio in any supported language → translated speech audio out."""
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    source_language = request.form.get("source_language", "en")
    target_language = request.form.get("target_language", "kn")

    if audio_file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"success": False, "error": f"Unsupported audio format: {ext}"}), 400

    try:
        filename = secure_filename(f"{uuid.uuid4().hex[:10]}_{audio_file.filename}")
        upload_path = os.path.join(Config.UPLOAD_DIR, filename)
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        audio_file.save(upload_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"File save failed: {e}"}), 500

    try:
        from models.speech_to_speech import SpeechToSpeechPipeline

        pipeline = SpeechToSpeechPipeline(
            wav2vec2_checkpoint=Config.WAV2VEC2_KANNADA_CHECKPOINT,
            tts_output_dir=Config.TTS_OUTPUT_DIR,
        )
        result = pipeline.translate_speech(
            upload_path, source_lang=source_language, target_lang=target_language
        )
        if not result.get("success"):
            return jsonify(result), 500

        result["audio_url"] = f"/api/audio/{result['audio_filename']}"
        return jsonify(result)
    except Exception as e:
        logger.error(f"Speech-to-speech error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            os.remove(upload_path)
        except OSError:
            pass


@app.route("/api/audio/<path:filename>", methods=["GET"])
def api_get_audio(filename):
    """Serve a generated TTS audio file."""
    safe_name = secure_filename(filename)
    return send_from_directory(Config.TTS_OUTPUT_DIR, safe_name, mimetype="audio/mpeg")


# ── API: AI Chatbot ───────────────────────────────────────────────────────────

@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    """Chat with the dataset-grounded Kannada AI chatbot."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    target_language = data.get("target_language", "kn")
    source_language = data.get("source_language", "auto")

    if not message:
        return jsonify({"success": False, "error": "No message provided"}), 400
    if len(message) > 2000:
        return jsonify({"success": False, "error": "Message too long (max 2000 chars)"}), 400

    if "chat_session_id" not in session:
        session["chat_session_id"] = uuid.uuid4().hex
    session_id = session["chat_session_id"]

    try:
        chatbot = _get_chatbot()
        result = chatbot.respond(message, target_language=target_language,
                                 session_id=session_id, source_language=source_language)
        if not result.get("success"):
            return jsonify(result), 503
        return jsonify(result)
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chatbot/reset", methods=["POST"])
def api_chatbot_reset():
    session_id = session.get("chat_session_id")
    if session_id:
        _get_chatbot().reset(session_id)
    return jsonify({"success": True})


@app.route("/api/chatbot/status", methods=["GET"])
def api_chatbot_status():
    return jsonify(_get_chatbot().status())


# ── API: Training ─────────────────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def api_start_training():
    """Start model training in a background thread."""
    global _training_status

    if _training_status["is_training"]:
        return jsonify({"success": False, "error": "Training already in progress"}), 400

    data = request.get_json(silent=True) or {}
    task = data.get("task", "wav2vec2")
    num_epochs = int(data.get("epochs", 10))
    batch_size = int(data.get("batch_size", 4))
    lr = float(data.get("learning_rate", 1e-4))

    _training_status.update({
        "is_training": True,
        "current_task": task,
        "epoch": 0,
        "total_epochs": num_epochs,
        "loss": None,
        "accuracy": None,
        "message": f"Starting {task} training...",
        "history": [],
    })

    def run_training():
        global _training_status
        try:
            config = Config()

            if task == "wav2vec2":
                from train.train_wav2vec2 import train_wav2vec2, Wav2Vec2TrainingArgs

                def cb(epoch, loss):
                    _training_status.update({
                        "epoch": epoch,
                        "loss": round(loss, 4),
                        "message": f"Epoch {epoch}/{num_epochs} — Loss: {loss:.4f}",
                    })
                    _training_status["history"].append({"epoch": epoch, "loss": round(loss, 4)})

                args = Wav2Vec2TrainingArgs(
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    learning_rate=lr,
                    output_dir=Config.WAV2VEC2_KANNADA_CHECKPOINT,
                )
                result = train_wav2vec2(args, config, progress_callback=cb)

            elif task == "classifier":
                from train.train_text_classifier import train_text_classifier

                def cb(epoch, loss, acc):
                    _training_status.update({
                        "epoch": epoch,
                        "loss": round(loss, 4),
                        "accuracy": round(acc, 4),
                        "message": f"Epoch {epoch}/{num_epochs} — Loss: {loss:.4f} | Acc: {acc:.4f}",
                    })
                    _training_status["history"].append({
                        "epoch": epoch,
                        "loss": round(loss, 4),
                        "accuracy": round(acc, 4),
                    })

                result = train_text_classifier(
                    config,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    output_dir=Config.TEXT_CLASSIFIER_CHECKPOINT,
                    progress_callback=cb,
                )
            else:
                result = {"success": False, "error": f"Unknown task: {task}"}

            _training_status["message"] = (
                "Training complete!" if result.get("success")
                else f"Error: {result.get('error', 'Unknown error')}"
            )

        except Exception as e:
            logger.error(f"Training thread error: {e}")
            _training_status["message"] = f"Training error: {e}"
        finally:
            _training_status["is_training"] = False

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"success": True, "message": f"Training started: {task}"})


@app.route("/api/train/status", methods=["GET"])
def api_training_status():
    return jsonify(_training_status)


@app.route("/api/train/stop", methods=["POST"])
def api_stop_training():
    global _training_status
    if not _training_status["is_training"]:
        return jsonify({"success": False, "error": "No training in progress"})
    _training_status["is_training"] = False
    _training_status["message"] = "Training stopped by user."
    return jsonify({"success": True})


# ── API: Dataset info ─────────────────────────────────────────────────────────

@app.route("/api/dataset/info", methods=["GET"])
def api_dataset_info():
    config = Config()
    info = {}

    # Count audio files without importing librosa
    audio_dir = config.AUDIO_DATASET_DIR
    if os.path.exists(audio_dir):
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    else:
        wav_files = []

    info["audio"] = {
        "directory": audio_dir,
        "exists": os.path.exists(audio_dir),
        "num_files": len(wav_files),
        "sample_files": sorted(wav_files)[:5],
    }

    info["classification_dataset"] = {
        "file": os.path.basename(config.CLASSIFICATION_DATASET),
        "exists": os.path.exists(config.CLASSIFICATION_DATASET),
        "num_samples": 0,
        "columns": [],
    }
    info["padakosha"] = {
        "file": "Kannada padakosha.xlsx",
        "exists": os.path.exists(config.PADAKOSHA_FILE),
        "num_entries": 0,
    }

    # If the real dataset files aren't present yet, the chatbot falls back to a
    # small bundled sample dataset (see data/sample_data/) so the app still works.
    info["using_sample_data"] = not (
        os.path.exists(config.CLASSIFICATION_DATASET) or os.path.exists(config.PADAKOSHA_FILE)
    )

    # Try to load Excel stats only if pandas/openpyxl available
    try:
        import pandas as pd
        if os.path.exists(config.CLASSIFICATION_DATASET):
            df = pd.read_excel(config.CLASSIFICATION_DATASET)
            info["classification_dataset"]["num_samples"] = len(df)
            info["classification_dataset"]["columns"] = list(df.columns)
        elif os.path.exists(config.SAMPLE_CLASSIFICATION_DATASET):
            df = pd.read_excel(config.SAMPLE_CLASSIFICATION_DATASET)
            info["classification_dataset"]["num_samples"] = len(df)
            info["classification_dataset"]["columns"] = list(df.columns)
            info["classification_dataset"]["sample_fallback"] = True
        if os.path.exists(config.PADAKOSHA_FILE):
            pad = pd.read_excel(config.PADAKOSHA_FILE)
            info["padakosha"]["num_entries"] = len(pad)
        elif os.path.exists(config.SAMPLE_PADAKOSHA_FILE):
            pad = pd.read_excel(config.SAMPLE_PADAKOSHA_FILE)
            info["padakosha"]["num_entries"] = len(pad)
            info["padakosha"]["sample_fallback"] = True
    except Exception:
        pass

    info["models"] = {
        "wav2vec2_checkpoint": os.path.exists(config.WAV2VEC2_KANNADA_CHECKPOINT),
        "text_classifier_checkpoint": os.path.exists(config.TEXT_CLASSIFIER_CHECKPOINT),
    }

    return jsonify(info)


# ── API: Model info ───────────────────────────────────────────────────────────

@app.route("/api/model/info", methods=["GET"])
def api_model_info():
    config = Config()
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    return jsonify({
        "wav2vec2": {
            "base_model": config.WAV2VEC2_BASE_MODEL,
            "checkpoint": config.WAV2VEC2_KANNADA_CHECKPOINT,
            "has_checkpoint": os.path.exists(config.WAV2VEC2_KANNADA_CHECKPOINT),
            "sample_rate": config.AUDIO_SAMPLE_RATE,
        },
        "text_classifier": {
            "base_model": "bert-base-multilingual-cased",
            "checkpoint": config.TEXT_CLASSIFIER_CHECKPOINT,
            "has_checkpoint": os.path.exists(config.TEXT_CLASSIFIER_CHECKPOINT),
        },
        "translation": {
            "engine": "Google Translate (deep-translator)",
            "supported_languages": Config.SUPPORTED_LANGUAGES,
        },
        "speech_to_speech": {
            "asr_kannada": "Fine-tuned wav2vec2 (falls back to generic ASR if untrained)",
            "asr_other_languages": "SpeechRecognition (Google Web Speech API)",
            "tts_engine": "gTTS",
        },
        "chatbot": {
            "grounding": "TF-IDF retrieval over the project's own trained datasets",
            "status": _get_chatbot().status() if _chatbot is not None else {"is_loaded": False},
        },
        "device": device,
    })


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "project": "Kannada SSL"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.TTS_OUTPUT_DIR, exist_ok=True)
    port = int(os.environ.get("PORT", "5000"))
    print("\n" + "=" * 50)
    print("  Kannada SSL — Web Application")
    print(f"  http://localhost:{port}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=port, debug=Config.DEBUG)
