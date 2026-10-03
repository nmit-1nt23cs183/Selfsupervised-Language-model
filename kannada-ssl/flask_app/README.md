# Kannada SSL — Self-Supervised Learning for Low-Resource Kannada Language Processing

A complete AIML research project implementing **Self-Supervised Learning (SSL)** for Kannada, a low-resource Indian language. The system trains wav2vec2 on Kannada speech and multilingual BERT on Kannada text, wrapped in a multilingual web interface.

## Key Feature

**Input text in ANY language → Translated to Kannada → Processed by Kannada SSL Model → Output in YOUR language**

## Functional Requirements

| # | Requirement | Description |
|---|---|---|
| FR-1 | Text-to-Text Translation | Type in any supported language, get output in any other supported language, routed through the Kannada SSL pipeline. |
| FR-2 | Speech-to-Text | Speak (mic) or upload a Kannada audio clip; transcribed using the project's fine-tuned wav2vec2 model. |
| FR-3 | **Speech-to-Speech Translation** | Speak (or upload audio) in any supported language and hear the translated result spoken back — full ASR → MT → TTS pipeline (`/speech-to-speech`, `/api/speech-to-speech`), plus an instant in-browser "Quick Mode" for live conversation. |
| FR-4 | **AI Chatbot** | A conversational chatbot (`/chatbot`, `/api/chatbot`) that answers questions grounded in the *same trained dataset* used by the project's own models (the Kannada topic-classification dataset and the Padakosha vocabulary) via TF-IDF retrieval, with an optional predicted-topic label from the trained text classifier. Works in any supported language. |
| FR-5 | Model Training Dashboard | Start/monitor/stop wav2vec2 and text-classifier training from the web UI or CLI. |
| FR-6 | Dataset & Model Info | Inspect dataset status and loaded model/checkpoint info via API. |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**System dependency for Speech-to-Speech:** the audio your browser records (WebM/Opus)
isn't readable directly by the ASR libraries, so the server converts it with **ffmpeg**
first. Install it separately (it's not a pip package):
```bash
# Linux
sudo apt install ffmpeg
# macOS
brew install ffmpeg
# Windows: download a build from https://ffmpeg.org/download.html and add it to PATH
```
Without ffmpeg on PATH, `/speech-to-speech` will return a clear "ffmpeg not installed" error instead of processing audio.

### 2. Place Datasets (see `data/README.md`)
```
data/dataset_kannada/     ← Extract WAV files here
data/KannadaKasturi_dataset_300_5label.xlsx
data/Kannada padakosha.xlsx
```
**Note:** the AI Chatbot indexes these same files. Until they're placed in `data/`,
`/chatbot` will correctly report "No dataset rows found to index" — that's expected,
not a bug.

### 3. Run the Web Application
```bash
python app.py
```
Open http://localhost:5000 in your browser.

### 4. Train Models (via web UI or CLI)

**wav2vec2 Speech Model:**
```bash
python train/train_wav2vec2.py --epochs 10 --batch-size 4 --lr 0.0001
```

**Text Classifier:**
```bash
python train/train_text_classifier.py --epochs 10 --batch-size 16 --lr 0.00002
```

---

## Project Structure

```
kannada-ssl-project/
├── app.py                          # Flask web application (main entry point)
├── config.py                       # All configuration settings
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── translator.py               # Multilingual pipeline (any lang ↔ Kannada)
│   ├── wav2vec2_model.py           # wav2vec2 wrapper for Kannada ASR
│   ├── text_classifier.py         # Kannada text classification model
│   ├── speech_to_speech.py         # NEW: ASR → Translate → TTS pipeline
│   └── chatbot.py                  # NEW: dataset-grounded AI chatbot (TF-IDF retrieval)
│
├── train/
│   ├── prepare_data.py             # Dataset loading and preprocessing
│   ├── train_wav2vec2.py           # wav2vec2 self-supervised pre-training
│   └── train_text_classifier.py   # BERT text classifier fine-tuning
│
├── utils/
│   ├── audio_utils.py              # Audio loading, resampling, feature extraction
│   ├── text_utils.py               # Kannada text preprocessing
│   ├── kannada_utils.py            # Dataset loading, vocabulary tools
│   ├── tts_utils.py                # NEW: text-to-speech synthesis (gTTS)
│   └── stt_utils.py                # NEW: generic speech-to-text for non-Kannada audio
│
├── templates/
│   ├── base.html                   # Base HTML template
│   ├── index.html                  # Home page with live demo
│   ├── speech_to_speech.html       # NEW: Speech-to-Speech page
│   ├── chatbot.html                # NEW: AI Chatbot page
│   └── about.html                  # Project description & architecture
│
├── static/
│   ├── css/style.css               # All styles
│   └── js/
│       ├── main.js                 # Translator page JavaScript
│       ├── speech_to_speech.js     # NEW: Speech-to-Speech page JavaScript
│       └── chatbot.js              # NEW: AI Chatbot page JavaScript
│
├── data/                           # Place datasets here (see data/README.md)
├── uploads/                        # Temporary audio uploads
└── checkpoints/                    # Saved model checkpoints
```

---

## Architecture

### Phase 1: Self-Supervised Pre-training (wav2vec2)
```
Kannada Audio → CNN Feature Encoder → Transformer Context Net → Contrastive Loss
```
- No transcriptions needed — learns from raw audio
- Learns phonetic representations of Kannada speech

### Phase 2: Downstream Fine-tuning
```
Labeled Kannada Data → Pre-trained Encoder → Task Head → Prediction
```
- ASR: CTC decoding for transcription
- Classification: 5-class Kannada text categorization

### Phase 3: Multilingual Pipeline
```
Any Language → Translate to Kannada → Kannada SSL Model → Translate to Output Language
```

### Phase 4: Speech-to-Speech Pipeline (NEW)
```
Source Audio → Speech-to-Text → Translate → Text-to-Speech → Target Audio
```
- Kannada source audio → the project's own fine-tuned wav2vec2 model
- Any other source language → generic ASR fallback (`utils/stt_utils.py`)
- Output audio synthesized with gTTS (`utils/tts_utils.py`)

### Phase 5: AI Chatbot (NEW)
```
User Message → Translate to Kannada → TF-IDF Retrieval over Trained Dataset
             → (optional) Text-Classifier Topic Prediction → Compose Answer → Translate Back
```
The chatbot never invents facts outside the dataset — if nothing in the trained data
is a close match, it says so honestly instead of hallucinating an answer.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/translate` | POST | Full multilingual pipeline |
| `/api/detect-language` | POST | Language detection |
| `/api/translate-to-kannada` | POST | Any language → Kannada |
| `/api/transcribe` | POST | Audio → Kannada transcription + translation |
| `/api/speech-to-speech` | POST | **NEW** — Audio in any language → translated speech audio out |
| `/api/audio/<filename>` | GET | **NEW** — Serve generated TTS audio |
| `/api/chatbot` | POST | **NEW** — Chat with the dataset-grounded AI chatbot |
| `/api/chatbot/reset` | POST | **NEW** — Reset the current chat session |
| `/api/chatbot/status` | GET | **NEW** — Chatbot index/session status |
| `/api/train/start` | POST | Start model training |
| `/api/train/status` | GET | Training progress |
| `/api/train/stop` | POST | Stop training |
| `/api/dataset/info` | GET | Dataset status |
| `/api/model/info` | GET | Model configuration |

---

## Supported Languages (Input/Output)

English, Kannada, Hindi, Telugu, Tamil, Malayalam, Marathi, Gujarati,
Bengali, Urdu, French, German, Spanish, Chinese, Arabic, Japanese,
Korean, Russian, Portuguese, Italian and more.

---

## Dependencies

- **PyTorch** — Deep learning framework
- **HuggingFace Transformers** — wav2vec2, BERT models
- **Flask** — Web framework
- **librosa** — Audio processing
- **deep-translator** — Multilingual translation
- **langdetect** — Language detection
- **pandas / openpyxl** — Dataset loading
- **scikit-learn** — ML utilities + TF-IDF retrieval for the AI chatbot
- **gTTS** — Text-to-speech synthesis (Speech-to-Speech output audio)
- **SpeechRecognition** — Generic speech-to-text for non-Kannada source audio (Speech-to-Speech input)

---

## Research Context

Kannada is a Dravidian language with ~44 million speakers, classified as a low-resource language for NLP due to limited labeled digital data. This project demonstrates that **self-supervised pre-training** on unlabeled Kannada audio/text can yield strong representations that transfer to downstream tasks with minimal labeled data — a critical finding for low-resource language AI.
