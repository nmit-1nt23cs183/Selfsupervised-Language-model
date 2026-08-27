# Kannada SSL — Self-Supervised Learning for Low-Resource Kannada Language Processing

A complete AIML research project implementing **Self-Supervised Learning (SSL)** for Kannada, a low-resource Indian language. The system trains wav2vec2 on Kannada speech and multilingual BERT on Kannada text, wrapped in a multilingual web interface.

## Key Feature

**Input text in ANY language → Translated to Kannada → Processed by Kannada SSL Model → Output in YOUR language**

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Datasets (see `data/README.md`)
```
data/dataset_kannada/     ← Extract WAV files here
data/KannadaKasturi_dataset_300_5label.xlsx
data/Kannada padakosha.xlsx
```

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
│   └── text_classifier.py         # Kannada text classification model
│
├── train/
│   ├── prepare_data.py             # Dataset loading and preprocessing
│   ├── train_wav2vec2.py           # wav2vec2 self-supervised pre-training
│   └── train_text_classifier.py   # BERT text classifier fine-tuning
│
├── utils/
│   ├── audio_utils.py              # Audio loading, resampling, feature extraction
│   ├── text_utils.py               # Kannada text preprocessing
│   └── kannada_utils.py            # Dataset loading, vocabulary tools
│
├── templates/
│   ├── base.html                   # Base HTML template
│   ├── index.html                  # Home page with live demo
│   ├── train.html                  # Training dashboard
│   ├── inference.html              # Inference demo (text + audio)
│   └── about.html                  # Project description & architecture
│
├── static/
│   ├── css/style.css               # All styles
│   └── js/main.js                  # Frontend JavaScript
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

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/translate` | POST | Full multilingual pipeline |
| `/api/detect-language` | POST | Language detection |
| `/api/translate-to-kannada` | POST | Any language → Kannada |
| `/api/transcribe` | POST | Audio → Kannada transcription + translation |
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
- **scikit-learn** — ML utilities

---

## Research Context

Kannada is a Dravidian language with ~44 million speakers, classified as a low-resource language for NLP due to limited labeled digital data. This project demonstrates that **self-supervised pre-training** on unlabeled Kannada audio/text can yield strong representations that transfer to downstream tasks with minimal labeled data — a critical finding for low-resource language AI.
