# Data Directory

Place your Kannada dataset files here before training.

## Note: bundled sample data

This project now ships with a small **sample dataset** in `data/sample_data/`
(75 Kannada sentences across the 5 categories + 50 Padakosha entries). The
chatbot and dataset-info page automatically use it whenever your real files
below aren't present, so the app works out of the box. Once you add your real
files here, they always take priority — just restart the app.

## Required Files

### 1. Audio Dataset (for wav2vec2 training)
```
data/dataset_kannada/
    D1_1.wav
    D1_2.wav
    ...
    D1_142.wav
```
Extract `dataset_kannada_1.zip` into this folder.

### 2. Text Classification Dataset
```
data/KannadaKasturi_dataset_300_5label.xlsx
```
300 Kannada text samples with 5 category labels:
- Politics
- Sports
- Entertainment
- Technology
- Business

### 3. Kannada Padakosha (Vocabulary)
```
data/Kannada padakosha.xlsx
```
Kannada dictionary/vocabulary entries.

### 4. Large Documents Dataset (Optional)
```
data/Kannada_Documents_Dataset - Individual_label.xlsx
data/Kannada_Documents_For_Classification.xlsx
```
Large-scale Kannada document corpus for extended training.

## After Placing Files

1. Check dataset status on the **Training** page
2. Start training from the **Training** page
3. Monitor progress and download checkpoints

## Directory Structure After Setup
```
data/
├── dataset_kannada/           ← Extract your WAV files here
│   ├── D1_1.wav
│   ├── D1_2.wav
│   └── ...
├── Kannada padakosha.xlsx
├── KannadaKasturi_dataset_300_5label.xlsx
├── Kannada_Documents_Dataset - Individual_label.xlsx  (optional)
└── Kannada_Documents_For_Classification.xlsx          (optional)
```
