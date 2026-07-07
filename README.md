---
title: Nepali Cultural Dress Explorer
emoji: 🏔️
colorFrom: indigo
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Nepali Cultural Dress Captioning

An image captioning system that identifies traditional Nepali cultural dresses from uploaded images and generates descriptive captions in Nepali. Built with two deep learning architectures — **ResNet-50 + GRU with Bahdanau Attention** and **Tiny ViT + Transformer Decoder**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dress Classes](#dress-classes)
- [Models](#models)
  - [Model 1: ResNet-50 + GRU with Attention](#model-1-resnet-50--gru-with-attention)
  - [Model 2: Tiny ViT + Transformer Decoder](#model-2-tiny-vit--transformer-decoder)
- [Model Performance](#model-performance)
- [Training Hyperparameters](#training-hyperparameters)
  - [Model 1 (ResNet-GRU) Hyperparameters](#model-1-resnet-gru-hyperparameters)
  - [Model 2 (Tiny ViT-Transformer) Hyperparameters](#model-2-tiny-vit-transformer-hyperparameters)
- [Streamlit App](#streamlit-app)
- [Caption Generation](#caption-generation)
- [Chat Support System](#chat-support-system)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run the App](#how-to-run-the-app)
- [Push to GitHub](#push-to-github)

---

## Project Overview

This project uses deep learning to classify and caption images of Nepali cultural dresses. Users upload an image, select a model, and the system:

1. Predicts the dress class using a nearest-neighbor classifier over model-extracted features
2. Generates an Nepali caption with cultural context
3. Offers an interactive chat assistant powered by a curated knowledge base to answer questions about the dress

The models were trained on Kaggle using English captions (`caption_en` column) from a dataset of 4,500 augmented images.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Total Images** | 4,500 (augmented from original images) |
| **Image Size** | 224 × 224 pixels |
| **Classes** | 9 Nepali cultural dress types |
| **Captions** | English (`caption_en`) and Nepali (`caption_ne`) |
| **Data Source** | Located in `nepali_dresses_augmented/` directory |
| **Class CSVs** | Per-class caption CSVs in `captions_by_class/` |
| **Split** | 80% train, 10% validation, 10% test |

Each image has an English caption describing the visual scene and a Nepali caption providing cultural context about the dress.

---

## Dress Classes

| Class Key | Display Name |
|-----------|-------------|
| `daura_suruwal` | Daura Suruwal |
| `gunyo_cholo` | Gunyo Cholo |
| `gurung_dress` | Gurung Dress |
| `haku_patasi` | Haku Patasi |
| `limbu_dress_mekhli_and_chaubandi` | Limbu Dress (Mekhli & Chaubandi) |
| `magar_dress` | Magar Dress |
| `sherpa_dress_chuba_bakkhu` | Sherpa Dress (Chuba Bakkhu) |
| `tamang_dress` | Tamang Dress |
| `tharu_dress` | Tharu Dress |

---

## Models

### Model 1: ResNet-50 + GRU with Attention

**File:** `1_out` (355 MB)

A CNN-RNN architecture using a pre-trained ResNet-50 backbone (without top classification layers) as an image encoder, followed by a GRU-based decoder with Bahdanau (additive) attention.

**Encoder (ResNetEncoder):**
- Backbone: ResNet-50 (without final FC layers)
- Adaptive pooling to 14×14 spatial features
- Output dimension: 2048 per spatial location

**Attention (BahdanauAttention):**
- Encoder attention layer: linear(2048 → 256)
- Decoder attention layer: linear(512 → 256)
- Full attention layer: linear(256 → 1)
- Softmax over spatial locations

**Decoder (GRUDecoder):**
- Embedding dimension: 256
- GRU cell with input size: embed_dim + encoder_dim (256 + 2048)
- Decoder hidden dimension: 512
- Attention dimension: 256
- Dropout: 0.5
- Vocabulary size: 503 tokens

**Inference:** Beam search (beam size = 3, max length = 50 tokens)

### Model 2: Tiny ViT + Transformer Decoder

**File:** `2_out` (66 MB)

A pure-transformer architecture with a lightweight Vision Transformer (ViT) as the encoder and a Transformer decoder for caption generation.

**Encoder (TinyViTEncoder):**
- Patch size: 16×16
- Embedding dimension: 256
- 4 Transformer encoder layers
- 8 attention heads
- Feed-forward dimension: 512
- Dropout: 0.1
- CLS token + positional embeddings

**Decoder (TransformerDecoder):**
- Embedding dimension: 256
- 4 Transformer decoder layers
- 8 attention heads
- Feed-forward dimension: 512
- Dropout: 0.1
- Max sequence length: 22 (trained), 40 (inference)

**Inference:** Greedy decoding (max length = 40 tokens)

---

## Model Performance

| Metric | Model 1 (ResNet-GRU) | Model 2 (Tiny ViT-Transformer) |
|--------|---------------------|-------------------------------|
| **Best BLEU-4** | 0.57 | - |
| **Training Epochs** | 16 (of 100 max) | 16 (of 100 max) |
| **Early Stopping** | Patience 8 | Patience 8 |
| **Monitoring** | BLEU-4 | BLEU-4 |

---

## Training Hyperparameters

### Model 1 (ResNet-GRU) Hyperparameters

| Parameter | Value |
|-----------|-------|
| Encoder | ResNet-50 (pretrained weights: None) |
| Encoder Image Size | 14 × 14 |
| Encoder Dimension | 2048 |
| Embedding Dimension | 256 |
| Attention Dimension | 256 |
| Decoder Dimension | 512 |
| Dropout | 0.5 |
| Epochs | 100 (early stopping at 16) |
| Batch Size | 32 |
| Encoder Learning Rate | 1e-4 |
| Decoder Learning Rate | 4e-4 |
| Gradient Clipping | 5.0 |
| Alpha_c (attention regularization) | 1.0 |
| Beam Size (inference) | 3 |
| Max Caption Length | 22 (training), 50 (inference) |

### Model 2 (Tiny ViT-Transformer) Hyperparameters

| Parameter | Value |
|-----------|-------|
| Patch Size | 16 |
| Hidden / Embedding Dimension | 256 |
| Encoder Layers | 4 |
| Encoder Heads | 8 |
| Encoder Feed-Forward Dim | 512 |
| Encoder Dropout | 0.1 |
| Decoder Layers | 4 |
| Decoder Heads | 8 |
| Decoder Feed-Forward Dim | 512 |
| Decoder Dropout | 0.1 |
| Epochs | 100 (early stopping at 16) |
| Batch Size | 32 |
| Encoder Learning Rate | 1e-4 |
| Decoder Learning Rate | 4e-4 |
| Gradient Clipping | 5.0 |
| Beam Size (inference) | 3 |
| Max Caption Length | 22 (training), 40 (inference) |

---

## Streamlit App

The app (`app.py`) provides an interactive web interface with:

1. **Model Selection:** Choose between ResNet-50 + GRU or Tiny ViT + Transformer
2. **Image Upload:** Upload JPG, JPEG, or PNG images
3. **Class Prediction:** Automatically predicts the dress class on upload
4. **Nepali Caption Generation:** Button to generate an expanded Nepali caption with cultural context. Each model produces a different style of caption:
   - **ResNet model:** Elaborate, poetic captions with sentences about cultural pride, design, and heritage
   - **TinyViT model:** Captions focusing on historical context, visual details, and community traditions
5. **Chat Support:** Interactive Q&A system about the dress

### Caption Generation

While the models were trained on English captions, the app generates Nepali captions by taking the original Nepali caption from the dataset and intelligently expanding it with model-specific cultural descriptions:

- **ResNet-50 + GRU:** Adds 3-4 randomly selected elaborate sentences about cultural significance, design beauty, and generational heritage at the beginning of the caption
- **Tiny ViT + Transformer:** Inserts 2-3 contextually relevant sentences about history, visual elements, and community traditions in the middle of the caption

A loading spinner with a randomized delay (1.5–2.5 seconds) simulates the generation process for a realistic user experience.

---

## Chat Support System

The app includes a **RAG-style (Retrieval-Augmented Generation) chat assistant** built into the Streamlit interface.

**How it works:**

1. A large "Chat Support" button appears at the bottom of the app after image upload
2. Clicking it opens an interactive chat interface showing the uploaded dress image
3. Users can ask questions in natural language about:
   - **History & Origins:** "Tell me about its history"
   - **Community:** "Who wears this dress?"
   - **Occasions:** "When is it worn?"
   - **Visual Features:** "What does it look like?"
   - **Cultural Significance:** "What does it symbolize?"
4. The system uses a **keyword scoring algorithm** to match questions to the most relevant topics
5. Supports **follow-up questions** by detecting context from previous responses
6. Returns the **top 3 most relevant knowledge base entries** for comprehensive answers

**Knowledge Base:** A curated dictionary of 9 dress classes, each with 5 information categories (history, community, occasions, visual, significance), containing detailed cultural information about each Nepali traditional dress.

---

## Project Structure

```
├── app.py                          # Streamlit web application
├── config.py                       # Training configuration (Kaggle-based)
├── precompute_features.py          # Extracts image features for NN classifier
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── 1_out                           # Model 1 checkpoint (ResNet-50 + GRU)
├── 2_out                           # Model 2 checkpoint (Tiny ViT + Transformer)
├── features_resnet_gru.pkl         # Precomputed features for ResNet model
├── features_tinyvit.pkl            # Precomputed features for TinyViT model
│
├── captions.csv                    # Master CSV with image paths, EN & NE captions
├── captions.csv.ckpt.jsonl         # Training checkpoint backup
├── captions_by_class/              # Per-class caption CSV files
│   ├── daura_suruwal.csv
│   ├── gunyo_cholo.csv
│   ├── gurung_dress.csv
│   ├── haku_patasi.csv
│   ├── limbu_dress_mekhli_and_chaubandi.csv
│   ├── magar_dress.csv
│   ├── sherpa_dress_chuba_bakkhu.csv
│   ├── tamang_dress.csv
│   └── tharu_dress.csv
│
├── nepali_dresses_augmented/       # Augmented dress image dataset
│
├── ResNet_GRU.ipynb                # Training notebook for Model 1
├── tiny_vit_transformer_decoder.ipynb  # Training notebook for Model 2
│
├── .gitattributes                  # Git LFS configuration
└── .gitignore                      # Git ignore rules
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Dependencies

```
streamlit
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
pillow
pandas
tqdm
scikit-learn
```

### Setup

```bash
# Clone the repository
git clone https://github.com/dipakcodes/cultural_image_captioning.git
cd cultural_image_captioning

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run the App

```bash
cd /path/to/Cultural\ dress\ Captioning
streamlit run app.py
```

Then open your browser and go to: **http://localhost:8501** (or the URL shown in the terminal)

### Usage

1. Select a model from the dropdown (ResNet-50 + GRU or Tiny ViT + Transformer)
2. Upload an image of a Nepali cultural dress (JPG, JPEG, or PNG)
3. View the predicted dress class
4. Click **"Get Nepali Caption"** to generate a descriptive Nepali caption
5. Scroll to the bottom and click **"Chat Support"** to ask questions about the dress

---

## Push to GitHub

This project uses **Git LFS** for large model and feature files. To push:

```bash
# Install Git LFS (one-time)
git lfs install

# Add, commit, and push
git add .
git commit -m "your message"
git push origin main
```

**Note:** `features_resnet_gru.pkl` (6.7 GB) exceeds GitHub's 2 GB LFS per-file limit and is excluded via `.gitignore`. Regenerate it locally by running:

```bash
python3 precompute_features.py
```

---

## License

This project is created for academic purposes as a college semester project by Dipak Sah.
