# 🎙️ AI Audio Story Generator with Background Sounds

This project is an AI-powered audio storytelling tool built with Streamlit. It takes a simple prompt and turns it into an expressive, narrated audio story — using the user's voice, injecting emotional tones, and adding intelligent background sounds based on the story content.

The goal is to create immersive and personalized storytelling experiences, ideal for children’s stories, interactive learning, or creative content generation.

---

## Features

- **Story Generation:** Uses Google Gemini 1.5 Flash to generate stories from a simple prompt. Supports multilingual story creation and optional personalization like child’s name or favorite animal.
- **Voice Cloning:** Clones the user’s voice using XTTSv2 from a short uploaded sample and narrates the story in that voice.
- **Emotion Detection:** Analyzes each sentence to predict emotional tone (happy, sad, angry, neutral) using a Transformer model and modulates the narration accordingly.
- **Background Sound Matching:** Uses the CLAP (Contrastive Language-Audio Pretraining) model to embed story context and match each line with suitable background sound effects.
- **Custom Parameters:** Users can adjust similarity threshold for sound matching, background sound volume, and pause duration between sentences.
- **Evaluation Script:** A separate evaluation script is provided to test the accuracy of the emotion detection system using test data.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
git clone https://github.com/Janvi-M/AudioStoryGen.git
cd AudioStoryGen

python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Download NLTK tokenizer data

```python
import nltk
nltk.download('punkt')
```

---

## API Key Setup

You need a Google Gemini API key. Get it from [Google AI Studio](https://aistudio.google.com/app/apikey), then create a `.env` file in the root directory:

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

Use the provided `.env.example` as a reference.

---

### Model and Data Setup

Some files are needed before running the app:

- CLAP Checkpoint:
The model file (630k-audioset-best.pt) will be auto-downloaded when the CLAP model runs for the first time (requires internet).
- trimmed_sounds/ Folder:
This folder contains background sound .wav files. It will be included in the repo.
- audio_embeddings.pt:
This file contains audio embeddings generated from trimmed_sounds/ using the CLAP model.
If not provided, you'll need to generate it separately and place it in the root directory.

---

## Running the Application

Once everything is set up, run the app using Streamlit:

```bash
streamlit run final-transformer_language.py
```

It will open in your default web browser.

---

## Emotion Detection Evaluation

To evaluate the performance of the emotion detection model:

1. Ensure `emotion_test_data.csv` is in the same directory. It should contain columns like `sentence` and `true_emotion`.

2. Run the script:

```bash
python emotion_detection.py
```

The script will:
- Load the emotion classifier from `final-transformer_language.py` (aliased as `app1`)
- Use the `detect_emotion` function to predict emotions
- Compare predictions with the true labels
- Generate classification reports and confusion matrix plots

> Note: While the script includes logic for `rules_only` and `hybrid` modes, the actual implementation in `detect_emotion` uses only transformer-based logic.

---

## Project Files

```

├── final-transformer_language.py     # Main Streamlit app
├── emotion_detection.py              # Evaluation script
├── emotion_test_data.csv             # Test data
├── audio_embeddings.pt               # Pre-computed background sound embeddings
├── requirements.txt                  # All Python dependencies
├── .env.example                      # Example for API key setup
├── trimmed_sounds/                   # Directory for background sound files
│   ├── forest.wav
│   ├── rain.wav
│   └── ...
```
