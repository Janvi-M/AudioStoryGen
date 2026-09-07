# AI Audio Story Generator

An AI-powered storytelling application that generates narrated stories from user prompts, adapts narration based on detected emotions, and adds contextually relevant background sounds.

## Features

- **Story Generation:** Uses Google Gemini to generate stories from user prompts.
- **Voice Cloning:** Uses XTTSv2 to generate narration using a short reference voice sample.
- **Emotion Detection:** Uses Transformer-based emotion classification to detect the emotional tone of individual sentences.
- **Emotion-Based Modulation:** Adjusts narration energy based on the detected emotion.
- **Semantic Sound Retrieval:** Uses CLAP (Contrastive Language-Audio Pretraining) embeddings to match story sentences with relevant background sounds.
- **Interactive UI:** Built with Streamlit with configurable sound similarity threshold, background volume, and sentence pauses.

## Pipeline

```text
User Story Prompt
       ↓
Gemini Story Generation
       ↓
Sentence-Level Emotion Detection
       ↓
XTTSv2 Voice Synthesis
       ↓
CLAP-Based Background Sound Retrieval
       ↓
Audio Mixing
       ↓
Final Narrated Story
```

## Technologies

**Python · Streamlit · Gemini · XTTSv2 · Transformers · CLAP · PyTorch · Librosa · NLTK**

## Project Structure

```text
├── 02_final-transformer_language.py
├── 01_emotion_detection.py
├── emotion_test_data.csv
├── trimmed sounds/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Janvi-M/AudioStoryGen.git
cd AudioStoryGen
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Do not commit the `.env` file.

### 5. Download NLTK data

```python
import nltk
nltk.download("punkt")
```

### 6. Run the application

```bash
streamlit run final-transformer_language.py
```

## Emotion Detection Evaluation

The emotion detection component can be evaluated using the provided test dataset:

```bash
python emotion_detection.py
```

The evaluation compares predicted emotions against the labeled test data and generates classification metrics and a confusion matrix.

## Notes

- XTTSv2 requires a short reference voice sample for voice cloning.
- The application uses a curated set of background sound effects stored in `trimmed sounds/`.
- CLAP audio embeddings are required for background sound matching. If the precomputed embeddings are not available, they need to be generated separately.
- Model downloads may require an internet connection and additional system resources.

## Limitations

- The project is a research/portfolio prototype rather than a production-ready audio generation system.
- Emotion classification is based on Transformer predictions and may not capture nuanced emotional context.
- Background sound retrieval depends on the size and diversity of the curated sound library.
- Voice synthesis and model inference can be computationally intensive.

## Research

This project was developed as part of research on immersive AI-based audio storytelling using semantic sound retrieval and emotion-driven voice synthesis.
