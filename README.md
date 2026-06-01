# 🎵 SongMind AI — Streamlit App

ML-powered song recommendations from images & mood text, following the full 9-step ML workflow.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
ANTHROPIC_API_KEY=your_key_here streamlit run app.py
```

Or set the key in your environment first:

```bash
export ANTHROPIC_API_KEY=your_key_here
streamlit run app.py
```

## Features

- 📸 Upload any image — scene, landscape, selfie — Claude analyzes it visually
- 💬 Describe your mood in natural language
- 🎭 Quick mood tag chips (Happy, Chill, Sad, etc.)
- 🧠 Full 9-step ML pipeline visualized in real-time
- 🎧 5 ranked song recommendations with match scores & mood/energy bars

## ML Workflow Steps Shown

1. Define Problem
2. Collect & Understand Data
3. Data Preprocessing
4. Feature Engineering
5. Split the Data
6. Choose & Train Model
7. Evaluate the Model
8. Tune Hyperparameters
9. Deploy & Monitor
