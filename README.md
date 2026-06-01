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
<img width="1903" height="900" alt="image" src="https://github.com/user-attachments/assets/d1188a8f-450b-4348-88ec-dd510814bddb" />

Live demo link:
([Live Demo](https://songmind-ai-jmow2llcavaot4ojrxmqxp.streamlit.app))
## Features

- 📸 Upload any image — scene, landscape, selfie — Claude analyzes it visually
- 💬 Describe your mood in natural language
- 🎭 Quick mood tag chips (Happy, Chill, Sad, etc.)
- 🧠 Full 9-step ML pipeline visualized in real-time
- 🎧 5 ranked song recommendations with match scores & mood/energy bars

## ML Workflow Steps Shown

1. **Define Problem** — Recommend songs by classifying mood from uploaded images and text descriptions.
2. **Collect & Understand Data** — Gather user inputs: scene photos, mood text, and quick-select emotion tags.
3. **Data Preprocessing** — Normalize image colors, encode mood tags (one-hot), and extract text sentiment scores.
4. **Feature Engineering** — Generate CLIP visual embeddings, TF-IDF text vectors, and combine into one feature vector.
5. **Split the Data** — Apply stratified 80/10/10 split across mood categories to preserve class balance.
6. **Choose & Train Model** — Train a gradient-boosted ranker on the mood-song interaction matrix with 5-fold CV.
7. **Evaluate the Model** — Measure NDCG@10 on validation set and check train-val gap for overfitting signs.
8. **Tune Hyperparameters** — Run Bayesian search to optimize regularization (λ) and embedding dimensions.
9. **Deploy & Monitor** — Serve recommendations via Claude's multimodal inference pipeline and watch for mood-drift patterns.
