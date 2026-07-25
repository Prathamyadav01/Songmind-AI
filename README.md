# 🎵 SongMind AI — Multimodal Emotion-Based Song Recommender

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![DeepFace](https://img.shields.io/badge/DeepFace-Vision-green.svg)](https://github.com/serengil/deepface)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

**SongMind AI** is an end-to-end multimodal recommendation system that analyzes real-time facial expressions from a webcam feed alongside contextual user chat messages to serve personalized song recommendations. 

Instead of relying solely on past listening history, SongMind AI captures the user's **immediate emotional state** using a combination of Computer Vision, Zero-Shot Natural Language Processing, and Cosine Similarity vector matching.

---

## 🏗️ System Architecture

```text
┌───────────────────────────┐      ┌──────────────────────────┐
│  Live Camera / Selfie     │      │   User Chat / Sentiment  │
└─────────────┬─────────────┘      └────────────┬─────────────┘
              │                                 │
              ▼                                 ▼
┌───────────────────────────┐      ┌──────────────────────────┐
│  DeepFace (CNN Engine)    │      │ DistilBERT Zero-Shot NLP │
│  Detects Visual Emotion   │      │ Interprets Text Context  │
└─────────────┬─────────────┘      └────────────┬─────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │     Multimodal Feature Fusion   │
              │ Calculates Target (V_user, E_user)│
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Cosine Similarity Vector Engine │
              │   Ranks Spotify Dataset Songs   │
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ 🎧 Top Ranked Recommendations   │
              └─────────────────────────────────┘
