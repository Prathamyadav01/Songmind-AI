import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from deepface import DeepFace
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. PAGE CONFIGURATION & SLEEK CSS STYLING ─────────────────────────────────
st.set_page_config(
    page_title="SongMind AI",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b0b12;
    color: #e2e8f0;
}

/* Hero Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Glassmorphism Card Containers */
.glass-card {
    background: rgba(22, 21, 35, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}

/* Status Badges */
.status-box {
    background: #141322;
    border-left: 4px solid #a855f7;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(168, 85, 247, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(168, 85, 247, 0.5) !important;
}

/* Inputs styling */
.stTextInput input {
    background: #161523 !important;
    border: 1px solid #2e2b44 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
.stTextInput input:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 1px #a855f7 !important;
}

hr {
    border-color: #1e1c33 !important;
}
</style>
""", unsafe_allow_html=True)


# ── 2. LOAD DATASET & LIGHTWEIGHT NLP MODEL ──────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv('spotify_data_combined.csv')

@st.cache_resource
def load_nlp_pipeline():
    # Fast & memory-friendly Zero-Shot Classifier
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

df = load_dataset()
nlp_classifier = load_nlp_pipeline()


# ── 3. OPTIMIZED ZERO-SHOT EMOTION MAPPINGS ───────────────────────────────────
CANDIDATE_LABELS = [
    "annoyed and stressed about work or duties",
    "sad, tired, and melancholic",
    "happy, cheerful, and excited",
    "calm, peaceful, and relaxed",
    "romantic and warm",
    "energetic and ready to party"
]

ZERO_SHOT_MAP = {
    "annoyed and stressed about work or duties": {'valence': 0.15, 'energy': 0.85, 'label': 'anger'},
    "sad, tired, and melancholic":               {'valence': 0.15, 'energy': 0.25, 'label': 'sadness'},
    "happy, cheerful, and excited":             {'valence': 0.85, 'energy': 0.80, 'label': 'joy'},
    "calm, peaceful, and relaxed":              {'valence': 0.50, 'energy': 0.30, 'label': 'neutral'},
    "romantic and warm":                        {'valence': 0.80, 'energy': 0.45, 'label': 'love'},
    "energetic and ready to party":             {'valence': 0.75, 'energy': 0.90, 'label': 'surprise'},
}

EMOTION_VECTORS = {
    'happy':    {'valence': 0.85, 'energy': 0.80},
    'sad':      {'valence': 0.15, 'energy': 0.25},
    'angry':    {'valence': 0.10, 'energy': 0.90},
    'disgust':  {'valence': 0.20, 'energy': 0.85},
    'neutral':  {'valence': 0.50, 'energy': 0.40},
    'surprise': {'valence': 0.70, 'energy': 0.85},
    'fear':     {'valence': 0.20, 'energy': 0.75},
}


# ── 4. HEADER ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎵 SongMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Multimodal Recommendation Engine • Computer Vision + Zero-Shot NLP</div>', unsafe_allow_html=True)


# ── 5. INPUT SECTION ──────────────────────────────────────────────────────────
st.markdown("### 📸 Step 1: Capture Facial Expression")
picture = st.camera_input("Take a quick webcam selfie")

st.markdown("### 💬 Step 2: Chat Your Mood")
user_text = st.text_input(
    "How is your day going?",
    placeholder="e.g., 'Got work to do on the weekend by my boss...', 'Feeling relaxed with coffee'",
)

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🧠 Analyze Mood & Generate Playlist")


# ── 6. MULTIMODAL INFERENCE & RECOMMENDATION ─────────────────────────────────
if analyze:
    if not picture and not user_text:
        st.warning("⚠️ Please snap a picture or enter a chat message to begin.")
    else:
        v_scores, e_scores = [], []
        
        # --- A. COMPUTER VISION (DeepFace) ---
        if picture:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(picture.getvalue())
                temp_path = temp_file.name

            try:
                cv_result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
                face_emotion = cv_result[0]['dominant_emotion'].lower()
                
                face_vec = EMOTION_VECTORS.get(face_emotion, EMOTION_VECTORS['neutral'])
                v_scores.append(face_vec['valence'])
                e_scores.append(face_vec['energy'])
                
                st.markdown(
                    f'<div class="status-box">📸 <b>Facial Expression:</b> Detected <span style="color:#a855f7;">{face_emotion.capitalize()}</span> '
                    f'(Valence: <code>{face_vec["valence"]}</code>, Energy: <code>{face_vec["energy"]}</code>)</div>',
                    unsafe_allow_html=True
                )
            except Exception as ex:
                st.error(f"Error analyzing image: {ex}")
            finally:
                os.remove(temp_path)

        # --- B. NATURAL LANGUAGE PROCESSING (Zero-Shot) ---
        if user_text:
            try:
                result = nlp_classifier(user_text, candidate_labels=CANDIDATE_LABELS)
                top_label = result['labels'][0]
                
                vec = ZERO_SHOT_MAP[top_label]
                v_scores.append(vec['valence'])
                e_scores.append(vec['energy'])
                
                st.markdown(
                    f'<div class="status-box">💬 <b>Text Sentiment:</b> Interpreted <span style="color:#6366f1;">{vec["label"].capitalize()}</span> '
                    f'(Valence: <code>{vec["valence"]}</code>, Energy: <code>{vec["energy"]}</code>)</div>',
                    unsafe_allow_html=True
                )
            except Exception as ex:
                st.error(f"Error analyzing text sentiment: {ex}")

        # --- C. MULTIMODAL FUSION ---
        target_valence = float(np.mean(v_scores))
        target_energy = float(np.mean(e_scores))
        
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].metric("Target Valence (Happiness)", f"{target_valence:.2f}")
        cols[1].metric("Target Energy (Intensity)", f"{target_energy:.2f}")

        # --- D. COSINE SIMILARITY MATRIX MATCHING ---
        user_vector = np.array([[target_valence, target_energy]])
        song_vectors = df[['valence', 'energy']].values

        similarities = cosine_similarity(user_vector, song_vectors)[0]
        
        df_copy = df.copy()
        df_copy['match_score'] = (similarities * 100).round(1)
        top_recommendations = df_copy.sort_values(by='match_score', ascending=False).head(5)

        # --- E. SLEEK RECOMMENDATION CARDS ---
        st.markdown("---")
        st.markdown("### 🎧 Top 5 Personal Recommendations")

        for idx, row in top_recommendations.iterrows():
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
                    <img src="{row['Cover Image']}" style="width: 68px; height: 68px; border-radius: 10px; object-fit: cover;">
                    <div style="flex-grow: 1;">
                        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{row['Song Name']}</div>
                        <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 2px;">{row['Artists']}</div>
                        <div style="font-size: 0.75rem; color: #a855f7; margin-top: 6px;">🏷️ {row['mood']} • Valence: {row['valence']} • Energy: {row['energy']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: #a855f7;">{row['match_score']}%</div>
                        <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">Match</div>
                        <a href="{row['Spotify URL']}" target="_blank" style="display: inline-block; margin-top: 6px; font-size: 0.8rem; color: #6366f1; text-decoration: none; font-weight: 600;">▶ Listen</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
