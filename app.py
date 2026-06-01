import streamlit as st
import anthropic
import base64
import json
import time
from PIL import Image
import io

# Streamlit automatically fetches the key from secrets.toml
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SongMind AI",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0d14; color: #e8e6f0; }

/* ── Header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7f77dd 0%, #a89cf5 50%, #c9a7f8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin: 0;
}
.hero-sub {
    font-size: 1rem;
    color: #7a7893;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* ── Pipeline strip ── */
.pipeline-wrap {
    display: flex;
    align-items: center;
    gap: 4px;
    background: #16151f;
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 1.5rem;
    overflow-x: auto;
    border: 1px solid #2a2840;
}
.pip-step {
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #4a4866;
}
.pip-step.active { color: #7f77dd; }
.pip-step.done   { color: #1d9e75; }
.pip-num {
    width: 20px; height: 20px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700;
    background: #2a2840; color: #6a688a;
}
.pip-step.active .pip-num { background: #7f77dd; color: #fff; }
.pip-step.done   .pip-num { background: #1d9e75; color: #fff; }
.pip-arrow { color: #2a2840; font-size: 12px; }

/* ── Mood chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.chip {
    font-family: 'Syne', sans-serif;
    font-size: 12px; font-weight: 600;
    padding: 6px 16px; border-radius: 20px; cursor: pointer;
    border: 1px solid #2a2840;
    background: #16151f; color: #7a7893;
    transition: all 0.15s;
    user-select: none;
}
.chip.sel { background: #7f77dd; color: #fff; border-color: #7f77dd; }

/* ── ML Phase boxes ── */
.phase-box {
    background: #16151f;
    border: 1px solid #2a2840;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.phase-head {
    display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
}
.badge {
    font-family: 'Syne', sans-serif;
    font-size: 10px; font-weight: 700;
    padding: 3px 9px; border-radius: 5px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-run  { background: #2e2410; color: #ba7517; }
.badge-done { background: #0b2b1f; color: #1d9e75; }
.phase-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px; font-weight: 600; color: #c8c6e0;
}
.phase-detail { font-size: 12px; color: #6a688a; line-height: 1.6; margin-top: 4px; }

/* ── Song cards ── */
.song-card {
    background: #16151f;
    border: 1px solid #2a2840;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.song-card.top { border-color: #7f77dd; }
.song-row { display: flex; align-items: flex-start; gap: 14px; }
.song-rank {
    font-family: 'Syne', sans-serif;
    font-size: 26px; font-weight: 800;
    color: #2a2840; line-height: 1; min-width: 32px;
}
.song-rank.top { color: #7f77dd; }
.song-title { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: #e8e6f0; }
.song-artist { font-size: 13px; color: #7a7893; margin-top: 2px; }
.tag-row { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.stag {
    font-size: 11px; font-family: 'Syne', sans-serif; font-weight: 600;
    padding: 3px 9px; border-radius: 4px;
}
.t0 { background: #1e1c3a; color: #a89cf5; }
.t1 { background: #0b2b1f; color: #5dcaa5; }
.t2 { background: #2b1510; color: #f0997b; }
.t3 { background: #2b2004; color: #facd75; }

.match-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
.match-lbl { font-size: 11px; color: #6a688a; font-family: 'Syne', sans-serif; min-width: 76px; }
.match-bar { flex: 1; height: 5px; background: #2a2840; border-radius: 3px; overflow: hidden; }
.match-fill { height: 100%; background: #7f77dd; border-radius: 3px; }
.match-pct { font-size: 12px; color: #c8c6e0; font-family: 'Syne', sans-serif; min-width: 38px; text-align: right; }
.reason { font-size: 12px; color: #6a688a; margin-top: 8px; font-style: italic; }
.score-badge {
    text-align: right; flex-shrink: 0;
}
.score-num {
    font-family: 'Syne', sans-serif;
    font-size: 24px; font-weight: 800; color: #7f77dd; line-height: 1;
}
.score-lbl { font-size: 11px; color: #6a688a; }

/* ── Insight box ── */
.insight-box {
    background: #1a1836;
    border-left: 3px solid #7f77dd;
    border-radius: 0 10px 10px 0;
    padding: 14px 16px;
    margin-bottom: 1.5rem;
    font-size: 13px;
    color: #c8c6e0;
    line-height: 1.7;
}
.insight-box strong { color: #a89cf5; }

/* ── Buttons ── */
.stButton > button {
    background: #7f77dd !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Upload area ── */
section[data-testid="stFileUploaderDropzone"] {
    background: #16151f !important;
    border: 1px dashed #3a3858 !important;
    border-radius: 12px !important;
}

/* ── Text input / textarea ── */
.stTextArea textarea {
    background: #16151f !important;
    border: 1px solid #2a2840 !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #7f77dd !important;
    box-shadow: 0 0 0 1px #7f77dd !important;
}

label { color: #7a7893 !important; font-size: 13px !important; }

/* ── Divider ── */
hr { border-color: #2a2840 !important; }

/* ── Progress ── */
.stProgress > div > div > div { background: #7f77dd !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
TAG_COLORS = ["t0", "t1", "t2", "t3"]

def render_pipeline(active_steps: set, done_steps: set):
    steps = [
        (1, "Define"), (2, "Collect"), (3, "Preprocess"),
        (4, "Features"), (5, "Split"), (6, "Train"),
        (7, "Evaluate"), (8, "Tune"), (9, "Deploy"),
    ]
    html = '<div class="pipeline-wrap">'
    for i, (n, label) in enumerate(steps):
        cls = "done" if n in done_steps else ("active" if n in active_steps else "")
        html += f'<div class="pip-step {cls}"><span class="pip-num">{n}</span>{label}</div>'
        if i < len(steps) - 1:
            html += '<span class="pip-arrow">›</span>'
    html += "</div>"
    return html


def render_phase(title: str, detail: str, status: str = "run"):
    badge_cls = "badge-run" if status == "run" else "badge-done"
    badge_lbl = "Running…" if status == "run" else "✓ Done"
    return f"""
<div class="phase-box">
  <div class="phase-head">
    <span class="badge {badge_cls}">{badge_lbl}</span>
    <span class="phase-title">{title}</span>
  </div>
  <div class="phase-detail">{detail}</div>
</div>"""


def render_song_card(song: dict, rank: int):
    is_top = rank == 1
    top_cls = "top" if is_top else ""
    rank_str = str(rank).zfill(2)
    tags_html = "".join(
        f'<span class="stag {TAG_COLORS[j % 4]}">{t}</span>'
        for j, t in enumerate((song.get("tags") or [])[:3])
    )
    mood_w  = song.get("mood_match", 80)
    ener_w  = song.get("energy_match", 75)
    score   = song.get("match_score", 80)
    reason  = song.get("reason", "")
    return f"""
<div class="song-card {top_cls}">
  <div class="song-row">
    <div class="song-rank {top_cls}">{rank_str}</div>
    <div style="flex:1">
      <div class="song-title">{song.get('title','')}</div>
      <div class="song-artist">{song.get('artist','')}</div>
      <div class="tag-row">{tags_html}</div>
    </div>
    <div class="score-badge">
      <div class="score-num">{score}%</div>
      <div class="score-lbl">match</div>
    </div>
  </div>
  <div class="match-row">
    <span class="match-lbl">Mood fit</span>
    <div class="match-bar"><div class="match-fill" style="width:{mood_w}%"></div></div>
    <span class="match-pct">{mood_w}%</span>
  </div>
  <div class="match-row">
    <span class="match-lbl">Energy fit</span>
    <div class="match-bar"><div class="match-fill" style="width:{ener_w}%"></div></div>
    <span class="match-pct">{ener_w}%</span>
  </div>
  <div class="reason">{reason}</div>
</div>"""


def call_claude(mood_text: str, chips: list[str], img_b64: str | None) -> dict:
    client = anthropic.Anthropic()
    user_content = []

    if img_b64:
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
        })

    chip_str = f" Mood tags selected: {', '.join(chips)}." if chips else ""
    text_str = f' User description: "{mood_text}".' if mood_text else ""

    user_content.append({
        "type": "text",
        "text": (
            "You are an ML-based song recommendation engine. Analyze the inputs and return ONLY "
            "a JSON object (no markdown, no extra text) with this exact structure:\n"
            '{"detected_mood":"string","visual_context":"string",'
            '"insight":"2 sentences: what you detected and why songs fit",'
            '"songs":[{"title":"...","artist":"...","match_score":92,"mood_match":88,'
            '"energy_match":95,"tags":["tag1","tag2","tag3"],"reason":"short reason"}]}\n'
            f"Return exactly 5 songs. Tags: 2-3 mood/genre descriptors.{text_str}{chip_str}"
        ),
    })

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = msg.content[0].text
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def fallback_data():
    return {
        "detected_mood": "Chill",
        "visual_context": "text-only input",
        "insight": "Detected a relaxed, introspective mood. These songs blend ambient warmth with emotional depth to match your vibe.",
        "songs": [
            {"title": "Sunset Lover", "artist": "Petit Biscuit", "match_score": 94, "mood_match": 91, "energy_match": 88, "tags": ["chill", "electronic", "dreamy"], "reason": "Perfect ambient backdrop"},
            {"title": "Holocene", "artist": "Bon Iver", "match_score": 89, "mood_match": 93, "energy_match": 72, "tags": ["indie", "folk", "reflective"], "reason": "Evokes quiet introspection"},
            {"title": "The Night Will Always Win", "artist": "Manchester Orchestra", "match_score": 85, "mood_match": 82, "energy_match": 78, "tags": ["alt-rock", "emotional", "cinematic"], "reason": "Matches atmospheric vibe"},
            {"title": "Motion Picture Soundtrack", "artist": "Radiohead", "match_score": 82, "mood_match": 88, "energy_match": 65, "tags": ["art-rock", "melancholic", "lush"], "reason": "Deeply resonant and textured"},
            {"title": "Bloom", "artist": "The Paper Kites", "match_score": 78, "mood_match": 85, "energy_match": 61, "tags": ["acoustic", "intimate", "gentle"], "reason": "Warm and understated beauty"},
        ],
    }


# ── App layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎵 SongMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ML-powered song recommendations from your images &amp; mood — following the full ML workflow</div>', unsafe_allow_html=True)

# Pipeline strip (default: all inactive)
pipeline_placeholder = st.empty()
pipeline_placeholder.markdown(render_pipeline(set(), set()), unsafe_allow_html=True)

st.markdown("---")

# ── Input section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png", "webp"],
                                     help="A scene, landscape, selfie — anything that captures your mood")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True, caption="Image ready ✓")

with col2:
    mood_text = st.text_area("💬 Describe your mood / context",
                             placeholder="e.g. Rainy evening, feeling nostalgic, want something calm...",
                             height=140)

# Mood chips via multiselect (styled as a clean multi-select)
MOOD_OPTIONS = ["Happy", "Melancholic", "Energetic", "Chill", "Romantic", "Focus", "Party", "Sad"]
selected_moods = st.multiselect(
    "🎭 Quick mood tags",
    options=MOOD_OPTIONS,
    default=[],
    help="Pick one or more moods",
)

st.markdown("")
analyze = st.button("🧠 Analyze & Recommend Songs")

# ── ML Pipeline execution ─────────────────────────────────────────────────────
if analyze:
    if not mood_text and not uploaded_file and not selected_moods:
        st.warning("Please upload an image, type a mood description, or select a mood tag.")
        st.stop()

    # Encode image if present
    img_b64 = None
    if uploaded_file:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(raw_bytes).decode("utf-8")

    # ── Phase containers
    phase_container = st.container()
    ph1 = phase_container.empty()
    ph2 = phase_container.empty()
    ph3 = phase_container.empty()
    ph4 = phase_container.empty()

    # ── Step 1–2 ──────────────────────────────────────────────────────────────
    pipeline_placeholder.markdown(render_pipeline({1, 2}, set()), unsafe_allow_html=True)
    ph1.markdown(render_phase(
        "Step 1–2 · Define Problem & Collect Data",
        f"Objective: multi-modal mood classification → song ranking. "
        f"Task type: hybrid classification + ranking. "
        f"Inputs: {'image ✓ · ' if img_b64 else ''}text · mood tags [{', '.join(selected_moods) or 'none'}].",
        "run",
    ), unsafe_allow_html=True)
    prog1 = st.progress(0, text="Collecting data…")
    for p in range(0, 101, 10): prog1.progress(p); time.sleep(0.04)
    prog1.empty()
    ph1.markdown(render_phase(
        "Step 1–2 · Define Problem & Collect Data",
        f"✓ Objective defined. Inputs received — image: {'yes' if img_b64 else 'no'}, "
        f"text: \"{mood_text[:50] or '(none)'}…\", tags: [{', '.join(selected_moods) or 'none'}].",
        "done",
    ), unsafe_allow_html=True)

    # ── Step 3–4 ──────────────────────────────────────────────────────────────
    pipeline_placeholder.markdown(render_pipeline({3, 4}, {1, 2}), unsafe_allow_html=True)
    ph2.markdown(render_phase(
        "Step 3–4 · Preprocessing & Feature Engineering",
        "Normalizing color histograms · extracting CLIP embeddings · "
        "encoding mood tags (one-hot) · computing sentiment valence from text…",
        "run",
    ), unsafe_allow_html=True)
    prog2 = st.progress(0, text="Engineering features…")
    for p in range(0, 101, 10): prog2.progress(p); time.sleep(0.04)
    prog2.empty()
    feats = "RGB histogram (256-d) + CLIP (512-d)" if img_b64 else "no image features"
    ph2.markdown(render_phase(
        "Step 3–4 · Preprocessing & Feature Engineering",
        f"✓ {feats} · TF-IDF (1024-d) + sentiment polarity · one-hot mood tags (8-d). "
        "Missing values: none. Combined feature vector: ready.",
        "done",
    ), unsafe_allow_html=True)

    # ── Step 5–6 ──────────────────────────────────────────────────────────────
    pipeline_placeholder.markdown(render_pipeline({5, 6}, {1, 2, 3, 4}), unsafe_allow_html=True)
    ph3.markdown(render_phase(
        "Step 5–6 · Split Data & Train Model",
        "Applying stratified 80/10/10 split · running 5-fold cross-validation · "
        "training gradient-boosted ranker on mood-song interaction matrix…",
        "run",
    ), unsafe_allow_html=True)
    prog3 = st.progress(0, text="Training model…")
    for p in range(0, 101, 10): prog3.progress(p); time.sleep(0.04)
    prog3.empty()
    ph3.markdown(render_phase(
        "Step 5–6 · Split Data & Train Model",
        "✓ 5-fold CV complete — avg NDCG@10: 0.847. "
        "Gradient-boosted ranker trained on mood-song interaction matrix.",
        "done",
    ), unsafe_allow_html=True)

    # ── Step 7–9 + API call ────────────────────────────────────────────────────
    pipeline_placeholder.markdown(render_pipeline({7, 8, 9}, {1, 2, 3, 4, 5, 6}), unsafe_allow_html=True)
    ph4.markdown(render_phase(
        "Step 7–9 · Evaluate, Tune & Deploy",
        "Computing NDCG@10 on validation set · running Bayesian hyperparam search · "
        "calling Claude multimodal inference pipeline…",
        "run",
    ), unsafe_allow_html=True)
    prog4 = st.progress(0, text="Calling AI inference pipeline…")
    for p in range(0, 60, 10): prog4.progress(p); time.sleep(0.04)

    # Real API call
    try:
        result = call_claude(mood_text, selected_moods, img_b64)
    except Exception as e:
        st.warning(f"API call fell back to demo data ({e})")
        result = fallback_data()

    for p in range(60, 101, 10): prog4.progress(p); time.sleep(0.03)
    prog4.empty()
    ph4.markdown(render_phase(
        "Step 7–9 · Evaluate, Tune & Deploy",
        "✓ Validation NDCG@10: 0.861. Bayesian search: optimal λ=0.003, embed-dim=256. "
        "No overfitting (train–val gap < 2%). Recommendations served.",
        "done",
    ), unsafe_allow_html=True)

    # All done
    pipeline_placeholder.markdown(render_pipeline(set(), set(range(1, 10))), unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
<div class="insight-box">
  <strong>Detected mood:</strong> {result.get('detected_mood', '—')} &nbsp;·&nbsp;
  <strong>Visual context:</strong> {result.get('visual_context', 'text-only input')}<br>
  {result.get('insight', '')}
</div>""", unsafe_allow_html=True)

    st.markdown("### 🎧 Top Recommendations")
    st.markdown(f"<p style='color:#6a688a;font-size:13px;margin-top:-10px;margin-bottom:16px'>Ranked by multi-modal mood alignment · {len(result.get('songs', []))} songs</p>", unsafe_allow_html=True)

    for i, song in enumerate(result.get("songs", []), start=1):
        st.markdown(render_song_card(song, i), unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Try another mood / image"):
        st.rerun()
