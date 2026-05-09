import streamlit as st
import numpy as np
from PIL import Image
import io
import json
from google import genai
from google.genai import types

st.set_page_config(page_title="Potato Disease Detector", page_icon="🥔", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&display=swap');

html,body,[class*="css"]{ font-family:'Outfit',sans-serif; }
.stApp{ background:linear-gradient(160deg,#0a1f0a 0%,#0d2b0d 40%,#0a1a0a 100%); color:#e8f5e9; }

.hero-title{ font-size:3rem;font-weight:900;line-height:1.05;
  background:linear-gradient(135deg,#69f0ae,#00e676,#76ff03);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.hero-sub{ color:#4caf50;font-size:0.9rem;font-weight:300;letter-spacing:0.06em; }

.upload-zone{ background:rgba(0,230,118,0.04);border:2px dashed rgba(0,230,118,0.35);
              border-radius:20px;padding:3rem;text-align:center; }
.upload-icon{ font-size:3.5rem; }
.upload-text{ color:#81c784;font-size:1rem;margin-top:0.8rem; }

.healthy-card { background:linear-gradient(135deg,rgba(0,230,118,0.08),rgba(118,255,3,0.06));
                border:2px solid #00e676; border-radius:20px; padding:2rem; text-align:center; }
.eb-card      { background:linear-gradient(135deg,rgba(255,160,0,0.1),rgba(255,100,0,0.08));
                border:2px solid #ff9800; border-radius:20px; padding:2rem; text-align:center; }
.lb-card      { background:linear-gradient(135deg,rgba(244,67,54,0.1),rgba(200,30,30,0.08));
                border:2px solid #f44336; border-radius:20px; padding:2rem; text-align:center; }
.disease-name { font-size:2rem;font-weight:900;margin-top:8px; }
.healthy-name { color:#00e676; }
.eb-name      { color:#ff9800; }
.lb-name      { color:#f44336; }
.confidence   { font-size:3.5rem;font-weight:900;line-height:1; }
.conf-label   { font-size:0.78rem;letter-spacing:0.1em;text-transform:uppercase;opacity:0.7;margin-top:4px; }

.prob-bar-wrap { background:rgba(255,255,255,0.04);border-radius:8px;height:10px;margin:4px 0; }
.info-box { background:rgba(0,0,0,0.3);border:1px solid rgba(0,230,118,0.15);
            border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:10px; }
.info-title { font-weight:700;font-size:1rem;margin-bottom:6px; }
.info-text  { color:#a5d6a7;font-size:0.86rem;line-height:1.6; }

.arch-step { background:rgba(0,230,118,0.05);border-left:3px solid #00e676;
             border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin-bottom:8px; }

div[data-testid="stSidebar"] { background:#071407;border-right:1px solid rgba(0,230,118,0.1); }
.stButton>button { background:linear-gradient(135deg,#00e676,#69f0ae) !important;
    color:#071407 !important;border:none !important;border-radius:50px !important;
    font-weight:700 !important;font-size:1rem !important;padding:0.7rem 2rem !important;width:100%; }
</style>
""", unsafe_allow_html=True)


# ── GEMINI VISION PREDICTION ──────────────────────────────────────
def predict_with_gemini(image: Image.Image):
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # Convert PIL image to bytes for the new SDK
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    prompt = """You are an expert plant pathologist specialising in potato leaf diseases.

Analyse this potato leaf image and classify it into exactly ONE of these three categories:
1. Healthy
2. Early Blight  (caused by Alternaria solani — brown circular spots with yellow halo / target ring pattern)
3. Late Blight   (caused by Phytophthora infestans — dark water-soaked lesions, often with white mould on underside)

Respond ONLY with a valid JSON object. No preamble, no markdown fences, no explanation outside the JSON.

JSON format (confidence values must sum to 1.0):
{
  "prediction": "<Healthy|Early Blight|Late Blight>",
  "confidence": {
    "Healthy": <0.0-1.0>,
    "Early Blight": <0.0-1.0>,
    "Late Blight": <0.0-1.0>
  },
  "reasoning": "<one sentence explanation of key visual features you observed>"
}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    pred_class = result["prediction"]
    conf = result["confidence"]
    total = sum(conf.values())
    probs = {k: v / total for k, v in conf.items()}
    reasoning = result.get("reasoning", "")
    return pred_class, probs, reasoning


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 About the Model")
    st.markdown("---")
    st.markdown("""
**Backend:** Gemini 2.0 Flash (Free)
**Input:** Any resolution RGB
**Classes:** 3
**Model:** gemini-2.0-flash
**Accuracy:** High (vision LLM)
    """)
    st.markdown("---")
    st.markdown("**CNN Architecture (Training Reference):**")
    for layer in ["Conv Block 1 — 32 filters","Conv Block 2 — 64 filters",
                  "Conv Block 3 — 128 filters","Conv Block 4 — 256 filters",
                  "Global Avg Pooling","Dense 256 + Dense 128","Output Softmax (3)"]:
        st.markdown(f'<div class="arch-step" style="font-size:0.8rem">{layer}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Augmentations used in training:**")
    st.markdown("Rotation ±25° · Flip · Zoom ±20° · Shift · Brightness ±20%")

# ── HEADER ────────────────────────────────────────────────────────
col_h, col_stats = st.columns([2, 1])
with col_h:
    st.markdown('<div class="hero-title">🥔 Potato Leaf<br>Disease Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Gemini Vision AI · Potato Disease Classification · TensorFlow/Keras CNN Reference</div>', unsafe_allow_html=True)

with col_stats:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, val, lab in zip([c1,c2,c3],["3","Vision AI","Free"],["Classes","Backend","API Tier"]):
        with col:
            st.markdown(f"""<div style="background:rgba(0,230,118,0.06);border:1px solid rgba(0,230,118,0.2);
                border-radius:12px;padding:0.8rem;text-align:center">
                <div style="font-size:1.5rem;font-weight:900;color:#00e676">{val}</div>
                <div style="font-size:0.72rem;color:#4caf50;text-transform:uppercase;letter-spacing:0.06em">{lab}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── UPLOAD ────────────────────────────────────────────────────────
col_up, col_result = st.columns([1, 1.5])

with col_up:
    st.markdown("#### 📤 Upload Leaf Image")
    uploaded = st.file_uploader("Choose a potato leaf image",
                                 type=["jpg","jpeg","png"],
                                 label_visibility="collapsed")

    if not uploaded:
        st.markdown("""<div class="upload-zone">
            <div class="upload-icon">🌿</div>
            <div class="upload-text">
                Drag & drop or click to upload<br>
                <small style="color:#388e3c">JPG · JPEG · PNG supported</small>
            </div></div>""", unsafe_allow_html=True)

        st.markdown("#### 🎯 Supported Leaf Types")
        for cls, icon, desc in [
            ("Healthy", "🟢", "Uniform green, no spots"),
            ("Early Blight", "🟡", "Brown circular spots, yellow halo"),
            ("Late Blight", "🔴", "Dark water-soaked lesions")
        ]:
            st.markdown(f"""<div class="info-box">
                <div class="info-title">{icon} {cls}</div>
                <div class="info-text">{desc}</div>
            </div>""", unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
        st.markdown(f"<small style='color:#4caf50'>Size: {image.size[0]}×{image.size[1]}px · Mode: {image.mode}</small>", unsafe_allow_html=True)

with col_result:
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.markdown("#### 🧠 AI Vision Analysis")

        with st.spinner("Analysing leaf with Gemini Vision AI..."):
            try:
                pred_class, probs, reasoning = predict_with_gemini(image)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

        confidence = probs[pred_class]
        card_class = {"Healthy":"healthy-card","Early Blight":"eb-card","Late Blight":"lb-card"}[pred_class]
        name_class = {"Healthy":"healthy-name","Early Blight":"eb-name","Late Blight":"lb-name"}[pred_class]
        icon_map   = {"Healthy":"✅","Early Blight":"⚠️","Late Blight":"🚨"}
        conf_color = {"Healthy":"#00e676","Early Blight":"#ff9800","Late Blight":"#f44336"}[pred_class]

        st.markdown(f"""<div class="{card_class}">
            <div style="font-size:2.5rem">{icon_map[pred_class]}</div>
            <div class="disease-name {name_class}">{pred_class}</div>
            <div class="confidence" style="color:{conf_color}">{confidence*100:.1f}%</div>
            <div class="conf-label">Confidence</div>
        </div>""", unsafe_allow_html=True)

        if reasoning:
            st.markdown(f"""<div style="background:rgba(0,0,0,0.25);border-left:3px solid #00e676;
                border-radius:0 10px 10px 0;padding:0.8rem 1rem;margin-top:12px">
                <span style="color:#69f0ae;font-size:0.82rem;font-style:italic">🔍 {reasoning}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**All Class Probabilities:**")
        bar_colors = {"Early Blight":"#ff9800","Late Blight":"#f44336","Healthy":"#00e676"}
        for cls in sorted(["Healthy","Early Blight","Late Blight"], key=lambda x: -probs[x]):
            prob = probs[cls]
            w = int(prob * 100)
            c = bar_colors[cls]
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.88rem;color:#c8e6c9">{cls}</span>
                    <span style="font-weight:700;color:{c}">{prob*100:.1f}%</span>
                </div>
                <div class="prob-bar-wrap">
                    <div style="background:{c};width:{w}%;height:10px;border-radius:8px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if pred_class == "Early Blight":
            st.markdown("""<div class="info-box">
                <div class="info-title">⚠️ About Early Blight</div>
                <div class="info-text">
                    <b>Cause:</b> Fungus Alternaria solani<br>
                    <b>Visual:</b> Brown/dark circular spots with yellow halos (target pattern)<br>
                    <b>Treatment:</b> Copper-based fungicides, remove affected leaves, avoid overhead irrigation<br>
                    <b>Risk:</b> Moderate — can be controlled if caught early
                </div></div>""", unsafe_allow_html=True)
        elif pred_class == "Late Blight":
            st.markdown("""<div class="info-box">
                <div class="info-title">🚨 About Late Blight</div>
                <div class="info-text">
                    <b>Cause:</b> Phytophthora infestans (caused the Irish Potato Famine)<br>
                    <b>Visual:</b> Dark water-soaked lesions, white mould on leaf underside<br>
                    <b>Treatment:</b> Systemic fungicides immediately, remove infected plants<br>
                    <b>Risk:</b> HIGH — can destroy entire crop within days if untreated
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="info-box">
                <div class="info-title">✅ Healthy Leaf</div>
                <div class="info-text">
                    <b>Status:</b> No disease detected<br>
                    <b>Visual:</b> Uniform green colour, no spots or lesions<br>
                    <b>Action:</b> Continue current care regime<br>
                    <b>Tip:</b> Monitor regularly — early detection is key
                </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("#### 📖 How the AI Vision Analysis Works")
        for title, desc in [
            ("1️⃣  Image Upload", "Leaf photo uploaded and passed directly to Gemini Vision"),
            ("2️⃣  Gemini 1.5 Flash", "Free vision model analyses the image with expert prompt"),
            ("3️⃣  Visual Feature Detection", "AI identifies spots, lesions, colour patterns, halo rings"),
            ("4️⃣  Disease Classification", "Compares features against Early Blight, Late Blight, Healthy profiles"),
            ("5️⃣  Confidence Scoring", "Returns probability for each of the 3 classes (sum = 1.0)"),
            ("6️⃣  Reasoning Output", "One-sentence explanation of key visual evidence observed"),
            ("7️⃣  Result Display", "Prediction card with confidence bars and treatment advice"),
            ("8️⃣  Structured JSON", "API returns clean JSON for reliable parsing"),
        ]:
            st.markdown(f"""<div class="arch-step">
                <div style="font-weight:700;color:#69f0ae;font-size:0.9rem">{title}</div>
                <div style="color:#a5d6a7;font-size:0.82rem;margin-top:2px">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── DATA AUGMENTATION VIZ ─────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🔄 Data Augmentation Techniques Used in Training")
aug_cols = st.columns(6)
augs = [("🔃","Rotation","±25°"),("↔️","H-Flip","Left/Right"),
        ("🔍","Zoom","±20%"),("↕️","Shift","±15%"),
        ("☀️","Brightness","±20%"),("🎨","Normalize","÷255")]
for col, (icon, name, val) in zip(aug_cols, augs):
    with col:
        st.markdown(f"""<div style="background:rgba(0,230,118,0.04);border:1px solid rgba(0,230,118,0.15);
            border-radius:12px;padding:0.8rem;text-align:center">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="font-weight:600;font-size:0.85rem;color:#69f0ae;margin-top:4px">{name}</div>
            <div style="color:#4caf50;font-size:0.78rem">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="color:#388e3c;font-size:0.78rem;text-align:center;">Potato Disease Detector · Gemini Vision AI (Free) · 3-Class Classification · Early Blight · Late Blight · Healthy</p>', unsafe_allow_html=True)
